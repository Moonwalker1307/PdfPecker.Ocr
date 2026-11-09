using PdfPecker.Ocr.Input;

namespace PdfPecker.Ocr.Pdf;

/// <summary>
/// Makes PDFs and images searchable: reads their pages with a recognizer and puts the text on them as an invisible
/// layer, with <see cref="SearchableLayer"/> and <see cref="PdfStamper"/>.
/// </summary>
internal static class SearchablePdf
{
    /// <summary>The most image bytes converted to PDF in one go, so a long list of scans does not sit in memory twice.</summary>
    internal const long MaxGroupBytes = 64L * 1024 * 1024;

    /// <summary>The most images converted to PDF in one go.</summary>
    internal const int MaxGroupImages = 500;

    /// <summary>Adds a text layer to the chosen pages of a PDF that have no text (or to all, with <see cref="OcrPageMode.Force"/>).</summary>
    public static byte[] MakeSearchable(byte[] pdf, OcrOptions? options, IPageRecognizer recognizer, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(pdf);
        options ??= new OcrOptions();
        options.Validate();
        var source = PdfPageSource.Open(pdf, options, cancellationToken);
        if (source.Pages.Any(i => !source.HasText(i)))
        {
            // PdfStamper refuses a signed PDF only when it stamps, after every page has been rendered and read. A stamp
            // of no pages finds that out first, by the stamper's own rule, for the cost of one rewrite of the file.
            PdfStamper.Stamp(pdf, SearchableLayer.Write([new LayerPage(null, 1, 1)], cancellationToken),
                new PdfStampOptions { Password = options.Password, Pages = _ => false }, cancellationToken);
        }

        var read = new Dictionary<int, OcrPage>();
        foreach (int index in source.Pages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (source.HasText(index)) continue;
            var rendered = source.Render(index, cancellationToken);
            using (rendered.Pixels)
                read[index] = recognizer.Recognize(rendered.Pixels, index, rendered.Dpi, options, cancellationToken);
        }
        // Nothing to add: the PDF is returned as it was, as a copy so the caller's array and the result stay separate.
        if (read.Count == 0) return (byte[])pdf.Clone();

        int last = read.Keys.Max();
        var layer = new LayerPage[last + 1];
        for (int i = 0; i <= last; i++)
        {
            var (w, h) = source.PageSize(i);
            layer[i] = new LayerPage(read.GetValueOrDefault(i), w, h);
        }
        var stamp = SearchableLayer.Write(layer, cancellationToken);
        return PdfStamper.Stamp(pdf, stamp, StampOptions(options.Password, read), cancellationToken);
    }

    /// <summary>Makes a searchable PDF of images, one page per image and per page of a multi-page TIFF.</summary>
    public static byte[] FromImages(IEnumerable<byte[]> images, OcrOptions? options, IPageRecognizer recognizer, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(images);
        options ??= new OcrOptions();
        options.Validate();
        var parts = new List<byte[]>();
        int pageIndex = 0;
        foreach (var group in Groups(images, options.Dpi))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var data = group.Images;
            // The pages of the result are sized by the images' resolution; the pixels come from a second PDF at one
            // PDF pixel per image pixel, which the renderer reproduces exactly (see ImageDecoder).
            var output = ImageToPdf.Convert(data, null, new ImageToPdfOptions { FitPageToImage = true, Dpi = group.Dpi }, cancellationToken);
            var sizes = PdfRenderer.Open(output);
            var native = PdfRenderer.Open(ImageDecoder.ToNativePdf(data, cancellationToken));
            var layer = new LayerPage[data.Count];
            for (int i = 0; i < data.Count; i++)
            {
                var decoded = ImageDecoder.Render(native, i, ImageInfo.Read(data[i]), group.Dpi, options.MaxPixels, cancellationToken);
                OcrPage page;
                using (decoded.Pixels)
                    page = recognizer.Recognize(decoded.Pixels, pageIndex++, decoded.Dpi, options, cancellationToken);
                var (w, h) = sizes.GetPageSize(i);
                // A page is at least 10 points on a side, so a smaller image sits in its top-left corner: the words are
                // placed by the image's resolution from that corner rather than stretched over the page.
                layer[i] = new LayerPage(page, w, h, ScaleByDpi: true);
            }
            var stamp = SearchableLayer.Write(layer, cancellationToken);
            parts.Add(PdfStamper.Stamp(output, stamp, StampOptions(null, null), cancellationToken));
        }
        if (parts.Count == 0) throw new ArgumentException("There are no images.", nameof(images));
        return parts.Count == 1 ? parts[0] : PdfMerger.Merge(parts, null, cancellationToken);
    }

    /// <summary>
    /// The stamp options for a text layer: stamp page i on page i at the page's own size, beneath the content, so the
    /// page's own drawing is left untouched rather than wrapped in a saved graphics state.
    /// </summary>
    private static PdfStampOptions StampOptions(string? password, Dictionary<int, OcrPage>? pages) => new()
    {
        MatchPages = true,
        Underlay = true,
        Password = password,
        Pages = pages == null ? null : pages.ContainsKey,
    };

    /// <summary>One <see cref="ImageToPdf"/> call's worth of images: consecutive images of one resolution.</summary>
    internal sealed record ImageGroup(float Dpi, List<byte[]> Images);

    /// <summary>
    /// Splits the images into groups of one resolution, within <see cref="MaxGroupBytes"/> and <see cref="MaxGroupImages"/>,
    /// with each page of a multi-page TIFF as an image of its own.
    /// </summary>
    internal static IEnumerable<ImageGroup> Groups(IEnumerable<byte[]> images, float fallbackDpi)
    {
        ImageGroup? group = null;
        long bytes = 0;
        int total = 0;
        foreach (var image in images)
        {
            if (image == null) throw new ArgumentNullException(nameof(images), "An image is null.");
            var info = ImageInfo.Read(image);
            if (info.Format == ImageFormat.Unknown)
                throw new InvalidDataException(FormattableString.Invariant($"Image {total + 1} is not a PNG, JPEG, GIF, WebP, BMP or TIFF image."));
            long left = ImageInfo.TiffFramesBudget(image.Length);
            for (int frame = 0; frame < info.FrameCount; frame++)
            {
                if (++total > ImageToPdf.MaxImages)
                    throw new ArgumentException(FormattableString.Invariant($"At most {ImageToPdf.MaxImages} images and TIFF pages can be read at once."), nameof(images));
                float dpi = info.Format == ImageFormat.Tiff && frame > 0 ? ImageInfo.TiffDpi(image, info.TiffFrames[frame]) : info.Dpi;
                if (dpi <= 0) dpi = fallbackDpi;
                // Each page of a multi-page TIFF becomes a file of its own with only the data it refers to.
                byte[] data = info.Format == ImageFormat.Tiff ? ImageInfo.TiffFrame(image, info, frame, left) : image;
                left -= data.Length;
                if (group != null && (group.Dpi != dpi || bytes + data.Length > MaxGroupBytes || group.Images.Count >= MaxGroupImages))
                {
                    yield return group;
                    group = null;
                }
                if (group == null)
                {
                    group = new ImageGroup(dpi, new List<byte[]>());
                    bytes = 0;
                }
                group.Images.Add(data);
                bytes += data.Length;
            }
        }
        if (group != null) yield return group;
    }
}
