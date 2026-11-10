namespace PdfPecker.Ocr.Input;

/// <summary>An image decoded for recognition.</summary>
/// <param name="Pixels">The pixels; the receiver disposes them.</param>
/// <param name="Dpi">
/// The resolution of <paramref name="Pixels"/>: the one the image records (or the fallback), lowered in proportion when
/// the image had more pixels than allowed and was read smaller.
/// </param>
internal readonly record struct DecodedImage(PagePixels Pixels, float Dpi);

/// <summary>
/// Decodes image files (PNG, JPEG, GIF, WebP, BMP and TIFF) to pixels with PdfPecker's <see cref="ImageToPixels"/>,
/// which gives the pixels <see cref="ImageToPdf"/> embeds and <see cref="PdfRenderer"/> draws at one pixel per image
/// pixel, without making a PDF of them.
/// </summary>
/// <remarks>
/// <para>
/// An image with more pixels than allowed still goes the long way: it becomes a page of a PDF with
/// <see cref="ImageToPdf"/>, and the renderer draws it smaller, within the budget. The pages are made at 96 dpi because
/// there one image pixel is exactly one CSS pixel, so the page is exactly the image's size. At other resolutions the
/// page size goes through CSS pixels with three decimals and the image comes back shifted by a fraction of a pixel.
/// The PDF is written without compression: it lives only until the page is rendered.
/// </para>
/// </remarks>
internal static class ImageDecoder
{
    /// <summary>The resolution at which <see cref="ImageToPdf"/> makes one PDF pixel of each image pixel.</summary>
    public const float NativeDpi = 96;

    /// <summary>
    /// Decodes the first page of an image file. Its resolution is the one it records, else <paramref name="fallbackDpi"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">The data is not an image PdfPecker reads, or it is damaged.</exception>
    public static DecodedImage Decode(byte[] image, float fallbackDpi, long maxPixels, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(image);
        var info = ImageInfo.Read(image);
        if (info.Format == ImageFormat.Unknown)
            throw new InvalidDataException("The data is not a PNG, JPEG, GIF, WebP, BMP or TIFF image.");
        return Decode(image, info, fallbackDpi, maxPixels, cancellationToken);
    }

    /// <summary>
    /// Decodes an image of one page that <paramref name="info"/> describes (a page of a multi-page TIFF is first made a
    /// file of its own with <see cref="ImageInfo.TiffFrame"/>). Its resolution is the one it records, else
    /// <paramref name="fallbackDpi"/>, lowered in proportion when it has more than <paramref name="maxPixels"/> pixels
    /// and is drawn smaller.
    /// </summary>
    /// <exception cref="InvalidDataException">The data is not an image PdfPecker reads, or it is damaged.</exception>
    public static DecodedImage Decode(byte[] image, ImageInfo info, float fallbackDpi, long maxPixels, CancellationToken cancellationToken)
    {
        long limit = Math.Min(maxPixels, ImageToPixelsOptions.MaxPixelLimit);
        if ((long)info.Width * info.Height > limit)
        {
            var renderer = PdfRenderer.Open(ToNativePdf(new[] { image }, cancellationToken));
            return Render(renderer, 0, info, fallbackDpi, maxPixels, cancellationToken);
        }
        float imageDpi = info.Dpi > 0 ? info.Dpi : fallbackDpi;
        var options = new ImageToPixelsOptions { MaxPixels = limit };
        var decoded = ImageToPixels.Convert(image, options, cancellationToken);
        if (decoded.Format == ImagePixelFormat.Rgba)
        {
            // Transparent pixels are flattened onto white with the renderer's arithmetic, as a rendered page has them.
            options.Format = ImagePixelFormat.Rgb;
            decoded = ImageToPixels.Convert(image, options, cancellationToken);
        }
        var pixels = decoded.Format == ImagePixelFormat.Gray
            ? PagePixels.FromGray(decoded.Pixels, decoded.Width, decoded.Height)
            : PagePixels.FromRgbOrGray(decoded.Pixels, decoded.Width, decoded.Height);
        return new DecodedImage(pixels, imageDpi);
    }

    /// <summary>A PDF with one page per image, each page exactly its image's size in pixels at <see cref="NativeDpi"/>.</summary>
    /// <exception cref="InvalidDataException">An image is not one PdfPecker reads, or it is damaged.</exception>
    public static byte[] ToNativePdf(IEnumerable<byte[]> images, CancellationToken cancellationToken)
        => ImageToPdf.Convert(images, new PdfOptions { Compress = false }, new ImageToPdfOptions { FitPageToImage = true, Dpi = NativeDpi },
            cancellationToken);

    /// <summary>
    /// Renders page <paramref name="pageIndex"/> of a PDF made by <see cref="ToNativePdf"/> at one pixel per image pixel,
    /// or smaller within <paramref name="maxPixels"/>, and gives it the image's resolution (from <paramref name="info"/>,
    /// else <paramref name="fallbackDpi"/>) scaled the same way.
    /// </summary>
    public static DecodedImage Render(PdfRenderer renderer, int pageIndex, ImageInfo info, float fallbackDpi, long maxPixels, CancellationToken cancellationToken)
    {
        float imageDpi = info.Dpi > 0 ? info.Dpi : fallbackDpi;
        int w = info.Width, h = info.Height;
        // HTML pages are at least 10 points on a side, so an image narrower or lower than that sits in the top-left
        // corner of a larger page, which the renderer rounds and stretches by a fraction of a pixel. At four times 72
        // dpi such a page is a whole number of pixels and each image pixel a block of 3 by 3 pixels whose centre is
        // the image pixel itself, so those centres give the image back unchanged.
        // The header's size is only a claim: PdfPecker decodes a BMP that wraps a PNG or JPEG, or a WebP frame, at the
        // size of the data, so the path is taken only when the page the renderer has is the one the header gives.
        if (w > 0 && h > 0 && (w < MinPagePixels || h < MinPagePixels) && 16L * Math.Max(w, MinPagePixels) * Math.Max(h, MinPagePixels) <= Math.Min(maxPixels, PageRaster.MaxRenderPixels)
            && PageMatches(renderer.GetPageSize(pageIndex), w, h))
            return new DecodedImage(SmallImage(renderer, pageIndex, w, h, cancellationToken), imageDpi);
        var page = PageRaster.Render(renderer, pageIndex, NativeDpi, maxPixels, cancellationToken);
        return new DecodedImage(page.Pixels, imageDpi * (page.Dpi / NativeDpi));
    }

    /// <summary>The fewest pixels on a side of a page at <see cref="NativeDpi"/>: 10 points.</summary>
    private const int MinPagePixels = 14;

    /// <summary>Whether a page at <see cref="NativeDpi"/> is no larger than the one an image of the given size makes.</summary>
    private static bool PageMatches((double Width, double Height) page, int width, int height)
        => page.Width <= Math.Max(width, MinPagePixels) * 0.75 + 0.5 && page.Height <= Math.Max(height, MinPagePixels) * 0.75 + 0.5;

    private static PagePixels SmallImage(PdfRenderer renderer, int pageIndex, int width, int height, CancellationToken cancellationToken)
    {
        // The page is at most 3 pixels per image pixel and a rounding pixel on a side; a larger one is drawn smaller
        // and then fails the size check below rather than taking the memory of a page the header did not announce.
        int maxDimension = 3 * Math.Max(Math.Max(width, height), MinPagePixels) + 3;
        var image = renderer.RenderPage(pageIndex, new ImageOptions { Dpi = 288, MaxDimension = maxDimension }, cancellationToken);
        if (image.Width < 3 * width || image.Height < 3 * height)
            throw new InvalidDataException("The image is not the size its header gives.");
        var rgba = image.Rgba;
        var small = new byte[checked(width * height * 4)];
        for (int y = 0; y < height; y++)
        {
            int row = (3 * y + 1) * image.Width;
            for (int x = 0; x < width; x++)
                rgba.AsSpan((row + 3 * x + 1) * 4, 4).CopyTo(small.AsSpan((y * width + x) * 4));
        }
        return PagePixels.FromRgba(small, width, height);
    }
}
