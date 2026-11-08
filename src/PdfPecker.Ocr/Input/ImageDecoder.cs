namespace PdfPecker.Ocr.Input;

/// <summary>An image decoded for recognition.</summary>
/// <param name="Pixels">The pixels; the receiver disposes them.</param>
/// <param name="Dpi">
/// The resolution of <paramref name="Pixels"/>: the one the image records (or the fallback), lowered in proportion when
/// the image had more pixels than allowed and was read smaller.
/// </param>
internal readonly record struct DecodedImage(PagePixels Pixels, float Dpi);

/// <summary>
/// Decodes image files (PNG, JPEG, GIF, WebP, BMP and TIFF) to pixels with PdfPecker's public API alone: the images
/// become pages of a PDF with <see cref="ImageToPdf"/>, and <see cref="PdfRenderer"/> draws each page back at one
/// pixel per image pixel.
/// </summary>
/// <remarks>
/// <para>
/// The pages are made at 96 dpi because there one image pixel is exactly one CSS pixel, so the page is exactly
/// the image's size and the renderer copies each pixel unchanged (a test checks this). At other resolutions the page
/// size goes through CSS pixels with three decimals and the image comes back shifted by a fraction of a pixel and
/// resampled. The PDF is written without compression: it lives only until the pages are rendered.
/// </para>
/// <para>
/// This is a roundabout way to decode an image; the cost (an HTML document holding the image as base64, a PDF, and a
/// full render) is noted in the package's design notes as a reason for a public image decoder in PdfPecker.
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
        var renderer = PdfRenderer.Open(ToNativePdf(new[] { image }, cancellationToken));
        return Render(renderer, 0, info, fallbackDpi, maxPixels, cancellationToken);
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
        if (w > 0 && h > 0 && (w < MinPagePixels || h < MinPagePixels) && 16L * Math.Max(w, MinPagePixels) * Math.Max(h, MinPagePixels) <= maxPixels)
            return new DecodedImage(SmallImage(renderer, pageIndex, w, h, cancellationToken), imageDpi);
        var page = PageRaster.Render(renderer, pageIndex, NativeDpi, maxPixels, cancellationToken);
        return new DecodedImage(page.Pixels, imageDpi * (page.Dpi / NativeDpi));
    }

    /// <summary>The fewest pixels on a side of a page at <see cref="NativeDpi"/>: 10 points.</summary>
    private const int MinPagePixels = 14;

    private static PagePixels SmallImage(PdfRenderer renderer, int pageIndex, int width, int height, CancellationToken cancellationToken)
    {
        var image = renderer.RenderPage(pageIndex, new ImageOptions { Dpi = 288, MaxDimension = int.MaxValue }, cancellationToken);
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
