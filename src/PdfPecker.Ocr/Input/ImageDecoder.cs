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
        return Render(renderer, 0, info.Dpi > 0 ? info.Dpi : fallbackDpi, maxPixels, cancellationToken);
    }

    /// <summary>A PDF with one page per image, each page exactly its image's size in pixels at <see cref="NativeDpi"/>.</summary>
    /// <exception cref="InvalidDataException">An image is not one PdfPecker reads, or it is damaged.</exception>
    public static byte[] ToNativePdf(IEnumerable<byte[]> images, CancellationToken cancellationToken)
        => ImageToPdf.Convert(images, new PdfOptions { Compress = false }, new ImageToPdfOptions { FitPageToImage = true, Dpi = NativeDpi },
            cancellationToken);

    /// <summary>
    /// Renders page <paramref name="pageIndex"/> of a PDF made by <see cref="ToNativePdf"/> at one pixel per image pixel,
    /// or smaller within <paramref name="maxPixels"/>, and gives it the image's resolution <paramref name="imageDpi"/>
    /// scaled the same way.
    /// </summary>
    public static DecodedImage Render(PdfRenderer renderer, int pageIndex, float imageDpi, long maxPixels, CancellationToken cancellationToken)
    {
        var page = PageRaster.Render(renderer, pageIndex, NativeDpi, maxPixels, cancellationToken);
        return new DecodedImage(page.Pixels, imageDpi * (page.Dpi / NativeDpi));
    }
}
