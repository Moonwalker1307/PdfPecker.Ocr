namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Mutated image files (PNG, JPEG, GIF, WebP, BMP and TIFF) through <see cref="ImageOcr.Recognize(byte[], OcrOptions?, CancellationToken)"/>:
/// the header reader, PdfPecker's decoders behind the image-to-PDF route, the pixel budget and the models on whatever
/// pixels come out. A mutated size field can state billions of pixels, which the pixel budget must refuse or read
/// smaller without decoding them all.
/// </summary>
internal static class OcrImageTarget
{
    /// <summary>
    /// A small pixel budget keeps a mutated image that still decodes cheap to read, so the time goes into many inputs
    /// rather than into the models on one large page.
    /// </summary>
    private static readonly OcrOptions Options = new() { MaxPixels = 250_000, MaxDegreeOfParallelism = 1 };

    public static Target Target { get; } = new()
    {
        Name = "ocr-image",
        Extension = ".img",
        MaxSize = 256 * 1024,
        MinTimeout = TimeSpan.FromSeconds(20),
        Seeds = ctx => ctx.Files(Path.Combine("tests", "PdfPecker.Ocr.Tests", "Fixtures"), 256 * 1024, ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
            .Concat(ctx.RenderedImages(PdfImageFormat.Png))
            .Concat(ctx.RenderedImages(PdfImageFormat.Jpeg))
            .Concat(ctx.RenderedImages(PdfImageFormat.WebP))
            .Concat(ctx.RenderedImages(PdfImageFormat.Tiff))
            .Concat(ctx.GeneratedGifAndBmp()),
        Run = (input, ct) => ImageOcr.Recognize(input, Options, ct),
    };
}
