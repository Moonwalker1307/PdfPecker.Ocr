using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Pdf;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Mutated multi-page TIFF files through the step of <see cref="ImageOcr.ToSearchablePdf"/> that makes each page a
/// file of its own (<see cref="ImageInfo.TiffFrame"/>, within the budget <see cref="SearchablePdf.Groups"/> keeps),
/// then PdfPecker's decoders and the pixel budget on each page, without the models. Mutated directories can point
/// pages at shared, overlapping or missing strips, and every page at the whole file.
/// </summary>
internal static class TiffPagesTarget
{
    private const long MaxPixels = 250_000;

    public static Target Target { get; } = new()
    {
        Name = "tiff-pages",
        Extension = ".tif",
        MaxSize = 256 * 1024,
        Seeds = ctx => ctx.Files(Path.Combine("tests", "PdfPecker.Ocr.Tests", "Fixtures"), 256 * 1024, ".tif", ".tiff")
            .Append(("generated:pages.tif", PdfRenderer.Open(PdfMerger.Merge(ctx.GeneratedPdfs().Select(p => p.Data).ToList(), null))
                .RenderAllToTiff(new ImageOptions { Dpi = 30 }))),
        Run = (input, ct) =>
        {
            foreach (var group in SearchablePdf.Groups([input], 72))
            {
                ct.ThrowIfCancellationRequested();
                var native = PdfRenderer.Open(ImageDecoder.ToNativePdf(group.Images, ct));
                for (int i = 0; i < group.Images.Count; i++)
                    ImageDecoder.Render(native, i, ImageInfo.Read(group.Images[i]), group.Dpi, MaxPixels, ct).Pixels.Dispose();
            }
        },
    };
}
