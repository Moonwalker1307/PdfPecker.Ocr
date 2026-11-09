namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Mutated PDFs through <see cref="PdfOcr.Recognize(byte[], OcrOptions?, CancellationToken)"/> and
/// <see cref="PdfOcr.MakeSearchable(byte[], OcrOptions?, CancellationToken)"/>: opening, the text check that decides
/// which pages to skip, rendering within the pixel budget, the models, the text layer and stamping it onto the pages.
/// </summary>
/// <remarks>
/// The input's length chooses the mode, so both are covered without paying for two runs per input: odd lengths read
/// every page with <see cref="OcrPageMode.Force"/> and make the result searchable, even lengths only read the pages
/// without text. A low resolution and pixel budget keep each page cheap; the page count of a mutated file is not
/// limited, since many pages are a case to cover too.
/// </remarks>
internal static class OcrPdfTarget
{
    public static Target Target { get; } = new()
    {
        Name = "ocr-pdf",
        Extension = ".pdf",
        MaxSize = 512 * 1024,
        MinTimeout = TimeSpan.FromSeconds(30),
        // MakeSearchable refuses a signed PDF rather than break its signature.
        ExtraExpected = ex => ex is InvalidOperationException && ex.Message.Contains("sign", StringComparison.OrdinalIgnoreCase),
        Seeds = ctx => ctx.Files(Path.Combine("tests", "PdfPecker.Ocr.Tests", "Fixtures"), 512 * 1024, ".pdf")
            .Concat(ctx.GeneratedPdfs())
            .Concat(ctx.ScannedPdfs()),
        Run = (input, ct) =>
        {
            var options = new OcrOptions { Dpi = 50, MaxPixels = 400_000, MaxDegreeOfParallelism = 1 };
            if (input.Length % 2 == 1)
            {
                options.PageMode = OcrPageMode.Force;
                byte[] searchable = PdfOcr.MakeSearchable(input, options, ct);
                if (searchable.Length == 0) throw new InvalidOperationException("MakeSearchable returned no bytes.");
            }
            else
            {
                PdfOcr.Recognize(input, options, ct);
            }
        },
    };
}
