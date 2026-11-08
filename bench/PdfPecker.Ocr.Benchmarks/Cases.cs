using System.Globalization;
using System.Net;
using System.Text;
using PdfPecker;
using PdfPecker.Ocr;
using TesseractComparison;

namespace OcrBench;

/// <summary>
/// One thing to time. <see cref="Run"/> does one unit of work; <see cref="Pixels"/> is the page area it handles (for the
/// megapixels-per-second column), 0 when that makes no sense. A case whose code is not there yet is
/// <see cref="Pending"/>: it is listed, with the reason, and not run.
/// </summary>
sealed record BenchCase(string Name, Action? Run, long Pixels, string? Pending = null, int MinRuns = 5, double BudgetSeconds = 10);

static class Cases
{
    public static IEnumerable<BenchCase> Create(string? corpusDir, int threads)
    {
        // Always available: the package's own information, which proves the harness runs against the build under test.
        yield return new BenchCase("info (OcrInfo.Version)", () => _ = OcrInfo.Version, 0, MinRuns: 1000, BudgetSeconds: 1);

        // Test pages. The built-in page is drawn by PdfPecker so that the harness needs no files; a corpus folder adds its pages.
        var pages = new List<(string Name, byte[] Pixels, int Width, int Height, int Channels)>();
        var (bp, bw, bh, _) = BuiltInPage(300);
        pages.Add(("page-300dpi-gray", bp, bw, bh, 1));
        var (bp2, bw2, bh2, png2) = BuiltInPage(150);
        pages.Add(("page-150dpi-gray", bp2, bw2, bh2, 1));
        if (corpusDir is not null && Directory.Exists(corpusDir))
            foreach (var cp in CorpusPage.Load(corpusDir, ["letter-en.300", "invoice-fr.150", "article-en.300", "smallprint-en.300", "letter-de.300.scan"]))
            {
                var (px, w, h, c) = PngDecoder.Decode(File.ReadAllBytes(cp.ImagePath));
                pages.Add(("corpus-" + cp.Id, px, w, h, c));
            }

#if OCR_API
        foreach (var (name, pixels, w, h, c) in pages)
        {
            var options = new OcrOptions { MaxDegreeOfParallelism = threads };
            if (c == 1)
            {
                yield return new BenchCase($"{name} ({threads}t)", () => ImageOcr.RecognizeGray(pixels, w, h, options), (long)w * h, MinRuns: 3, BudgetSeconds: 20);
                yield return new BenchCase($"{name} (1t)", () => ImageOcr.RecognizeGray(pixels, w, h, new OcrOptions { MaxDegreeOfParallelism = 1 }), (long)w * h, MinRuns: 3, BudgetSeconds: 30);
            }
            else
            {
                var rgba = ToRgba(pixels, w, h, c);
                yield return new BenchCase($"{name} ({threads}t)", () => ImageOcr.RecognizeRgba(rgba, w, h, options), (long)w * h, MinRuns: 3, BudgetSeconds: 20);
            }
        }

        // A scanned PDF: the 150 dpi built-in page as an image-only PDF, made searchable.
        var scan = ImageToPdf.Convert([png2]);
        yield return new BenchCase("searchable-pdf (1 page, 150 dpi)", () => PdfOcr.MakeSearchable(scan, new OcrOptions { MaxDegreeOfParallelism = threads }), (long)bw2 * bh2, MinRuns: 3, BudgetSeconds: 20);
#else
        foreach (var (name, _, w, h, _) in pages)
            yield return new BenchCase($"{name} ({threads}t)", null, (long)w * h, Pending: "needs -p:OcrApi=true (ImageOcr.RecognizeGray / RecognizeRgba)");
        yield return new BenchCase("searchable-pdf (1 page, 150 dpi)", null, 0, Pending: "needs -p:OcrApi=true (PdfOcr.MakeSearchable)");
#endif

#if OCR_INTERNALS
        // The internal contracts of the design brief. The signatures below are the ones it names; adjust them to the real code.
        // TextDetector.Detect(page pixels) -> quads; TextRecognizer.Recognize(page pixels, quads) -> lines.
        var (dp, dw, dh) = (bp, bw, bh);
        yield return new BenchCase("detection (300 dpi page)", () => throw new NotImplementedException("wire up TextDetector.Detect"), (long)dw * dh, MinRuns: 3);
        yield return new BenchCase("recognition (lines of the 300 dpi page)", () => throw new NotImplementedException("wire up TextRecognizer.Recognize"), (long)dw * dh, MinRuns: 3);
#else
        yield return new BenchCase("detection (300 dpi page)", null, (long)bw * bh, Pending: "needs -p:OcrInternals=true (TextDetector.Detect)");
        yield return new BenchCase("recognition (lines of the 300 dpi page)", null, (long)bw * bh, Pending: "needs -p:OcrInternals=true (TextRecognizer.Recognize)");
        yield return new BenchCase("kernels: 1x1 conv as GEMM, depthwise conv, attention", null, 0, Pending: "needs -p:OcrInternals=true (Inference kernels)");
#endif
    }

    static byte[] ToRgba(byte[] pixels, int w, int h, int channels)
    {
        var rgba = new byte[w * h * 4];
        for (int i = 0; i < w * h; i++)
        {
            rgba[i * 4] = pixels[i * channels];
            rgba[i * 4 + 1] = pixels[i * channels + (channels == 3 ? 1 : 0)];
            rgba[i * 4 + 2] = pixels[i * channels + (channels == 3 ? 2 : 0)];
            rgba[i * 4 + 3] = 255;
        }
        return rgba;
    }

    /// <summary>The page text: 32 explicit lines of a business letter, black on white, drawn by PdfPecker at the given resolution.</summary>
    static (byte[] Pixels, int Width, int Height, byte[] Png) BuiltInPage(int dpi)
    {
        var sb = new StringBuilder();
        string[] sentences =
        [
            "Thank you for your letter of 14 October about the new agreement for the maintenance of the",
            "ventilation system in your office building. We have reviewed the offer and can confirm that",
            "the price is unchanged from last year, even though the cost of spare parts has risen.",
            "Our engineers will visit on Thursday 6 November between 8 and 12 o'clock. Please make sure",
            "that they can reach the basement and the attic, so that the filters can be replaced and the",
            "motors checked. If the time does not suit you, please call us on 020 7946 0123 or write to",
            "service@example.com. Invoices are payable within 30 days; the amount due is $4,218.50.",
        ];
        sb.Append("<!doctype html><html><head><meta charset=\"utf-8\"><style>@page{size:A4;margin:20mm}body{font:11pt/1.4 'Liberation Serif',serif}div{white-space:nowrap}</style></head><body>");
        for (int i = 0; i < 32; i++) sb.Append("<div>").Append(WebUtility.HtmlEncode(sentences[i % sentences.Length])).Append("</div>");
        sb.Append("</body></html>");
        var pdf = HtmlToPdf.Convert(sb.ToString());
        var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = dpi });
        var rgba = image.Rgba;
        var gray = new byte[image.Width * image.Height];
        for (int i = 0; i < gray.Length; i++) gray[i] = (byte)((rgba[i * 4] * 77 + rgba[i * 4 + 1] * 150 + rgba[i * 4 + 2] * 29 + 128) >> 8);
        return (gray, image.Width, image.Height, image.ToPng());
    }

}
