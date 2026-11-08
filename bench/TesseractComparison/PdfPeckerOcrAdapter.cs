namespace TesseractComparison;

/// <summary>
/// PdfPecker.Ocr through its public API: ImageOcr.RecognizeGray for gray pages (read in place) and RecognizeRgba for colour
/// ones, with the package's default settings and MaxDegreeOfParallelism set to the thread count. The app builds it in by
/// default (PDFPECKER_OCR); with -p:WithPdfPeckerOcr=false the adapter is a stub.
/// </summary>
sealed class PdfPeckerOcrAdapter : IOcrEngineAdapter
{
    public const bool Available =
#if PDFPECKER_OCR
        true;
#else
        false;
#endif

    int _threads = Environment.ProcessorCount;

    public string Name => "pdfpecker-ocr";
    public bool InProcess => true;
    public bool Managed => true;

    public string Description =>
#if PDFPECKER_OCR
        $"PdfPecker.Ocr {PdfPecker.Ocr.OcrInfo.Version}, MaxDegreeOfParallelism {_threads}";
#else
        "PdfPecker.Ocr (not built in: stub)";
#endif

    public void Initialize(int threads)
    {
        _threads = threads;
#if PDFPECKER_OCR
        // The models load on first use and the engine's worker (threads, buffers) is built on the first page of a thread
        // count; a blank page does both here, outside the timed runs, as the other adapters load their models here.
        var blank = new byte[64 * 64];
        blank.AsSpan().Fill(255);
        PdfPecker.Ocr.ImageOcr.RecognizeGray(blank, 64, 64, new PdfPecker.Ocr.OcrOptions { MaxDegreeOfParallelism = threads });
#endif
    }

    public void Prepare(CorpusPage page) { }

    public OcrResult Recognize(ReadOnlySpan<byte> pixels, int width, int height, PixelFormat format)
    {
#if PDFPECKER_OCR
        var options = new PdfPecker.Ocr.OcrOptions { MaxDegreeOfParallelism = _threads };
        var page = format == PixelFormat.Gray8
            ? PdfPecker.Ocr.ImageOcr.RecognizeGray(pixels, width, height, options)
            : PdfPecker.Ocr.ImageOcr.RecognizeRgba(PixelConversion.ToRgba(pixels, width, height, format), width, height, options);
        var words = new List<Box>();
        foreach (var line in page.Lines)
            foreach (var w in line.Words)
                words.Add(new Box(w.Text, w.Box.X, w.Box.Y, w.Box.Width, w.Box.Height, w.Confidence));
        return new OcrResult { Text = string.Join('\n', page.Lines.Select(l => l.Text)), Words = words };
#else
        throw new NotSupportedException("This build has no PdfPecker.Ocr: it was built with -p:WithPdfPeckerOcr=false.");
#endif
    }

    public void Dispose() { }
}
