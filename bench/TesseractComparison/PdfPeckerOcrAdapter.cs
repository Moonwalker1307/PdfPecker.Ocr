namespace TesseractComparison;

/// <summary>
/// PdfPecker.Ocr through its public API. The adapter is a stub until the app is built with -p:WithPdfPeckerOcr=true, which
/// references src/PdfPecker.Ocr and defines PDFPECKER_OCR; the real branch is written against the API in the design brief
/// (ImageOcr.RecognizeGray and RecognizeRgba returning an OcrPage of lines and words with boxes in page pixels) and has to be
/// checked against the code once it exists.
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

    public void Initialize(int threads) => _threads = threads;

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
        throw new NotSupportedException("This build has no PdfPecker.Ocr: build with -p:WithPdfPeckerOcr=true once the package has its API.");
#endif
    }

    public void Dispose() { }
}
