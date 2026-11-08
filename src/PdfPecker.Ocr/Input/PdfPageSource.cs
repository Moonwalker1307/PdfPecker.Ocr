using System.Text;

namespace PdfPecker.Ocr.Input;

/// <summary>
/// The pages of a PDF that OCR reads: which pages were chosen, which of them already have text and are skipped, and
/// each page rendered to pixels at the resolution the options ask for, within their pixel budget.
/// </summary>
internal sealed class PdfPageSource
{
    private readonly PdfRenderer _renderer;
    private readonly OcrOptions _options;
    private readonly Dictionary<int, PdfTextPage> _withText;

    private PdfPageSource(PdfRenderer renderer, OcrOptions options, int[] pages, Dictionary<int, PdfTextPage> withText)
    {
        _renderer = renderer;
        _options = options;
        Pages = pages;
        _withText = withText;
    }

    /// <summary>The number of pages in the PDF.</summary>
    public int PageCount => _renderer.PageCount;

    /// <summary>The chosen pages, 0-based, in document order and once each.</summary>
    public IReadOnlyList<int> Pages { get; }

    /// <summary>Opens a PDF and finds the pages to read.</summary>
    /// <exception cref="PdfPasswordException">The PDF is encrypted and the password does not open it.</exception>
    /// <exception cref="InvalidDataException">The data is not a readable PDF.</exception>
    /// <exception cref="NotSupportedException">The PDF uses an encryption handler other than the standard one.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The page range names a page the PDF does not have.</exception>
    public static PdfPageSource Open(byte[] pdf, OcrOptions options, CancellationToken cancellationToken)
    {
        var renderer = PdfRenderer.Open(pdf, options.Password);
        int count = renderer.PageCount;
        int[] pages = options.Pages is { } range
            ? range.ToPageNumbers(count).Select(n => n - 1).Distinct().Order().ToArray()
            : Enumerable.Range(0, count).ToArray();
        var withText = new Dictionary<int, PdfTextPage>();
        if (options.PageMode == OcrPageMode.SkipPagesWithText && pages.Length > 0)
        {
            foreach (var page in ExtractText(pdf, options, pages, cancellationToken))
                if (HasText(page.Text, options.MinPageTextCharacters)) withText[page.PageNumber - 1] = page;
        }
        return new PdfPageSource(renderer, options, pages, withText);
    }

    /// <summary>Whether a chosen page is left alone because it already has text.</summary>
    public bool HasText(int pageIndex) => _withText.ContainsKey(pageIndex);

    /// <summary>The page's displayed size in points: its crop box after its rotation.</summary>
    public (double Width, double Height) PageSize(int pageIndex) => _renderer.GetPageSize(pageIndex);

    /// <summary>Renders a page for recognition.</summary>
    /// <exception cref="InvalidDataException">The page cannot be rendered.</exception>
    public RenderedPage Render(int pageIndex, CancellationToken cancellationToken)
        => PageRaster.Render(_renderer, pageIndex, _options.Dpi, _options.MaxPixels, cancellationToken);

    /// <summary>
    /// The text a page already has, as an OCR result in the pixels the page would have been read at, with confidence 1
    /// and the upright boxes PdfPecker's text extraction reports.
    /// </summary>
    public OcrPage ExistingText(int pageIndex)
    {
        var text = _withText[pageIndex];
        var (w, h) = PageSize(pageIndex);
        float dpi = PageRaster.FitDpi(w, h, _options.Dpi, _options.MaxPixels);
        int width = (int)Math.Max(1, Math.Round(w * dpi / 72)), height = (int)Math.Max(1, Math.Round(h * dpi / 72));
        double sx = width / Math.Max(w, 1e-9), sy = height / Math.Max(h, 1e-9);
        var lines = new List<OcrLine>(text.Lines.Count);
        foreach (var line in text.Lines)
        {
            var words = line.Words.Select(word => new OcrWord(word.Text, Box(word.X, word.Y, word.Width, word.Height, sx, sy), 1f));
            lines.Add(new OcrLine(line.Text, Box(line.X, line.Y, line.Width, line.Height, sx, sy), 1f, words));
        }
        return new OcrPage(pageIndex, width, height, dpi, lines);
    }

    /// <summary>Whether text has at least <paramref name="min"/> letters or digits.</summary>
    internal static bool HasText(string text, int min)
    {
        int n = 0;
        foreach (var rune in text.EnumerateRunes())
            if (Rune.IsLetterOrDigit(rune) && ++n >= min) return true;
        return false;
    }

    private static IReadOnlyList<PdfTextPage> ExtractText(byte[] pdf, OcrOptions options, int[] pages, CancellationToken cancellationToken)
    {
        var textOptions = new PdfTextOptions
        {
            Password = options.Password,
            Pages = PdfPageRange.FromPages(pages.Select(p => p + 1)),
            MaxPages = int.MaxValue,
            MaxDegreeOfParallelism = options.MaxDegreeOfParallelism,
        };
        try
        {
            return PdfText.ExtractPages(pdf, textOptions, cancellationToken);
        }
        catch (InvalidDataException)
        {
            // Text that cannot be extracted is text a reader cannot search either, so the pages are read with OCR;
            // the renderer reports the PDF as damaged if it cannot draw them either.
            return Array.Empty<PdfTextPage>();
        }
    }

    private static OcrQuad Box(double x, double y, double w, double h, double sx, double sy)
        => OcrGeometry.FromBox((float)(x * sx), (float)(y * sy), (float)(w * sx), (float)(h * sy));
}
