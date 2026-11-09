namespace PdfPecker.Ocr;

/// <summary>Settings for reading text with OCR. A fresh instance holds the defaults.</summary>
public sealed class OcrOptions
{
    /// <summary>
    /// The resolution PDF pages are rendered at for reading, in pixels per inch, and the resolution assumed for an image
    /// that records none. 300 suits body text; 200 reads faster and is enough for text of 10 points and larger, while
    /// small print needs more. Must be a finite number from 10 to 2,400. Default: 300.
    /// </summary>
    public float Dpi { get; set; } = 300;

    /// <summary>
    /// Look for text lines that are upside down (turned 180 degrees) and read them the right way round. Turning it off
    /// saves a little time on documents known to be upright. Default: true.
    /// </summary>
    public bool DetectRotation { get; set; } = true;

    /// <summary>
    /// Lines the recognizer is less sure of than this, from 0 to 1, are left out of the results and the text layer.
    /// Default: 0.5.
    /// </summary>
    public float MinConfidence { get; set; } = 0.5f;

    /// <summary>Which pages of a PDF are read: only those without text, or all. Default: <see cref="OcrPageMode.SkipPagesWithText"/>.</summary>
    public OcrPageMode PageMode { get; set; } = OcrPageMode.SkipPagesWithText;

    /// <summary>
    /// The least number of letters and digits that PdfPecker's text extraction must find on a page for
    /// <see cref="OcrPageMode.SkipPagesWithText"/> to treat it as having text. Default: 10, enough to skip a title page
    /// with a few words but not a scan with a stamped page number.
    /// </summary>
    public int MinPageTextCharacters { get; set; } = 10;

    /// <summary>The user or owner password of an encrypted PDF.</summary>
    public string? Password { get; set; }

    /// <summary>
    /// The pages of a PDF to read, numbered from 1 as in PdfPecker (for example <c>PdfPageRange.Parse("1-3,7")</c>).
    /// Pages are read in document order and once each, however the range orders or repeats them; pages that are not
    /// chosen are left out of the results and are not changed by
    /// <see cref="PdfOcr.MakeSearchable(byte[], OcrOptions?, CancellationToken)"/>. Default: null, every page.
    /// </summary>
    public PdfPageRange? Pages { get; set; }

    /// <summary>
    /// How many threads one page may use at most; no more than the number of processors are used, since more would only
    /// take turns. The results are the same whatever it is. Must be at least 1. Default: the number of processors.
    /// </summary>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;

    /// <summary>
    /// The most pixels a page is read at, which bounds the memory and time one page can take. A PDF page that would have
    /// more at <see cref="Dpi"/>, or an image that has more, is read at the highest resolution that stays within the
    /// limit, and its <see cref="OcrPage.Width"/>, <see cref="OcrPage.Height"/> and <see cref="OcrPage.Dpi"/> say so.
    /// Must be at least 10,000. Default: 100,000,000 (a page of 84 by 119 cm at 300 dpi). A page is never rendered with
    /// more than about 536 million pixels, 2 GB of RGBA, whatever the limit: a page that would need more within it fails
    /// with <see cref="PdfLimitException"/>.
    /// </summary>
    public long MaxPixels { get; set; } = 100_000_000;

    /// <summary>Checks the settings, so a wrong value fails before any work starts.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of its range.</exception>
    internal void Validate()
    {
        if (!(Dpi >= 10 && Dpi <= 2400))
            throw new ArgumentOutOfRangeException(nameof(Dpi), Dpi, "Dpi must be a number from 10 to 2,400.");
        if (!(MinConfidence >= 0 && MinConfidence <= 1))
            throw new ArgumentOutOfRangeException(nameof(MinConfidence), MinConfidence, "MinConfidence must be a number from 0 to 1.");
        if (PageMode is not (OcrPageMode.SkipPagesWithText or OcrPageMode.Force))
            throw new ArgumentOutOfRangeException(nameof(PageMode), PageMode, "PageMode is not a defined OcrPageMode.");
        if (MinPageTextCharacters < 1)
            throw new ArgumentOutOfRangeException(nameof(MinPageTextCharacters), MinPageTextCharacters, "MinPageTextCharacters must be at least 1.");
        if (MaxDegreeOfParallelism < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxDegreeOfParallelism), MaxDegreeOfParallelism, "MaxDegreeOfParallelism must be at least 1.");
        if (MaxPixels < 10_000)
            throw new ArgumentOutOfRangeException(nameof(MaxPixels), MaxPixels, "MaxPixels must be at least 10,000.");
    }
}
