using System.Collections.ObjectModel;
using PdfPecker.Ocr.Output;

namespace PdfPecker.Ocr;

/// <summary>The text OCR read on one page or image: its lines in reading order, and the size it was read at.</summary>
/// <remarks>
/// Every position on the page is in pixels from its top-left corner, in the <see cref="Width"/> by <see cref="Height"/>
/// pixels the page was read at. Multiply by 72 / <see cref="Dpi"/> for points. Instances are immutable and can be shared
/// between threads.
/// </remarks>
public sealed class OcrPage
{
    private string? _text;

    internal OcrPage(int pageIndex, int width, int height, float dpi, IEnumerable<OcrLine> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);
        PageIndex = pageIndex;
        Width = width;
        Height = height;
        Dpi = dpi;
        Lines = new ReadOnlyCollection<OcrLine>(lines.ToArray());
        Confidence = WeightedConfidence(Lines);
    }

    /// <summary>
    /// The page's position in its document, from 0: the page index of a PDF, the frame of a multi-page image, or the
    /// position of an image in a list. A single image is page 0.
    /// </summary>
    public int PageIndex { get; }

    /// <summary>The width the page was read at, in pixels.</summary>
    public int Width { get; }

    /// <summary>The height the page was read at, in pixels.</summary>
    public int Height { get; }

    /// <summary>
    /// The resolution the page was read at, in pixels per inch: the rendering resolution of a PDF page, or the
    /// resolution an image records (<see cref="OcrOptions.Dpi"/> when it records none).
    /// </summary>
    public float Dpi { get; }

    /// <summary>The lines in reading order: top to bottom, and column by column where the page has columns.</summary>
    public IReadOnlyList<OcrLine> Lines { get; }

    /// <summary>The page's text: the lines in reading order, each ended by a line feed (U+000A) except the last.</summary>
    public string Text => _text ??= PlainText.Join(Lines);

    /// <summary>
    /// How sure the recognizer is of the page, from 0 to 1: the average confidence of its lines, each counted by its
    /// number of characters, so a long line weighs more than a short one. A page without lines has a confidence of 0.
    /// </summary>
    public float Confidence { get; }

    /// <summary>
    /// The page as an hOCR 1.2 document (XHTML with the classes <c>ocr_page</c>, <c>ocr_line</c> and
    /// <c>ocrx_word</c>, with their boxes in pixels and the words' confidences), which tools such as hocr-tools and
    /// OCRmyPDF read.
    /// </summary>
    public string ToHocr() => HocrWriter.Write(new[] { this });

    /// <summary>The page index and size, for debugging and logs.</summary>
    public override string ToString() => FormattableString.Invariant($"Page {PageIndex}: {Width} x {Height} px, {Lines.Count} lines");

    private static float WeightedConfidence(IReadOnlyList<OcrLine> lines)
    {
        double sum = 0, weight = 0;
        foreach (var line in lines)
        {
            if (!float.IsFinite(line.Confidence)) continue;
            // A line without letters still counts once, so a page of empty lines has their confidence and not 0.
            int n = Math.Max(1, line.Text.Length);
            sum += (double)Math.Clamp(line.Confidence, 0f, 1f) * n;
            weight += n;
        }
        return weight > 0 ? (float)(sum / weight) : 0f;
    }
}
