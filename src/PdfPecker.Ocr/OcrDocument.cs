using System.Collections.ObjectModel;
using PdfPecker.Ocr.Output;

namespace PdfPecker.Ocr;

/// <summary>The text OCR read in a document, page by page.</summary>
/// <remarks>Instances are immutable and can be shared between threads.</remarks>
public sealed class OcrDocument
{
    private string? _text;

    internal OcrDocument(IEnumerable<OcrPage> pages)
    {
        ArgumentNullException.ThrowIfNull(pages);
        Pages = new ReadOnlyCollection<OcrPage>(pages.ToArray());
    }

    /// <summary>The pages that were read, in document order. Each knows its place in the document from <see cref="OcrPage.PageIndex"/>.</summary>
    public IReadOnlyList<OcrPage> Pages { get; }

    /// <summary>
    /// The document's text: the pages' <see cref="OcrPage.Text"/> separated by a form feed (U+000C), as
    /// <see cref="PdfText.Extract(byte[], PdfTextOptions?, CancellationToken)"/> and pdftotext separate pages.
    /// </summary>
    public string Text => _text ??= PlainText.Join(Pages);

    /// <summary>
    /// The document as one hOCR 1.2 document with an <c>ocr_page</c> element for each page (see
    /// <see cref="OcrPage.ToHocr"/>).
    /// </summary>
    public string ToHocr() => HocrWriter.Write(Pages);

    /// <summary>The number of pages, for debugging and logs.</summary>
    public override string ToString() => FormattableString.Invariant($"{Pages.Count} pages");
}
