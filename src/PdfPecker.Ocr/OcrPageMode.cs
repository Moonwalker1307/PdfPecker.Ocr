namespace PdfPecker.Ocr;

/// <summary>Which pages of a PDF are read with OCR.</summary>
public enum OcrPageMode
{
    /// <summary>
    /// Pages that already have text are not read again: <see cref="PdfOcr.MakeSearchable(byte[], OcrOptions?, CancellationToken)"/>
    /// leaves them as they are, and recognition returns the text they have instead. A page has text when PdfPecker's text
    /// extraction finds at least <see cref="OcrOptions.MinPageTextCharacters"/> letters or digits on it, which includes
    /// the invisible text of a page that was made searchable before. Born-digital pages and pages that went through OCR
    /// already are skipped, while scans with only a few stamped characters, such as a page number, are still read.
    /// </summary>
    SkipPagesWithText,

    /// <summary>
    /// Every chosen page is read, whatever text it has. A page that was made searchable before gets a second text layer,
    /// so its words are found twice.
    /// </summary>
    Force,
}
