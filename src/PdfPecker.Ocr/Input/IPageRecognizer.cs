namespace PdfPecker.Ocr.Input;

/// <summary>
/// Reads the text of one page of pixels. The OCR engine implements it; the input, PDF and output code depends only on
/// this, so tests can put a recognizer that returns known words in its place.
/// </summary>
internal interface IPageRecognizer
{
    /// <summary>Reads a page.</summary>
    /// <param name="pixels">The page's pixels; the caller keeps them and disposes them afterwards.</param>
    /// <param name="pageIndex">The page's position in its document, for <see cref="OcrPage.PageIndex"/>.</param>
    /// <param name="dpi">The pixels' resolution, for <see cref="OcrPage.Dpi"/>.</param>
    /// <param name="options">The settings, already checked.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The page's lines in reading order, with <see cref="OcrPage.Width"/> and <see cref="OcrPage.Height"/> those of <paramref name="pixels"/>.</returns>
    OcrPage Recognize(PagePixels pixels, int pageIndex, float dpi, OcrOptions options, CancellationToken cancellationToken);
}
