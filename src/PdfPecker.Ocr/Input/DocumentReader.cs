namespace PdfPecker.Ocr.Input;

/// <summary>
/// Reads whole documents with a recognizer: the page loops behind <c>PdfOcr.Recognize</c> and
/// <c>ImageOcr.Recognize</c>, kept apart from the engine so they can be tested with a recognizer that returns known
/// words.
/// </summary>
internal static class DocumentReader
{
    /// <summary>
    /// Reads the chosen pages of a PDF. With <see cref="OcrPageMode.SkipPagesWithText"/>, a page that has text is not
    /// read with OCR; its result is the text it has (see <see cref="PdfPageSource.ExistingText"/>), so the document's
    /// text is complete either way.
    /// </summary>
    public static OcrDocument ReadPdf(byte[] pdf, OcrOptions? options, IPageRecognizer recognizer, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(pdf);
        options ??= new OcrOptions();
        options.Validate();
        var source = PdfPageSource.Open(pdf, options, cancellationToken);
        var pages = new List<OcrPage>(source.Pages.Count);
        foreach (int index in source.Pages)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (source.HasText(index))
            {
                pages.Add(source.ExistingText(index));
                continue;
            }
            var rendered = source.Render(index, cancellationToken);
            using (rendered.Pixels)
                pages.Add(recognizer.Recognize(rendered.Pixels, index, rendered.Dpi, options, cancellationToken));
        }
        return new OcrDocument(pages);
    }

    /// <summary>Reads an image file (the first page of a multi-page TIFF).</summary>
    public static OcrPage ReadImage(byte[] image, OcrOptions? options, IPageRecognizer recognizer, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(image);
        options ??= new OcrOptions();
        options.Validate();
        var decoded = ImageDecoder.Decode(image, options.Dpi, options.MaxPixels, cancellationToken);
        using (decoded.Pixels)
            return recognizer.Recognize(decoded.Pixels, 0, decoded.Dpi, options, cancellationToken);
    }
}
