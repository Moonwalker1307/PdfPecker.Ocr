using PdfPecker.Ocr.Input;

namespace PdfPecker.Ocr;

public static partial class PdfOcr
{
    /// <summary>
    /// Reads the text of a scanned PDF: each chosen page is rendered at <see cref="OcrOptions.Dpi"/> and read with OCR.
    /// </summary>
    /// <remarks>
    /// With <see cref="OcrPageMode.SkipPagesWithText"/> (the default), a page that already has text is not read with
    /// OCR: its result is the text PdfPecker extracts from it, with a confidence of 1, so the document's text is complete
    /// either way. Pages are read one after another, each with up to <see cref="OcrOptions.MaxDegreeOfParallelism"/>
    /// threads. A page that would have more pixels than <see cref="OcrOptions.MaxPixels"/> is rendered at a lower
    /// resolution, and its <see cref="OcrPage.Dpi"/> says which.
    /// </remarks>
    /// <param name="pdf">The PDF.</param>
    /// <param name="options">Resolution, pages, password, rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The chosen pages in document order.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="pdf"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of its range, or <see cref="OcrOptions.Pages"/> names a page the PDF does not have.</exception>
    /// <exception cref="PdfPasswordException">The PDF is encrypted and <see cref="OcrOptions.Password"/> does not open it.</exception>
    /// <exception cref="InvalidDataException">The PDF is not readable, or a page cannot be rendered.</exception>
    /// <exception cref="NotSupportedException">The PDF uses an encryption handler other than the standard one.</exception>
    /// <exception cref="PdfLimitException">A page needs more than 2 GB of pixels at the resolution it is read at.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public static OcrDocument Recognize(byte[] pdf, OcrOptions? options = null, CancellationToken cancellationToken = default)
        => DocumentReader.ReadPdf(pdf, options, PageRecognizer.Default, cancellationToken);

    /// <summary>Reads the text of a scanned PDF read from a stream, from its current position to its end. The stream is not disposed.</summary>
    /// <param name="pdf">The PDF.</param>
    /// <param name="options">Resolution, pages, password, rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <exception cref="ArgumentNullException"><paramref name="pdf"/> is null.</exception>
    /// <exception cref="IOException">Reading the stream fails.</exception>
    /// <inheritdoc cref="Recognize(byte[], OcrOptions?, CancellationToken)"/>
    public static OcrDocument Recognize(Stream pdf, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pdf);
        return Recognize(ReadAll(pdf), options, cancellationToken);
    }

    /// <summary>Reads the text of the scanned PDF file at <paramref name="path"/>.</summary>
    /// <param name="path">The PDF file.</param>
    /// <param name="options">Resolution, pages, password, rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="IOException">The file cannot be read, for example because it does not exist.</exception>
    /// <exception cref="UnauthorizedAccessException">The process may not read the file.</exception>
    /// <inheritdoc cref="Recognize(byte[], OcrOptions?, CancellationToken)"/>
    public static OcrDocument RecognizeFile(string path, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        return Recognize(File.ReadAllBytes(path), options, cancellationToken);
    }
}
