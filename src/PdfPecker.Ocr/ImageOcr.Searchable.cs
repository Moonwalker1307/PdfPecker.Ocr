using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Pdf;

namespace PdfPecker.Ocr;

/// <summary>Reads the text of images and turns scans into searchable PDFs. All methods are static and thread-safe.</summary>
public static partial class ImageOcr
{
    /// <summary>
    /// Turns images into a searchable PDF: one page per image (and per page of a multi-page TIFF), each the size the
    /// image's resolution gives it, with the text OCR reads on it as an invisible layer that viewers select, copy and
    /// search.
    /// </summary>
    /// <remarks>
    /// PNG, JPEG, GIF (the first frame), WebP, BMP and TIFF are read. An image that records no resolution is taken to be
    /// at <see cref="OcrOptions.Dpi"/>, which sets its page size: an image 2,480 pixels wide at 300 dpi gives an A4 page.
    /// </remarks>
    /// <param name="images">The image files' bytes, in page order.</param>
    /// <param name="options">Resolution, confidence and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The PDF.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="images"/> or one of the images is null.</exception>
    /// <exception cref="ArgumentException">There are no images, or more than <see cref="ImageToPdf.MaxImages"/> images and TIFF pages.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of its range.</exception>
    /// <exception cref="InvalidDataException">An image is not in a supported format or is damaged.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public static byte[] ToSearchablePdf(IEnumerable<byte[]> images, OcrOptions? options = null, CancellationToken cancellationToken = default)
        => SearchablePdf.FromImages(images, options, PageRecognizer.Default, cancellationToken);
}
