using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Layout;

namespace PdfPecker.Ocr;

public static partial class ImageOcr
{
    /// <summary>
    /// Reads the text of an image file: PNG, JPEG, GIF (the first frame), WebP, BMP or TIFF (the first page).
    /// </summary>
    /// <remarks>
    /// The image is read at its own size; its resolution, which it records or <see cref="OcrOptions.Dpi"/> stands in
    /// for, only goes into <see cref="OcrPage.Dpi"/>. An image with more pixels than <see cref="OcrOptions.MaxPixels"/>
    /// is read at the largest size within the limit, and the page's size and resolution say so; the file itself is first
    /// decoded at its full size, which only PdfPecker's limit of 100 million pixels per image bounds. Transparent parts
    /// are read as if on white.
    /// </remarks>
    /// <param name="image">The image file's bytes.</param>
    /// <param name="options">Rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The text OCR read, with <see cref="OcrPage.PageIndex"/> 0.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of its range.</exception>
    /// <exception cref="InvalidDataException">The data is not an image in a supported format, or it is damaged.</exception>
    /// <exception cref="PdfLimitException">The image needs more than 2 GB of pixels at the size it is read at.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public static OcrPage Recognize(byte[] image, OcrOptions? options = null, CancellationToken cancellationToken = default)
        => DocumentReader.ReadImage(image, options, PageRecognizer.Default, cancellationToken);

    /// <summary>Reads the text of an image file read from a stream, from its current position to its end. The stream is not disposed.</summary>
    /// <param name="image">The image file.</param>
    /// <param name="options">Rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="image"/> is not readable.</exception>
    /// <exception cref="IOException">Reading the stream fails.</exception>
    /// <inheritdoc cref="Recognize(byte[], OcrOptions?, CancellationToken)"/>
    public static OcrPage Recognize(Stream image, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (!image.CanRead) throw new ArgumentException("The stream is not readable.", nameof(image));
        return Recognize(PdfOcr.ReadAll(image, cancellationToken), options, cancellationToken);
    }

    /// <summary>Reads the text of the image file at <paramref name="path"/>.</summary>
    /// <param name="path">The image file.</param>
    /// <param name="options">Rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="IOException">The file cannot be read, for example because it does not exist.</exception>
    /// <exception cref="UnauthorizedAccessException">The process may not read the file.</exception>
    /// <inheritdoc cref="Recognize(byte[], OcrOptions?, CancellationToken)"/>
    public static OcrPage RecognizeFile(string path, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        return Recognize(File.ReadAllBytes(path), options, cancellationToken);
    }

    /// <summary>
    /// Reads the text of pixels in memory, four bytes each: red, green, blue and alpha (not premultiplied), rows top to
    /// bottom without padding, such as a camera frame or a bitmap from another library.
    /// </summary>
    /// <remarks>
    /// Transparent pixels are read as if on white. The pixels are taken to be at <see cref="OcrOptions.Dpi"/>, which only
    /// goes into <see cref="OcrPage.Dpi"/>. The buffer may be longer than the image; the rest is ignored.
    /// </remarks>
    /// <param name="rgba">The pixels, at least <paramref name="width"/> × <paramref name="height"/> × 4 bytes.</param>
    /// <param name="width">The width in pixels.</param>
    /// <param name="height">The height in pixels.</param>
    /// <param name="options">Rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The text OCR read, with <see cref="OcrPage.PageIndex"/> 0.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/> or <paramref name="height"/> is less than 1, or an option is out of its range.</exception>
    /// <exception cref="ArgumentException"><paramref name="rgba"/> is shorter than <paramref name="width"/> × <paramref name="height"/> × 4 bytes.</exception>
    /// <exception cref="PdfLimitException">The image has more pixels than <see cref="OcrOptions.MaxPixels"/>.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public static OcrPage RecognizeRgba(ReadOnlySpan<byte> rgba, int width, int height, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        options = CheckPixels(rgba.Length, width, height, 4, options, nameof(rgba));
        using var pixels = PagePixels.FromRgba(rgba, width, height);
        return OcrEngine.Instance.Recognize(pixels, 0, options.Dpi, options, cancellationToken);
    }

    /// <summary>
    /// Reads the text of gray pixels in memory, one byte each from 0 (black) to 255 (white), rows top to bottom without
    /// padding. The buffer is read where it is, without a copy.
    /// </summary>
    /// <remarks>
    /// The pixels are taken to be at <see cref="OcrOptions.Dpi"/>, which only goes into <see cref="OcrPage.Dpi"/>. The
    /// buffer may be longer than the image; the rest is ignored.
    /// </remarks>
    /// <param name="gray">The pixels, at least <paramref name="width"/> × <paramref name="height"/> bytes.</param>
    /// <param name="width">The width in pixels.</param>
    /// <param name="height">The height in pixels.</param>
    /// <param name="options">Rotation, confidence, threads and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The text OCR read, with <see cref="OcrPage.PageIndex"/> 0.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/> or <paramref name="height"/> is less than 1, or an option is out of its range.</exception>
    /// <exception cref="ArgumentException"><paramref name="gray"/> is shorter than <paramref name="width"/> × <paramref name="height"/> bytes.</exception>
    /// <exception cref="PdfLimitException">The image has more pixels than <see cref="OcrOptions.MaxPixels"/>.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public static OcrPage RecognizeGray(ReadOnlySpan<byte> gray, int width, int height, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        options = CheckPixels(gray.Length, width, height, 1, options, nameof(gray));
        var view = new PixelView(gray[..(width * height)], width, height, 1);
        return OcrEngine.Recognize(view, 0, options.Dpi, options, cancellationToken);
    }

    /// <summary>Checks a raw pixel buffer and the options before any work, and returns the options to use.</summary>
    private static OcrOptions CheckPixels(int length, int width, int height, int channels, OcrOptions? options, string name)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        options ??= new OcrOptions();
        options.Validate();
        // Compared by division, since width x height x channels can pass the range of a long.
        long pixels = (long)width * height;
        if (pixels > length / channels)
        {
            throw new ArgumentException(FormattableString.Invariant($"{width} x {height} pixels need {(decimal)pixels * channels} bytes, but there are {length}."), name);
        }

        // A buffer that is really there can still be larger than the caller allows a page to be; there is no file
        // resolution to lower here, so it is refused rather than read smaller.
        if (pixels > options.MaxPixels)
        {
            throw new PdfLimitException(FormattableString.Invariant($"An image of {width} x {height} pixels is larger than MaxPixels ({options.MaxPixels})."));
        }

        return options;
    }
}
