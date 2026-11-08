namespace PdfPecker.Ocr.Detection;

/// <summary>
/// A page's pixels as the pipeline reads them: 8-bit gray (one channel), RGB (three) or RGBA (four, alpha ignored, so
/// callers composite transparent images onto white first), rows top to bottom without padding.
/// </summary>
internal readonly ref struct PixelView
{
    public PixelView(ReadOnlySpan<byte> data, int width, int height, int channels)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "The image must be at least one pixel wide and high.");
        }

        if (channels is not (1 or 3 or 4))
        {
            throw new ArgumentOutOfRangeException(nameof(channels), "The image must have 1, 3 or 4 channels.");
        }

        if (data.Length < (long)width * height * channels)
        {
            throw new ArgumentException("The pixel buffer is smaller than width * height * channels.", nameof(data));
        }

        Data = data;
        Width = width;
        Height = height;
        Channels = channels;
    }

    public ReadOnlySpan<byte> Data { get; }

    public int Width { get; }

    public int Height { get; }

    public int Channels { get; }
}
