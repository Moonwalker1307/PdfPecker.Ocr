using System.Buffers;

namespace PdfPecker.Ocr.Input;

/// <summary>How a pixel of <see cref="PagePixels"/> is stored; the value is the number of bytes per pixel.</summary>
internal enum PixelFormat
{
    /// <summary>One byte of brightness, 0 black to 255 white.</summary>
    Gray8 = 1,

    /// <summary>Three bytes: red, green and blue, in that order.</summary>
    Rgb24 = 3,
}

/// <summary>
/// The pixels of a page or image, ready for recognition: 8 bits per channel, gray or RGB, rows top to bottom without
/// padding, on a white background (transparency is resolved when the pixels are made).
/// </summary>
/// <remarks>
/// The buffer comes from <see cref="ArrayPool{T}.Shared"/> and goes back there on <see cref="Dispose"/>, so a page of
/// 300 dpi does not leave tens of megabytes for the garbage collector each time. Use the pixels only until the instance
/// is disposed. A gray page is stored as gray, a third of the memory of RGB: recognition feeds the same value to all
/// three channels.
/// </remarks>
internal sealed class PagePixels : IDisposable
{
    private byte[]? _buffer;

    private PagePixels(int width, int height, PixelFormat format, byte[] buffer)
    {
        Width = width;
        Height = height;
        Format = format;
        _buffer = buffer;
    }

    /// <summary>The width in pixels.</summary>
    public int Width { get; }

    /// <summary>The height in pixels.</summary>
    public int Height { get; }

    /// <summary>How each pixel is stored.</summary>
    public PixelFormat Format { get; }

    /// <summary>Bytes per pixel: 1 for gray, 3 for RGB.</summary>
    public int Channels => (int)Format;

    /// <summary>Bytes per row.</summary>
    public int Stride => Width * Channels;

    /// <summary>The number of bytes of <see cref="Pixels"/>.</summary>
    public int Length => Width * Height * Channels;

    /// <summary>The pixels, row by row from the top.</summary>
    /// <exception cref="ObjectDisposedException">The instance was disposed.</exception>
    public ReadOnlySpan<byte> Pixels => Buffer().AsSpan(0, Length);

    /// <summary>The pixels, for code in this package that fills them.</summary>
    internal Span<byte> WritablePixels => Buffer().AsSpan(0, Length);

    /// <summary>One row of pixels.</summary>
    public ReadOnlySpan<byte> Row(int y) => Buffer().AsSpan(y * Stride, Stride);

    /// <summary>A page of uninitialised pixels from the pool.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A side is less than 1.</exception>
    /// <exception cref="PdfLimitException">The page needs more than 2 GB.</exception>
    public static PagePixels Rent(int width, int height, PixelFormat format)
    {
        if (width < 1) throw new ArgumentOutOfRangeException(nameof(width), width, "The width must be at least 1 pixel.");
        if (height < 1) throw new ArgumentOutOfRangeException(nameof(height), height, "The height must be at least 1 pixel.");
        long length = (long)width * height * (int)format;
        if (length > Array.MaxLength)
            throw new PdfLimitException(FormattableString.Invariant($"A page of {width} x {height} pixels is larger than 2 GB."));
        return new PagePixels(width, height, format, ArrayPool<byte>.Shared.Rent((int)length));
    }

    /// <summary>Copies RGBA pixels (straight alpha), putting transparent pixels on white, and stores a gray image as gray.</summary>
    /// <exception cref="ArgumentException"><paramref name="rgba"/> is shorter than <paramref name="width"/> × <paramref name="height"/> × 4.</exception>
    public static PagePixels FromRgba(ReadOnlySpan<byte> rgba, int width, int height)
    {
        CheckLength(rgba.Length, width, height, 4, nameof(rgba));
        var src = rgba[..(width * height * 4)];
        var pixels = Rent(width, height, PixelConvert.IsGray(src) ? PixelFormat.Gray8 : PixelFormat.Rgb24);
        if (pixels.Format == PixelFormat.Gray8) PixelConvert.RgbaToGray(src, pixels.WritablePixels);
        else PixelConvert.RgbaToRgb(src, pixels.WritablePixels);
        return pixels;
    }

    /// <summary>Copies RGB pixels.</summary>
    /// <exception cref="ArgumentException"><paramref name="rgb"/> is shorter than <paramref name="width"/> × <paramref name="height"/> × 3.</exception>
    public static PagePixels FromRgb(ReadOnlySpan<byte> rgb, int width, int height)
    {
        CheckLength(rgb.Length, width, height, 3, nameof(rgb));
        var pixels = Rent(width, height, PixelFormat.Rgb24);
        rgb[..pixels.Length].CopyTo(pixels.WritablePixels);
        return pixels;
    }

    /// <summary>Copies RGB pixels, storing them as gray when every pixel has equal red, green and blue.</summary>
    /// <exception cref="ArgumentException"><paramref name="rgb"/> is shorter than <paramref name="width"/> × <paramref name="height"/> × 3.</exception>
    public static PagePixels FromRgbOrGray(ReadOnlySpan<byte> rgb, int width, int height)
    {
        CheckLength(rgb.Length, width, height, 3, nameof(rgb));
        var src = rgb[..(width * height * 3)];
        if (!PixelConvert.IsGrayRgb(src)) return FromRgb(src, width, height);
        var pixels = Rent(width, height, PixelFormat.Gray8);
        PixelConvert.RgbToGray(src, pixels.WritablePixels);
        return pixels;
    }

    /// <summary>Copies gray pixels.</summary>
    /// <exception cref="ArgumentException"><paramref name="gray"/> is shorter than <paramref name="width"/> × <paramref name="height"/>.</exception>
    public static PagePixels FromGray(ReadOnlySpan<byte> gray, int width, int height)
    {
        CheckLength(gray.Length, width, height, 1, nameof(gray));
        var pixels = Rent(width, height, PixelFormat.Gray8);
        gray[..pixels.Length].CopyTo(pixels.WritablePixels);
        return pixels;
    }

    /// <summary>Returns the buffer to the pool. Calling it again does nothing.</summary>
    public void Dispose()
    {
        var buffer = Interlocked.Exchange(ref _buffer, null);
        if (buffer != null) ArrayPool<byte>.Shared.Return(buffer);
    }

    private byte[] Buffer() => _buffer ?? throw new ObjectDisposedException(nameof(PagePixels));

    private static void CheckLength(int length, int width, int height, int channels, string name)
    {
        if (width < 1) throw new ArgumentOutOfRangeException(nameof(width), width, "The width must be at least 1 pixel.");
        if (height < 1) throw new ArgumentOutOfRangeException(nameof(height), height, "The height must be at least 1 pixel.");
        long needed = (long)width * height * channels;
        if (needed > Array.MaxLength)
            throw new PdfLimitException(FormattableString.Invariant($"An image of {width} x {height} pixels is larger than 2 GB."));
        if (length < needed)
            throw new ArgumentException(FormattableString.Invariant($"{width} x {height} pixels need {needed} bytes, but there are {length}."), name);
    }
}
