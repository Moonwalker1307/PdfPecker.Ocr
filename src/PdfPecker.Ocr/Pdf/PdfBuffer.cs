using System.Globalization;
using System.IO.Compression;

namespace PdfPecker.Ocr.Pdf;

/// <summary>
/// A growable byte buffer for PDF syntax: ASCII text, numbers in the invariant culture and hex strings, written
/// straight to bytes so a page of thousands of words makes no intermediate strings.
/// </summary>
internal sealed class PdfBuffer
{
    /// <summary>The largest coordinate written; PDF readers are only required to handle numbers about this size.</summary>
    public const double MaxNumber = 32_767;

    private byte[] _data;

    public PdfBuffer(int capacity = 256) => _data = new byte[Math.Max(16, capacity)];

    /// <summary>The number of bytes written.</summary>
    public int Length { get; private set; }

    /// <summary>The bytes written so far.</summary>
    public ReadOnlySpan<byte> Written => _data.AsSpan(0, Length);

    public PdfBuffer Ascii(string text)
    {
        var span = Reserve(text.Length);
        for (int i = 0; i < text.Length; i++) span[i] = (byte)text[i];
        Length += text.Length;
        return this;
    }

    public PdfBuffer Bytes(ReadOnlySpan<byte> bytes)
    {
        bytes.CopyTo(Reserve(bytes.Length));
        Length += bytes.Length;
        return this;
    }

    public PdfBuffer Byte(byte b)
    {
        Reserve(1)[0] = b;
        Length++;
        return this;
    }

    public PdfBuffer Space() => Byte((byte)' ');

    public PdfBuffer Int(long value)
    {
        var span = Reserve(20);
        value.TryFormat(span, out int n, default, CultureInfo.InvariantCulture);
        Length += n;
        return this;
    }

    /// <summary>
    /// A real number with at most four decimals and no exponent, clamped to ±<see cref="MaxNumber"/>; a value that is
    /// not a number is written as 0, so no input can produce a token a PDF reader rejects.
    /// </summary>
    public PdfBuffer Real(double value)
    {
        if (double.IsNaN(value)) value = 0;
        value = Math.Clamp(value, -MaxNumber, MaxNumber);
        value = Math.Round(value, 4);
        if (value == 0) value = 0; // no negative zero
        var span = Reserve(24);
        value.TryFormat(span, out int n, "0.####", CultureInfo.InvariantCulture);
        Length += n;
        return this;
    }

    /// <summary>A 16-bit code as four hex digits, for the codes of an Identity-H string.</summary>
    public PdfBuffer Hex4(int code)
    {
        var span = Reserve(4);
        span[0] = HexDigit(code >> 12);
        span[1] = HexDigit(code >> 8);
        span[2] = HexDigit(code >> 4);
        span[3] = HexDigit(code);
        Length += 4;
        return this;
    }

    public byte[] ToArray() => Written.ToArray();

    /// <summary>The bytes written, compressed for a <c>/FlateDecode</c> stream.</summary>
    public byte[] Deflate() => Deflate(Written);

    /// <summary>Compresses bytes for a <c>/FlateDecode</c> stream (zlib format).</summary>
    public static byte[] Deflate(ReadOnlySpan<byte> data)
    {
        var output = new MemoryStream(data.Length / 3 + 64);
        using (var z = new ZLibStream(output, CompressionLevel.Optimal, leaveOpen: true)) z.Write(data);
        return output.ToArray();
    }

    private static byte HexDigit(int v) => (byte)"0123456789ABCDEF"[v & 15];

    private Span<byte> Reserve(int count)
    {
        if (Length + count > _data.Length)
        {
            long size = Math.Max((long)_data.Length * 2, (long)Length + count);
            if (size > Array.MaxLength) throw new PdfLimitException("The text layer is larger than 2 GB.");
            Array.Resize(ref _data, (int)size);
        }
        return _data.AsSpan(Length, count);
    }
}
