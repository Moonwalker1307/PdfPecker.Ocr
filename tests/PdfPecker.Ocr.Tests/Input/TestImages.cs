using System.Buffers.Binary;
using System.IO.Compression;

namespace PdfPecker.Ocr.Tests;

/// <summary>Image files made in code, for formats PdfPecker reads but does not write.</summary>
internal static class TestImages
{
    /// <summary>A bottom-up 24-bit BMP of RGB pixels.</summary>
    public static byte[] Bmp(byte[] rgb, int w, int h, int pixelsPerMetre)
    {
        int stride = (w * 3 + 3) & ~3;
        var bmp = new byte[54 + stride * h];
        var s = bmp.AsSpan();
        "BM"u8.CopyTo(s);
        BinaryPrimitives.WriteInt32LittleEndian(s[2..], bmp.Length);
        BinaryPrimitives.WriteInt32LittleEndian(s[10..], 54);
        BinaryPrimitives.WriteInt32LittleEndian(s[14..], 40);
        BinaryPrimitives.WriteInt32LittleEndian(s[18..], w);
        BinaryPrimitives.WriteInt32LittleEndian(s[22..], h);
        BinaryPrimitives.WriteInt16LittleEndian(s[26..], 1);
        BinaryPrimitives.WriteInt16LittleEndian(s[28..], 24);
        BinaryPrimitives.WriteInt32LittleEndian(s[34..], stride * h);
        BinaryPrimitives.WriteInt32LittleEndian(s[38..], pixelsPerMetre);
        BinaryPrimitives.WriteInt32LittleEndian(s[42..], pixelsPerMetre);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int src = (y * w + x) * 3, dst = 54 + (h - 1 - y) * stride + x * 3;
                bmp[dst] = rgb[src + 2];
                bmp[dst + 1] = rgb[src + 1];
                bmp[dst + 2] = rgb[src];
            }
        return bmp;
    }

    /// <summary>
    /// A GIF of gray pixels with a 256-level gray palette. The LZW stream sends a clear code before the code width would
    /// grow past nine bits, which is valid LZW without a dictionary search, so the encoder stays short.
    /// </summary>
    public static byte[] Gif(byte[] gray, int w, int h)
    {
        var gif = new List<byte>();
        gif.AddRange("GIF89a"u8.ToArray());
        gif.AddRange([(byte)w, (byte)(w >> 8), (byte)h, (byte)(h >> 8), 0xF7, 0, 0]);
        for (int i = 0; i < 256; i++) gif.AddRange([(byte)i, (byte)i, (byte)i]);
        gif.AddRange([0x2C, 0, 0, 0, 0, (byte)w, (byte)(w >> 8), (byte)h, (byte)(h >> 8), 0, 8]);

        var codes = new List<byte>();
        int bitBuffer = 0, bits = 0;
        void Emit(int code)
        {
            bitBuffer |= code << bits;
            bits += 9;
            while (bits >= 8)
            {
                codes.Add((byte)bitBuffer);
                bitBuffer >>= 8;
                bits -= 8;
            }
        }

        const int Clear = 256, End = 257;
        Emit(Clear);
        int sinceClear = 0;
        for (int i = 0; i < w * h; i++)
        {
            // Each literal after the first adds a dictionary entry; 254 of them keep the next code below 512.
            if (sinceClear == 254)
            {
                Emit(Clear);
                sinceClear = 0;
            }

            Emit(gray[i]);
            sinceClear++;
        }

        Emit(End);
        if (bits > 0) codes.Add((byte)bitBuffer);
        for (int i = 0; i < codes.Count; i += 255)
        {
            int n = Math.Min(255, codes.Count - i);
            gif.Add((byte)n);
            gif.AddRange(codes.GetRange(i, n));
        }

        gif.Add(0);
        gif.Add(0x3B);
        return gif.ToArray();
    }

    /// <summary>
    /// An uncompressed 8-bit gray TIFF in one strip whose header states <paramref name="w"/> by <paramref name="h"/>
    /// pixels, holding <paramref name="data"/> as the strip: shorter than the stated size, it makes a damaged or
    /// bomb-like file.
    /// </summary>
    public static byte[] Tiff(int w, int h, byte[] data)
    {
        const int Entries = 9;
        int ifd = 8, stripOffset = ifd + 2 + (Entries * 12) + 4;
        var tiff = new byte[stripOffset + data.Length];
        "II"u8.CopyTo(tiff);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(2), 42);
        BinaryPrimitives.WriteUInt32LittleEndian(tiff.AsSpan(4), (uint)ifd);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(ifd), Entries);
        int at = ifd + 2;
        void Tag(ushort tag, ushort type, uint value)
        {
            Span<byte> s = tiff;
            BinaryPrimitives.WriteUInt16LittleEndian(s[at..], tag);
            BinaryPrimitives.WriteUInt16LittleEndian(s[(at + 2)..], type);
            BinaryPrimitives.WriteUInt32LittleEndian(s[(at + 4)..], 1);
            if (type == 3) BinaryPrimitives.WriteUInt16LittleEndian(s[(at + 8)..], (ushort)value);
            else BinaryPrimitives.WriteUInt32LittleEndian(s[(at + 8)..], value);
            at += 12;
        }

        Tag(256, 4, (uint)w);
        Tag(257, 4, (uint)h);
        Tag(258, 3, 8);
        Tag(259, 3, 1);
        Tag(262, 3, 1);
        Tag(273, 4, (uint)stripOffset);
        Tag(277, 3, 1);
        Tag(278, 4, (uint)h);
        Tag(279, 4, (uint)data.Length);
        data.CopyTo(tiff.AsSpan(stripOffset));
        return tiff;
    }

    /// <summary>
    /// A PNG of the given size and bit depth (gray, colour type 0) whose image data is <paramref name="rows"/> rows of
    /// zeros, each with filter byte 0, deflated as they are written: a few hundred kilobytes can state billions of
    /// pixels. Rows short of the height make the data end early.
    /// </summary>
    public static byte[] ZeroPng(int w, int h, int bitDepth, int rows)
    {
        long rowBytes = 1 + (((long)w * bitDepth) + 7) / 8;
        var idat = new MemoryStream();
        using (var z = new ZLibStream(idat, CompressionLevel.Fastest, leaveOpen: true))
        {
            var row = new byte[rowBytes];
            for (int y = 0; y < rows; y++) z.Write(row);
        }

        var png = new MemoryStream();
        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr, (uint)w);
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(4), (uint)h);
        ihdr[8] = (byte)bitDepth;
        PngChunk(png, "IHDR", ihdr);
        PngChunk(png, "IDAT", idat.ToArray());
        PngChunk(png, "IEND", []);
        return png.ToArray();
    }

    /// <summary>Writes a PNG chunk with its CRC, so a changed chunk is still a well-formed one.</summary>
    public static void PngChunk(Stream png, string type, byte[] data)
    {
        Span<byte> four = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(four, (uint)data.Length);
        png.Write(four);
        byte[] typed = [.. System.Text.Encoding.ASCII.GetBytes(type), .. data];
        png.Write(typed);
        BinaryPrimitives.WriteUInt32BigEndian(four, Crc32(typed));
        png.Write(four);
    }

    /// <summary>A PNG with its IHDR width and height changed, and the chunk's CRC written again to match.</summary>
    public static byte[] WithPngSize(byte[] png, uint w, uint h)
    {
        var copy = (byte[])png.Clone();
        BinaryPrimitives.WriteUInt32BigEndian(copy.AsSpan(16), w);
        BinaryPrimitives.WriteUInt32BigEndian(copy.AsSpan(20), h);
        BinaryPrimitives.WriteUInt32BigEndian(copy.AsSpan(29), Crc32(copy.AsSpan(12, 17)));
        return copy;
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? 0xEDB88320 ^ (crc >> 1) : crc >> 1;
        }

        return ~crc;
    }
}
