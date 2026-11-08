using System.IO.Compression;
using System.Text;

namespace Corpus;

/// <summary>A word box in pixels, origin top-left.</summary>
readonly record struct WordBox(string Text, double X, double Y, double Width, double Height);

/// <summary>An 8-bit image with one (gray) or three (RGB) channels.</summary>
sealed class Bitmap(int width, int height, int channels, byte[] data)
{
    public int Width { get; } = width;
    public int Height { get; } = height;
    public int Channels { get; } = channels;
    public byte[] Data { get; } = data;

    public static Bitmap FromRgba(byte[] rgba, int width, int height, bool colour)
    {
        if (colour)
        {
            var rgb = new byte[width * height * 3];
            for (int i = 0, j = 0; i < rgba.Length; i += 4, j += 3) { rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2]; }
            return new Bitmap(width, height, 3, rgb);
        }
        var gray = new byte[width * height];
        for (int i = 0, j = 0; j < gray.Length; i += 4, j++) gray[j] = (byte)((rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29 + 128) >> 8);
        return new Bitmap(width, height, 1, gray);
    }

    /// <summary>Writes a PNG with one IDAT stream and no filtering beyond Sub, which is plenty for text pages.</summary>
    public byte[] ToPng()
    {
        int stride = Width * Channels;
        var raw = new byte[(stride + 1) * Height];
        for (int y = 0; y < Height; y++)
        {
            int o = y * (stride + 1);
            raw[o] = 1; // Sub
            for (int x = 0; x < stride; x++)
            {
                byte left = x >= Channels ? Data[y * stride + x - Channels] : (byte)0;
                raw[o + 1 + x] = (byte)(Data[y * stride + x] - left);
            }
        }
        using var ms = new MemoryStream();
        ms.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        WriteBe(ihdr, 0, Width);
        WriteBe(ihdr, 4, Height);
        ihdr[8] = 8;
        ihdr[9] = (byte)(Channels == 1 ? 0 : 2);
        Chunk(ms, "IHDR", ihdr);
        using var z = new MemoryStream();
        using (var zs = new ZLibStream(z, CompressionLevel.Optimal, leaveOpen: true)) zs.Write(raw);
        Chunk(ms, "IDAT", z.ToArray());
        Chunk(ms, "IEND", []);
        return ms.ToArray();
    }

    static void WriteBe(byte[] b, int o, int v)
    {
        b[o] = (byte)(v >> 24); b[o + 1] = (byte)(v >> 16); b[o + 2] = (byte)(v >> 8); b[o + 3] = (byte)v;
    }

    static void Chunk(Stream s, string type, byte[] data)
    {
        var len = new byte[4];
        WriteBe(len, 0, data.Length);
        s.Write(len);
        var t = Encoding.ASCII.GetBytes(type);
        s.Write(t);
        s.Write(data);
        uint crc = 0xFFFFFFFF;
        foreach (var b in t) crc = Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        foreach (var b in data) crc = Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        crc ^= 0xFFFFFFFF;
        var c = new byte[4];
        WriteBe(c, 0, (int)crc);
        s.Write(c);
    }

    static readonly uint[] Table = MakeTable();

    static uint[] MakeTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }
}
