using System.Buffers.Binary;

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

}
