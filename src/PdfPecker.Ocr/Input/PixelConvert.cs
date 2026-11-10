using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace PdfPecker.Ocr.Input;

/// <summary>
/// Converts RGBA pixels to RGB or gray on a white background. PdfPecker hands over every rendered page as RGBA, so
/// this runs once per page over tens of megabytes; the vector paths handle the usual opaque pixels four or sixteen at a
/// time and leave transparent ones to the scalar code.
/// </summary>
internal static class PixelConvert
{
    /// <summary>Whether every pixel has equal red, green and blue, so the image loses nothing as gray.</summary>
    public static bool IsGray(ReadOnlySpan<byte> rgba)
    {
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            // Each pixel's red is compared with its green and its green with its blue; blue and alpha with themselves.
            var perm = Vector128.Create((byte)1, 2, 2, 3, 5, 6, 6, 7, 9, 10, 10, 11, 13, 14, 14, 15);
            ref byte src = ref MemoryMarshal.GetReference(rgba);
            for (; i + 16 <= rgba.Length; i += 16)
            {
                var v = Vector128.LoadUnsafe(ref src, (nuint)i);
                if (v != Vector128.Shuffle(v, perm)) return false;
            }
        }
        for (; i + 4 <= rgba.Length; i += 4)
            if (rgba[i] != rgba[i + 1] || rgba[i + 1] != rgba[i + 2]) return false;
        return true;
    }

    /// <summary>Whether every RGB pixel has equal red, green and blue.</summary>
    public static bool IsGrayRgb(ReadOnlySpan<byte> rgb)
    {
        for (int i = 0; i + 3 <= rgb.Length; i += 3)
            if (rgb[i] != rgb[i + 1] || rgb[i + 1] != rgb[i + 2]) return false;
        return true;
    }

    /// <summary>Converts RGB whose channels are equal to gray, taking the red channel.</summary>
    public static void RgbToGray(ReadOnlySpan<byte> rgb, Span<byte> gray)
    {
        int pixels = rgb.Length / 3;
        if (gray.Length < pixels) throw new ArgumentException("The gray buffer is too short.", nameof(gray));
        for (int p = 0; p < pixels; p++) gray[p] = rgb[p * 3];
    }

    /// <summary>Converts RGBA to RGB; <paramref name="rgb"/> holds three quarters of <paramref name="rgba"/>'s length.</summary>
    public static void RgbaToRgb(ReadOnlySpan<byte> rgba, Span<byte> rgb)
    {
        int pixels = rgba.Length / 4;
        if (rgb.Length < pixels * 3) throw new ArgumentException("The RGB buffer is too short.", nameof(rgb));
        int p = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            var alpha = Vector128.Create(0xFF000000u).AsByte();
            var pack = Vector128.Create((byte)0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80);
            ref byte src = ref MemoryMarshal.GetReference(rgba);
            ref byte dst = ref MemoryMarshal.GetReference(rgb);
            // Each step stores 16 bytes but advances 12, so it stops while the store still fits.
            for (; p + 4 <= pixels && p * 3 + 16 <= rgb.Length; p += 4)
            {
                var v = Vector128.LoadUnsafe(ref src, (nuint)(p * 4));
                if ((v & alpha) != alpha)
                {
                    for (int k = 0; k < 4; k++) Composite(rgba.Slice((p + k) * 4, 4), rgb.Slice((p + k) * 3, 3));
                    continue;
                }
                Vector128.Shuffle(v, pack).StoreUnsafe(ref dst, (nuint)(p * 3));
            }
        }
        for (; p < pixels; p++) Composite(rgba.Slice(p * 4, 4), rgb.Slice(p * 3, 3));
    }

    /// <summary>Converts RGBA whose channels are equal to gray, taking the red channel; <paramref name="gray"/> holds a quarter of <paramref name="rgba"/>'s length.</summary>
    public static void RgbaToGray(ReadOnlySpan<byte> rgba, Span<byte> gray)
    {
        int pixels = rgba.Length / 4;
        if (gray.Length < pixels) throw new ArgumentException("The gray buffer is too short.", nameof(gray));
        int p = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            var alpha = Vector128.Create(0xFF000000u);
            var low = Vector128.Create(0xFFu);
            ref byte src = ref MemoryMarshal.GetReference(rgba);
            ref byte dst = ref MemoryMarshal.GetReference(gray);
            for (; p + 16 <= pixels; p += 16)
            {
                var a = Vector128.LoadUnsafe(ref src, (nuint)(p * 4)).AsUInt32();
                var b = Vector128.LoadUnsafe(ref src, (nuint)(p * 4 + 16)).AsUInt32();
                var c = Vector128.LoadUnsafe(ref src, (nuint)(p * 4 + 32)).AsUInt32();
                var d = Vector128.LoadUnsafe(ref src, (nuint)(p * 4 + 48)).AsUInt32();
                if (((a & b & c & d) & alpha) != alpha)
                {
                    for (int k = 0; k < 16; k++) gray[p + k] = CompositeChannel(rgba[(p + k) * 4], rgba[(p + k) * 4 + 3]);
                    continue;
                }
                var ab = Vector128.Narrow(a & low, b & low);
                var cd = Vector128.Narrow(c & low, d & low);
                Vector128.Narrow(ab, cd).StoreUnsafe(ref dst, (nuint)p);
            }
        }
        for (; p < pixels; p++) gray[p] = CompositeChannel(rgba[p * 4], rgba[p * 4 + 3]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Composite(ReadOnlySpan<byte> px, Span<byte> rgb)
    {
        byte a = px[3];
        rgb[0] = CompositeChannel(px[0], a);
        rgb[1] = CompositeChannel(px[1], a);
        rgb[2] = CompositeChannel(px[2], a);
    }

    /// <summary>One channel of a straight-alpha pixel drawn on white, rounded to the nearest value.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte CompositeChannel(byte value, byte alpha)
        => alpha == 255 ? value : (byte)((value * alpha + 255 * (255 - alpha) + 127) / 255);
}
