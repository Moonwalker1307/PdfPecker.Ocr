using System.Buffers;

namespace PdfPecker.Ocr.Detection;

/// <summary>
/// Builds the detection network's input from page pixels: optional vertical padding, a bilinear resize to multiples
/// of 32, and rapidocr's normalisation <c>(px / 255 - 0.5) / 0.5</c> in its BGR channel order. Ported from rapidocr's
/// <c>DetPreProcess</c> and <c>apply_vertical_padding</c> (Apache-2.0, https://github.com/RapidAI/RapidOCR).
/// </summary>
internal static class DetectionInput
{
    // rapidocr computes (float32(px) * float32(1/255) - 0.5) / 0.5 with the subtraction and division in double and the
    // result stored as float32; a table of the 256 results reproduces that exactly.
    private static readonly float[] Normalize = BuildNormalize();

    private static float[] BuildNormalize()
    {
        var table = new float[256];
        float scale = (float)(1 / 255.0);
        for (int i = 0; i < 256; i++)
        {
            table[i] = (float)(((double)(i * scale) - 0.5) / 0.5);
        }

        return table;
    }

    /// <summary>The geometry of one detection run: padding, padded size and network input size.</summary>
    internal readonly record struct Layout(int PaddingTop, int PaddedWidth, int PaddedHeight, int InputWidth, int InputHeight);

    /// <summary>Works out the padding and the network input size for an image of the given size.</summary>
    public static Layout Plan(int width, int height, DetectionOptions options)
    {
        int pad = 0;
        if (options.VerticalPadding && (height <= options.PaddingMinHeight || (options.PaddingWidthHeightRatio > 0 && (double)width / height > options.PaddingWidthHeightRatio)))
        {
            // rapidocr's get_padding_h: the padded height is twice max(width / ratio, min_height).
            int newHeight = Math.Max((int)(width / options.PaddingWidthHeightRatio), options.PaddingMinHeight) * 2;
            pad = (int)(Math.Abs(newHeight - height) / 2.0);
        }

        int paddedHeight = height + (2 * pad);
        double ratio = options.Resize switch
        {
            DetectionResize.Min => Math.Min(width, paddedHeight) < options.MinSideLength
                ? (double)options.MinSideLength / Math.Min(width, paddedHeight)
                : 1.0,
            DetectionResize.Max => Math.Max(width, paddedHeight) > options.MaxSideLength
                ? (double)options.MaxSideLength / Math.Max(width, paddedHeight)
                : 1.0,
            _ => Math.Min(
                (double)options.MaxSideLength / Math.Max(width, paddedHeight),
                Math.Max(1.0, (double)options.MinSideLength / Math.Min(width, paddedHeight))),
        };

        // The pixel bound wins over every mode; it keeps the input within MaxInputPixels before rounding to 32.
        double pixels = (double)width * paddedHeight * ratio * ratio;
        if (pixels > options.MaxInputPixels)
        {
            ratio *= Math.Sqrt(options.MaxInputPixels / pixels);
        }

        int inputWidth = RoundTo32((int)(width * ratio));
        int inputHeight = RoundTo32((int)(paddedHeight * ratio));
        return new Layout(pad, width, paddedHeight, inputWidth, inputHeight);
    }

    // int(round(x / 32) * 32) with Python's round half to even, but at least 32 so tiny images still get an input.
    private static int RoundTo32(int size) => Math.Max(32, (int)Math.Round(size / 32.0, MidpointRounding.ToEven) * 32);

    /// <summary>
    /// Writes the network input for <paramref name="image"/> into <paramref name="input"/> (3 planes of
    /// <c>InputHeight x InputWidth</c>, in the order B, G, R as rapidocr feeds OpenCV's BGR images).
    /// </summary>
    public static void Fill(PixelView image, Layout layout, Span<float> input)
    {
        int dw = layout.InputWidth, dh = layout.InputHeight, plane = dw * dh;
        int planes = image.Channels == 1 ? 1 : 3;
        byte[] resized = ArrayPool<byte>.Shared.Rent(plane * planes);
        try
        {
            for (int p = 0; p < planes; p++)
            {
                // Plane p holds B, G, R in turn, which are channels 2, 1, 0 of an RGB image.
                int channel = planes == 1 ? 0 : 2 - p;
                LinearResize.Resize(image, channel, layout.PaddingTop, layout.PaddedHeight, resized.AsSpan(p * plane, plane), dw, dh);
            }

            float[] table = Normalize;
            for (int p = 0; p < 3; p++)
            {
                ReadOnlySpan<byte> src = resized.AsSpan((planes == 1 ? 0 : p) * plane, plane);
                Span<float> dst = input.Slice(p * plane, plane);
                for (int i = 0; i < src.Length; i++)
                {
                    dst[i] = table[src[i]];
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(resized);
        }
    }
}

/// <summary>
/// OpenCV's <c>cv2.resize</c> with <c>INTER_LINEAR</c> on 8-bit images: half-pixel centres, 11-bit fixed-point weights
/// and the rounding of OpenCV's vectorised vertical pass, so the result equals OpenCV's to within one grey level.
/// OpenCV switches to area averaging for an exact halving of both sides; this does not.
/// </summary>
internal static class LinearResize
{
    private const int CoefBits = 11;
    private const int CoefScale = 1 << CoefBits;

    /// <summary>
    /// Resizes one channel of <paramref name="image"/>, seen as padded with <paramref name="padTop"/> black rows above
    /// and as many below up to <paramref name="paddedHeight"/>, to <paramref name="dw"/> x <paramref name="dh"/>.
    /// </summary>
    public static void Resize(PixelView image, int channel, int padTop, int paddedHeight, Span<byte> dst, int dw, int dh)
    {
        int sw = image.Width, sh = paddedHeight;
        int[] rented = ArrayPool<int>.Shared.Rent((dw * 3) + (dw * 2));
        try
        {
            Span<int> xofs = rented.AsSpan(0, dw);
            Span<int> alpha = rented.AsSpan(dw, dw * 2);
            Span<int> row0 = rented.AsSpan(dw * 3, dw);
            Span<int> row1 = rented.AsSpan(dw * 4, dw);
            Coefficients(sw, dw, xofs, alpha);

            double scaleY = (double)sh / dh;
            int cached0 = int.MinValue, cached1 = int.MinValue;
            for (int dy = 0; dy < dh; dy++)
            {
                float fy = (float)(((dy + 0.5) * scaleY) - 0.5);
                int sy = (int)MathF.Floor(fy);
                fy -= sy;
                if (sy < 0)
                {
                    fy = 0;
                    sy = 0;
                }

                if (sy >= sh - 1)
                {
                    fy = 0;
                    sy = sh - 1;
                }

                int sy1 = Math.Min(sy + 1, sh - 1);

                // Reuse horizontally resized rows from the previous output row, as OpenCV does.
                if (sy == cached1)
                {
                    row1.CopyTo(row0);
                    cached0 = cached1;
                }

                if (sy != cached0)
                {
                    HorizontalRow(image, channel, sy - padTop, xofs, alpha, row0);
                    cached0 = sy;
                }

                if (sy1 != cached1)
                {
                    HorizontalRow(image, channel, sy1 - padTop, xofs, alpha, row1);
                    cached1 = sy1;
                }

                short b0 = SaturateShort((1f - fy) * CoefScale), b1 = SaturateShort(fy * CoefScale);
                Span<byte> d = dst.Slice(dy * dw, dw);
                for (int x = 0; x < dw; x++)
                {
                    // OpenCV's SIMD path: ((b0 * (s0 >> 4)) >> 16) + ((b1 * (s1 >> 4)) >> 16), then (v + 2) >> 2.
                    int v = (((b0 * (row0[x] >> 4)) >> 16) + ((b1 * (row1[x] >> 4)) >> 16) + 2) >> 2;
                    d[x] = (byte)Math.Clamp(v, 0, 255);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    private static void Coefficients(int sw, int dw, Span<int> xofs, Span<int> alpha)
    {
        double scaleX = (double)sw / dw;
        for (int dx = 0; dx < dw; dx++)
        {
            float fx = (float)(((dx + 0.5) * scaleX) - 0.5);
            int sx = (int)MathF.Floor(fx);
            fx -= sx;
            if (sx < 0)
            {
                fx = 0;
                sx = 0;
            }

            if (sx >= sw - 1)
            {
                fx = 0;
                sx = sw - 1;
            }

            xofs[dx] = sx;
            alpha[2 * dx] = SaturateShort((1f - fx) * CoefScale);
            alpha[(2 * dx) + 1] = SaturateShort(fx * CoefScale);
        }
    }

    private static void HorizontalRow(PixelView image, int channel, int y, ReadOnlySpan<int> xofs, ReadOnlySpan<int> alpha, Span<int> row)
    {
        if ((uint)y >= (uint)image.Height)
        {
            // A padding row is black.
            row.Clear();
            return;
        }

        int cn = image.Channels, sw = image.Width;
        ReadOnlySpan<byte> src = image.Data.Slice(y * sw * cn, sw * cn);
        for (int x = 0; x < row.Length; x++)
        {
            int sx = xofs[x];
            int sx1 = Math.Min(sx + 1, sw - 1);
            row[x] = (src[(sx * cn) + channel] * alpha[2 * x]) + (src[(sx1 * cn) + channel] * alpha[(2 * x) + 1]);
        }
    }

    // saturate_cast<short>(float) rounds to nearest, ties to even, as cvRound does.
    private static short SaturateShort(float v) => (short)Math.Clamp((int)MathF.Round(v, MidpointRounding.ToEven), short.MinValue, short.MaxValue);
}
