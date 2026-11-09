namespace Corpus;

/// <summary>
/// Degradations made on raw gray pixels with a seeded generator, so a scan's usual faults can be added to a clean page
/// without a decoder or any outside library. Geometric ones also move the ground truth boxes.
/// </summary>
static class Degrade
{
    /// <summary>Additive Gaussian noise with the given standard deviation in gray levels.</summary>
    public static Bitmap Noise(Bitmap src, double sigma, Rng rng)
    {
        var d = new byte[src.Data.Length];
        for (int i = 0; i < d.Length; i++) d[i] = Clamp(src.Data[i] + rng.NextGaussian() * sigma);
        return new Bitmap(src.Width, src.Height, 1, d);
    }

    /// <summary>A separable Gaussian blur with the given sigma in pixels, edges clamped.</summary>
    public static Bitmap Blur(Bitmap src, double sigma)
    {
        int radius = (int)Math.Ceiling(sigma * 3);
        var kernel = new double[radius * 2 + 1];
        double sum = 0;
        for (int i = -radius; i <= radius; i++) sum += kernel[i + radius] = Math.Exp(-(i * i) / (2 * sigma * sigma));
        for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;
        int w = src.Width, h = src.Height;
        var tmp = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double acc = 0;
                for (int k = -radius; k <= radius; k++) acc += kernel[k + radius] * src.Data[y * w + Math.Clamp(x + k, 0, w - 1)];
                tmp[y * w + x] = (float)acc;
            }
        var d = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double acc = 0;
                for (int k = -radius; k <= radius; k++) acc += kernel[k + radius] * tmp[Math.Clamp(y + k, 0, h - 1) * w + x];
                d[y * w + x] = Clamp(acc);
            }
        return new Bitmap(w, h, 1, d);
    }

    /// <summary>Rotates the page about its centre by the angle (degrees, positive = clockwise on screen), bilinear, white fill.</summary>
    public static Bitmap Rotate(Bitmap src, double degrees)
    {
        int w = src.Width, h = src.Height;
        double a = degrees * Math.PI / 180, cos = Math.Cos(a), sin = Math.Sin(a);
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
        var d = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                // The inverse mapping: where in the source does this destination pixel come from.
                double dx = x - cx, dy = y - cy;
                double sx = cos * dx + sin * dy + cx, sy = -sin * dx + cos * dy + cy;
                int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                double fx = sx - x0, fy = sy - y0;
                double v00 = Px(src, x0, y0), v10 = Px(src, x0 + 1, y0), v01 = Px(src, x0, y0 + 1), v11 = Px(src, x0 + 1, y0 + 1);
                d[y * w + x] = Clamp((v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy);
            }
        return new Bitmap(w, h, 1, d);
    }

    /// <summary>Moves a ground truth box with <see cref="Rotate"/>: the axis-aligned box of the rotated corners.</summary>
    public static WordBox RotateBox(WordBox b, int w, int h, double degrees)
    {
        double a = degrees * Math.PI / 180, cos = Math.Cos(a), sin = Math.Sin(a);
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var (px, py) in new[] { (b.X, b.Y), (b.X + b.Width, b.Y), (b.X, b.Y + b.Height), (b.X + b.Width, b.Y + b.Height) })
        {
            double dx = px - cx, dy = py - cy;
            double nx = cos * dx - sin * dy + cx, ny = sin * dx + cos * dy + cy;
            minX = Math.Min(minX, nx); maxX = Math.Max(maxX, nx);
            minY = Math.Min(minY, ny); maxY = Math.Max(maxY, ny);
        }
        return b with { X = minX, Y = minY, Width = maxX - minX, Height = maxY - minY };
    }

    /// <summary>The page turned upside down.</summary>
    public static Bitmap Rotate180(Bitmap src)
    {
        var d = new byte[src.Data.Length];
        for (int i = 0; i < d.Length; i++) d[i] = src.Data[d.Length - 1 - i];
        return new Bitmap(src.Width, src.Height, 1, d);
    }

    public static WordBox Rotate180Box(WordBox b, int w, int h) => b with { X = w - b.X - b.Width, Y = h - b.Y - b.Height };

    /// <summary>Squeezes the gray range into [low, high], as a faint photocopy does.</summary>
    public static Bitmap LowContrast(Bitmap src, int low, int high, double noiseSigma, Rng rng)
    {
        var d = new byte[src.Data.Length];
        for (int i = 0; i < d.Length; i++) d[i] = Clamp(low + src.Data[i] / 255.0 * (high - low) + rng.NextGaussian() * noiseSigma);
        return new Bitmap(src.Width, src.Height, 1, d);
    }

    static readonly int[] LumaTable =
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];

    /// <summary>
    /// What a baseline JPEG of the given quality does to a gray page: 8x8 DCT, the IJG luminance table scaled for the
    /// quality, rounding of every coefficient, and the inverse DCT. The entropy coding that follows is lossless, so the
    /// pixels are those a JPEG decoder would give. This avoids a JPEG decoder (PdfPecker's is internal) and resampling.
    /// </summary>
    public static Bitmap JpegQuality(Bitmap src, int quality)
    {
        int scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
        var q = new double[64];
        for (int i = 0; i < 64; i++) q[i] = Math.Clamp((LumaTable[i] * scale + 50) / 100, 1, 255);
        var cosT = new double[8, 8];
        for (int u = 0; u < 8; u++)
            for (int x = 0; x < 8; x++)
                cosT[u, x] = Math.Cos((2 * x + 1) * u * Math.PI / 16) * (u == 0 ? Math.Sqrt(0.125) : 0.5);
        int w = src.Width, h = src.Height;
        var d = new byte[w * h];
        var block = new double[64];
        var tmp = new double[64];
        for (int by = 0; by < h; by += 8)
            for (int bx = 0; bx < w; bx += 8)
            {
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        block[y * 8 + x] = src.Data[Math.Min(by + y, h - 1) * w + Math.Min(bx + x, w - 1)] - 128.0;
                // Forward DCT, quantise, inverse DCT.
                for (int y = 0; y < 8; y++)
                    for (int u = 0; u < 8; u++)
                    {
                        double s = 0;
                        for (int x = 0; x < 8; x++) s += block[y * 8 + x] * cosT[u, x];
                        tmp[y * 8 + u] = s;
                    }
                for (int u = 0; u < 8; u++)
                    for (int v = 0; v < 8; v++)
                    {
                        double s = 0;
                        for (int y = 0; y < 8; y++) s += tmp[y * 8 + u] * cosT[v, y];
                        block[v * 8 + u] = Math.Round(s / q[v * 8 + u]) * q[v * 8 + u];
                    }
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                    {
                        double s = 0;
                        for (int u = 0; u < 8; u++) s += block[y * 8 + u] * cosT[u, x];
                        tmp[y * 8 + x] = s;
                    }
                for (int x = 0; x < 8; x++)
                    for (int y = 0; y < 8; y++)
                    {
                        double s = 0;
                        for (int v = 0; v < 8; v++) s += tmp[v * 8 + x] * cosT[v, y];
                        if (by + y < h && bx + x < w) d[(by + y) * w + bx + x] = Clamp(s + 128);
                    }
            }
        return new Bitmap(w, h, 1, d);
    }

    static double Px(Bitmap b, int x, int y) => x < 0 || y < 0 || x >= b.Width || y >= b.Height ? 255 : b.Data[y * b.Width + x];

    static byte Clamp(double v) => (byte)Math.Clamp((int)Math.Round(v), 0, 255);
}
