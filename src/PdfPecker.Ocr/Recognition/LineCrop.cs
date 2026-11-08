using System.Numerics;
using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// A detected quad seen as a horizontal strip of text, the way rapidocr's <c>get_rotate_crop_image</c> cuts it from
/// the page: the quad is warped to a <see cref="CropWidth"/> x <see cref="CropHeight"/> rectangle, turned 90 degrees
/// counter-clockwise when it is at least 1.5 times as tall as wide, and turned 180 degrees when the orientation
/// classifier says the text is upside down.
/// </summary>
/// <remarks>
/// rapidocr warps every crop into an image of its own (bicubic) and then resizes it to the network's height of 48
/// (bilinear). Here the crop exists only as a mapping from strip coordinates to page coordinates, and the network
/// input is sampled straight from the page pixels with one bilinear lookup per input pixel: no intermediate image,
/// and full page resolution even when the detector ran on a smaller copy.
/// </remarks>
internal sealed class LineCrop
{
    // The homography from the unit square to the quad: (u, v) -> ((a u + b v + c) / (g u + h v + 1), ...).
    private readonly double _a, _b, _c, _d, _e, _f, _g, _h;

    public LineCrop(Quad quad, int index)
    {
        Quad = quad;
        Index = index;
        Vector2 p0 = quad.P0, p1 = quad.P1, p2 = quad.P2, p3 = quad.P3;

        // int(max(norm(p0 - p1), norm(p2 - p3))) and the same for the height, as rapidocr sizes the crop; at least
        // one pixel so a degenerate quad still maps somewhere.
        WarpWidth = Math.Max(1, (int)Math.Max(Vector2.Distance(p0, p1), Vector2.Distance(p2, p3)));
        WarpHeight = Math.Max(1, (int)Math.Max(Vector2.Distance(p0, p3), Vector2.Distance(p1, p2)));
        Rotated90 = WarpHeight * 1.0 / WarpWidth >= 1.5;

        // Heckbert's square-to-quad mapping, in double precision.
        double sx = p0.X - p1.X + p2.X - p3.X, sy = p0.Y - p1.Y + p2.Y - p3.Y;
        double dx1 = p1.X - p2.X, dx2 = p3.X - p2.X, dy1 = p1.Y - p2.Y, dy2 = p3.Y - p2.Y;
        double den = (dx1 * dy2) - (dx2 * dy1);
        if ((sx == 0 && sy == 0) || Math.Abs(den) < 1e-12)
        {
            _g = 0;
            _h = 0;
        }
        else
        {
            _g = ((sx * dy2) - (dx2 * sy)) / den;
            _h = ((dx1 * sy) - (sx * dy1)) / den;
        }

        _a = p1.X - p0.X + (_g * p1.X);
        _b = p3.X - p0.X + (_h * p3.X);
        _c = p0.X;
        _d = p1.Y - p0.Y + (_g * p1.Y);
        _e = p3.Y - p0.Y + (_h * p3.Y);
        _f = p0.Y;
    }

    public Quad Quad { get; }

    /// <summary>The position of the quad in the detector's output, which results are mapped back to.</summary>
    public int Index { get; }

    /// <summary>The size of the warped rectangle before any turn.</summary>
    public int WarpWidth { get; }

    public int WarpHeight { get; }

    /// <summary>Whether the strip is turned 90 degrees counter-clockwise, for a tall quad.</summary>
    public bool Rotated90 { get; }

    /// <summary>Whether the strip is turned 180 degrees, set when the orientation classifier says so.</summary>
    public bool Rotated180 { get; set; }

    /// <summary>The strip's width after the turns: the length along the text.</summary>
    public int CropWidth => Rotated90 ? WarpHeight : WarpWidth;

    /// <summary>The strip's height after the turns.</summary>
    public int CropHeight => Rotated90 ? WarpWidth : WarpHeight;

    /// <summary>Width over height, which rapidocr sorts and sizes batches by.</summary>
    public double AspectRatio => CropWidth / (double)CropHeight;

    /// <summary>Maps a point of the strip (after the turns, pixel centres at whole numbers) to the page.</summary>
    public Vector2 ToPage(double x, double y)
    {
        if (Rotated180)
        {
            x = CropWidth - 1 - x;
            y = CropHeight - 1 - y;
        }

        if (Rotated90)
        {
            // np.rot90 turns counter-clockwise: strip (x, y) is warped pixel (warpWidth - 1 - y, x).
            (x, y) = (WarpWidth - 1 - y, x);
        }

        // warpPerspective maps the rectangle's corners (0, 0) and (width, height) onto the quad's corners.
        double u = x / WarpWidth, v = y / WarpHeight;
        double w = (_g * u) + (_h * v) + 1;
        return new Vector2((float)(((_a * u) + (_b * v) + _c) / w), (float)(((_d * u) + (_e * v) + _f) / w));
    }

    /// <summary>
    /// Maps a point given in the strip's continuous coordinates, where the strip spans 0 to CropWidth and 0 to
    /// CropHeight, to the page. Unlike <see cref="ToPage"/> this maps edges, not pixel centres, so a strip corner
    /// lands on the matching quad corner.
    /// </summary>
    public Vector2 EdgeToPage(double x, double y)
    {
        if (Rotated180)
        {
            x = CropWidth - x;
            y = CropHeight - y;
        }

        if (Rotated90)
        {
            (x, y) = (WarpWidth - y, x);
        }

        double u = x / WarpWidth, v = y / WarpHeight;
        double w = (_g * u) + (_h * v) + 1;
        return new Vector2((float)(((_a * u) + (_b * v) + _c) / w), (float)(((_d * u) + (_e * v) + _f) / w));
    }

    /// <summary>
    /// Samples the part of the strip from <paramref name="windowX"/> that is <paramref name="windowWidth"/> long
    /// (the whole strip for recognition, a window of it for the orientation classifier), resized to
    /// <paramref name="width"/> x <paramref name="height"/> as OpenCV's resize maps pixels (half-pixel centres,
    /// clamped at the edges), into the three planes of <paramref name="planes"/>, whose rows are
    /// <paramref name="stride"/> floats apart, normalised as rapidocr does. The planes hold B, G, R as rapidocr feeds
    /// them. Pixels outside the page repeat the nearest edge pixel, as the replicated border of rapidocr's warp does.
    /// </summary>
    public void Sample(PixelView page, Span<float> planes, int planeSize, int stride, int width, int height, double windowX, double windowWidth)
    {
        double scaleX = windowWidth / width, scaleY = (double)CropHeight / height;
        int cn = page.Channels, pw = page.Width, ph = page.Height;
        ReadOnlySpan<byte> data = page.Data;
        float[] table = RecNormalize;
        for (int v = 0; v < height; v++)
        {
            double y = Math.Clamp(((v + 0.5) * scaleY) - 0.5, 0, CropHeight - 1);
            for (int u = 0; u < width; u++)
            {
                double x = windowX + Math.Clamp(((u + 0.5) * scaleX) - 0.5, 0, windowWidth - 1);
                Vector2 p = ToPage(x, y);
                float fx = Math.Clamp(p.X, 0, pw - 1), fy = Math.Clamp(p.Y, 0, ph - 1);
                if (float.IsNaN(fx) || float.IsNaN(fy))
                {
                    fx = 0;
                    fy = 0;
                }

                int x0 = (int)fx, y0 = (int)fy;
                int x1 = Math.Min(x0 + 1, pw - 1), y1 = Math.Min(y0 + 1, ph - 1);
                float ax = fx - x0, ay = fy - y0;
                int i00 = ((y0 * pw) + x0) * cn, i01 = ((y0 * pw) + x1) * cn, i10 = ((y1 * pw) + x0) * cn, i11 = ((y1 * pw) + x1) * cn;
                int offset = (v * stride) + u;
                if (cn == 1)
                {
                    float n = table[(int)(Lerp(data[i00], data[i01], data[i10], data[i11], ax, ay) + 0.5f)];
                    planes[offset] = n;
                    planes[planeSize + offset] = n;
                    planes[(2 * planeSize) + offset] = n;
                    continue;
                }

                for (int c = 0; c < 3; c++)
                {
                    // Plane 0 is blue, channel 2 of RGB.
                    int ch = 2 - c;
                    planes[(c * planeSize) + offset] = table[(int)(Lerp(data[i00 + ch], data[i01 + ch], data[i10 + ch], data[i11 + ch], ax, ay) + 0.5f)];
                }
            }
        }
    }

    private static float Lerp(byte p00, byte p01, byte p10, byte p11, float ax, float ay)
    {
        float top = p00 + ((p01 - p00) * ax);
        float bottom = p10 + ((p11 - p10) * ax);
        return top + ((bottom - top) * ay);
    }

    // rapidocr's rec and cls inputs: float32(px) / 255, then - 0.5 and / 0.5, all in float32. The sampled value is
    // rounded to a whole grey level first, as the 8-bit images rapidocr resizes are.
    private static readonly float[] RecNormalize = BuildNormalize();

    private static float[] BuildNormalize()
    {
        var table = new float[256];
        for (int i = 0; i < 256; i++)
        {
            table[i] = ((i / 255f) - 0.5f) / 0.5f;
        }

        return table;
    }
}
