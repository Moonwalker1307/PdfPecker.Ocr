using System.Buffers;

namespace PdfPecker.Ocr.Detection;

/// <summary>A point in double precision, for the geometry of the detection post-processing.</summary>
internal readonly record struct PointD(double X, double Y);

/// <summary>
/// The plane geometry DB post-processing needs: convex hulls, minimum-area rectangles, the rounded polygon offset
/// that "unclips" a shrunk text kernel, and the mean of a probability map inside a polygon. Each mirrors the routine
/// rapidocr calls (OpenCV's <c>minAreaRect</c>, <c>boxPoints</c> and <c>fillPoly</c>, Clipper 6.4's
/// <c>ClipperOffset</c> as wrapped by pyclipper) closely enough that the boxes match rapidocr's to the pixel on the
/// test pages, without their allocations.
/// </summary>
internal static class Geometry
{
    /// <summary>
    /// Computes the convex hull of points sorted by (y, x), as the raster scan of a component produces them, with
    /// Andrew's monotone chain. Collinear points are dropped. Returns the number of hull points written to
    /// <paramref name="hull"/>, which must hold <c>2 * points.Length</c> entries.
    /// </summary>
    public static int ConvexHullSorted(ReadOnlySpan<PointD> points, Span<PointD> hull)
    {
        int n = points.Length;
        if (n < 3)
        {
            points.CopyTo(hull);
            return n == 2 && points[0] == points[1] ? 1 : n;
        }

        int k = 0;
        for (int i = 0; i < n; i++)
        {
            while (k >= 2 && Cross(hull[k - 2], hull[k - 1], points[i]) <= 0)
            {
                k--;
            }

            hull[k++] = points[i];
        }

        for (int i = n - 2, lower = k + 1; i >= 0; i--)
        {
            while (k >= lower && Cross(hull[k - 2], hull[k - 1], points[i]) <= 0)
            {
                k--;
            }

            hull[k++] = points[i];
        }

        // The last point repeats the first.
        return k - 1;
    }

    private static double Cross(PointD o, PointD a, PointD b) => ((a.X - o.X) * (b.Y - o.Y)) - ((a.Y - o.Y) * (b.X - o.X));

    /// <summary>
    /// Puts a convex hull into the order OpenCV's <c>convexHull(points, clockwise: true)</c> returns inside
    /// <c>minAreaRect</c>, because OpenCV's rotating calipers work in single precision and their rounding depends on
    /// where the hull starts: counter-clockwise as seen on the page (y down), starting at the topmost point, the
    /// leftmost of a tie for a region's contour (whose tracing starts there) or the rightmost for an offset polygon
    /// (whose last point Clipper leaves there).
    /// </summary>
    public static void OrderHullLikeOpenCv(Span<PointD> hull, bool leftmostStart)
    {
        int n = hull.Length;
        if (n < 3)
        {
            return;
        }

        double area = 0;
        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            area += (hull[j].X * hull[i].Y) - (hull[i].X * hull[j].Y);
        }

        if (area > 0)
        {
            hull.Reverse();
        }

        int start = 0;
        for (int i = 1; i < n; i++)
        {
            PointD p = hull[i], s = hull[start];
            if (p.Y < s.Y || (p.Y == s.Y && (leftmostStart ? p.X < s.X : p.X > s.X)))
            {
                start = i;
            }
        }

        hull[..start].Reverse();
        hull[start..].Reverse();
        hull.Reverse();
    }

    /// <summary>
    /// OpenCV's <c>minAreaRect</c> on a hull in <see cref="OrderHullLikeOpenCv"/> order: the rotating calipers of
    /// OpenCV's <c>rotcalipers.cpp</c> (Apache-2.0) in single precision, then <c>boxPoints</c>, so the corners carry
    /// the same rounding as rapidocr's (a centre of 240.49997 instead of 240.5 moves a truncated corner by a pixel).
    /// Writes the four corners in <c>boxPoints</c> order and returns the shorter side, 0 for fewer than three points.
    /// </summary>
    public static float MinAreaRect(ReadOnlySpan<PointD> hull, Span<PointD> corners)
    {
        int n = hull.Length;
        if (n < 3)
        {
            // OpenCV gives a point or a segment a height of 0, which every caller drops.
            for (int i = 0; i < 4; i++)
            {
                corners[i] = n == 0 ? default : hull[i < 2 || n == 1 ? 0 : n - 1];
            }

            return 0;
        }

        Span<float> vx = n <= 256 ? stackalloc float[n] : new float[n];
        Span<float> vy = n <= 256 ? stackalloc float[n] : new float[n];
        Span<float> inv = n <= 256 ? stackalloc float[n] : new float[n];
        int left = 0, bottom = 0, right = 0, top = 0;
        float leftX = (float)hull[0].X, rightX = leftX, topY = (float)hull[0].Y, bottomY = topY;
        for (int i = 0; i < n; i++)
        {
            float x = (float)hull[i].X, y = (float)hull[i].Y;
            if (x < leftX)
            {
                leftX = x;
                left = i;
            }

            if (x > rightX)
            {
                rightX = x;
                right = i;
            }

            if (y > topY)
            {
                topY = y;
                top = i;
            }

            if (y < bottomY)
            {
                bottomY = y;
                bottom = i;
            }

            PointD next = hull[i + 1 < n ? i + 1 : 0];
            double dx = (float)next.X - (double)x, dy = (float)next.Y - (double)y;
            vx[i] = (float)dx;
            vy[i] = (float)dy;
            inv[i] = (float)(1.0 / Math.Sqrt((dx * dx) + (dy * dy)));
        }

        float orientation = 0;
        double ax = vx[n - 1], ay = vy[n - 1];
        for (int i = 0; i < n; i++)
        {
            double bx = vx[i], by = vy[i];
            double convexity = (ax * by) - (ay * bx);
            if (convexity != 0)
            {
                orientation = convexity > 0 ? 1f : -1f;
                break;
            }

            ax = bx;
            ay = by;
        }

        if (orientation == 0)
        {
            corners.Clear();
            return 0;
        }

        float baseA = orientation, baseB = 0;
        Span<int> seq = [bottom, right, top, left];
        Span<float> dp = stackalloc float[4];
        float minArea = float.MaxValue;
        int bestLeft = 0, bestBottom = 0;
        float bestA = 0, bestB = 0, bestWidth = 0, bestHeight = 0;
        for (int k = 0; k < n; k++)
        {
            dp[0] = (baseA * vx[seq[0]]) + (baseB * vy[seq[0]]);
            dp[1] = (-baseB * vx[seq[1]]) + (baseA * vy[seq[1]]);
            dp[2] = (-baseA * vx[seq[2]]) - (baseB * vy[seq[2]]);
            dp[3] = (baseB * vx[seq[3]]) - (baseA * vy[seq[3]]);
            float maxCos = dp[0] * inv[seq[0]];
            int main = 0;
            for (int i = 1; i < 4; i++)
            {
                float cos = dp[i] * inv[seq[i]];
                if (cos > maxCos)
                {
                    main = i;
                    maxCos = cos;
                }
            }

            int p = seq[main];
            float leadX = vx[p] * inv[p], leadY = vy[p] * inv[p];
            (baseA, baseB) = main switch
            {
                0 => (leadX, leadY),
                1 => (leadY, -leadX),
                2 => (-leadX, -leadY),
                _ => (-leadY, leadX),
            };

            seq[main] = seq[main] + 1 == n ? 0 : seq[main] + 1;

            float wx = (float)hull[seq[1]].X - (float)hull[seq[3]].X, wy = (float)hull[seq[1]].Y - (float)hull[seq[3]].Y;
            float width = (wx * baseA) + (wy * baseB);
            float hx = (float)hull[seq[2]].X - (float)hull[seq[0]].X, hy = (float)hull[seq[2]].Y - (float)hull[seq[0]].Y;
            float height = (-hx * baseB) + (hy * baseA);
            float area = width * height;
            if (area <= minArea)
            {
                minArea = area;
                bestLeft = seq[3];
                bestA = baseA;
                bestWidth = width;
                bestB = baseB;
                bestHeight = height;
                bestBottom = seq[0];
            }
        }

        float a1 = bestA, b1 = bestB, a2 = -bestB, b2 = bestA;
        float c1 = (a1 * (float)hull[bestLeft].X) + ((float)hull[bestLeft].Y * b1);
        float c2 = (a2 * (float)hull[bestBottom].X) + ((float)hull[bestBottom].Y * b2);
        float idet = 1f / ((a1 * b2) - (a2 * b1));
        float px = ((c1 * b2) - (c2 * b1)) * idet;
        float py = ((a1 * c2) - (a2 * c1)) * idet;
        float o1x = a1 * bestWidth, o1y = b1 * bestWidth, o2x = a2 * bestHeight, o2y = b2 * bestHeight;

        float cx = px + ((o1x + o2x) * 0.5f);
        float cy = py + ((o1y + o2y) * 0.5f);
        float w = (float)Math.Sqrt(((double)o1x * o1x) + ((double)o1y * o1y));
        float h = (float)Math.Sqrt(((double)o2x * o2x) + ((double)o2y * o2y));
        float angle = (float)Math.Atan2(o1y, o1x);
        angle = (float)((angle * 180f) / Math.PI);

        // OpenCV 4.5.1 and later report angles in [-90, 0): a rectangle at 0 degrees is the same one at -90 with its
        // sides swapped, and boxPoints then lists the corners in a different order.
        if (angle >= 0)
        {
            angle -= 90;
            (w, h) = (h, w);
        }

        BoxPoints(cx, cy, w, h, angle, corners);
        return Math.Min(w, h);
    }

    /// <summary>OpenCV's <c>RotatedRect::points</c>, in single precision as OpenCV computes it.</summary>
    private static void BoxPoints(float cx, float cy, float w, float h, float angleDegrees, Span<PointD> corners)
    {
        double radians = angleDegrees * Math.PI / 180.0;
        float b = (float)Math.Cos(radians) * 0.5f;
        float a = (float)Math.Sin(radians) * 0.5f;
        float p0x = cx - (a * h) - (b * w);
        float p0y = cy + (b * h) - (a * w);
        float p1x = cx + (a * h) - (b * w);
        float p1y = cy - (b * h) - (a * w);
        corners[0] = new PointD(p0x, p0y);
        corners[1] = new PointD(p1x, p1y);
        corners[2] = new PointD((2 * cx) - p0x, (2 * cy) - p0y);
        corners[3] = new PointD((2 * cx) - p1x, (2 * cy) - p1y);
    }

    /// <summary>
    /// Orders a rectangle's corners as rapidocr's <c>get_mini_boxes</c> does after <c>cv2.boxPoints</c>: the two
    /// leftmost corners become top-left and bottom-left by their y, the other two top-right and bottom-right. The
    /// corners are rounded to single precision first, the precision OpenCV returns them in.
    /// </summary>
    public static void OrderMiniBox(ReadOnlySpan<PointD> corners, Span<PointD> box)
    {
        Span<PointD> p = stackalloc PointD[4];
        for (int i = 0; i < 4; i++)
        {
            p[i] = new PointD((float)corners[i].X, (float)corners[i].Y);
        }

        // A stable insertion sort by x, as Python's sorted() is stable.
        for (int i = 1; i < 4; i++)
        {
            PointD v = p[i];
            int j = i - 1;
            while (j >= 0 && p[j].X > v.X)
            {
                p[j + 1] = p[j];
                j--;
            }

            p[j + 1] = v;
        }

        (PointD tl, PointD bl) = p[1].Y > p[0].Y ? (p[0], p[1]) : (p[1], p[0]);
        (PointD tr, PointD br) = p[3].Y > p[2].Y ? (p[2], p[3]) : (p[3], p[2]);
        box[0] = tl;
        box[1] = tr;
        box[2] = br;
        box[3] = bl;
    }

    /// <summary>
    /// The unclip distance rapidocr computes with shapely: the polygon's area times the ratio over its perimeter.
    /// </summary>
    public static double UnclipDistance(ReadOnlySpan<PointD> polygon, double ratio)
    {
        double area = 0, length = 0;
        for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
        {
            area += (polygon[j].X * polygon[i].Y) - (polygon[i].X * polygon[j].Y);
            double dx = polygon[i].X - polygon[j].X, dy = polygon[i].Y - polygon[j].Y;
            length += Math.Sqrt((dx * dx) + (dy * dy));
        }

        return length == 0 ? 0 : Math.Abs(area) * 0.5 * ratio / length;
    }

    /// <summary>
    /// Offsets a closed polygon outward by <paramref name="delta"/> with round joins, as Clipper 6.4's
    /// <c>ClipperOffset</c> does for pyclipper (Boost Software License; this is a port of the parts rapidocr uses):
    /// the input is truncated to integers as pyclipper converts floats, arcs follow the default arc tolerance of 0.25,
    /// and output points are rounded half away from zero. Returns the number of points written, 0 when the polygon
    /// has fewer than three distinct points.
    /// </summary>
    public static int OffsetPolygonRound(ReadOnlySpan<PointD> polygon, double delta, List<PointD> output)
    {
        output.Clear();
        Span<long> xs = stackalloc long[8];
        Span<long> ys = stackalloc long[8];
        if (polygon.Length > 8)
        {
            throw new ArgumentException("Only small polygons are supported.", nameof(polygon));
        }

        // Strip duplicate points as ClipperOffset.AddPath does.
        int highI = polygon.Length - 1;
        long x0 = (long)polygon[0].X, y0 = (long)polygon[0].Y;
        while (highI > 0 && (long)polygon[highI].X == x0 && (long)polygon[highI].Y == y0)
        {
            highI--;
        }

        int len = 0;
        xs[len] = x0;
        ys[len++] = y0;
        for (int i = 1; i <= highI; i++)
        {
            long x = (long)polygon[i].X, y = (long)polygon[i].Y;
            if (x != xs[len - 1] || y != ys[len - 1])
            {
                xs[len] = x;
                ys[len++] = y;
            }
        }

        if (len < 3 || delta < 1e-20)
        {
            return 0;
        }

        // Clipper wants a positive orientation (its Area >= 0) for an outward offset.
        double a = 0;
        for (int i = 0, j = len - 1; i < len; j = i++)
        {
            a += ((double)xs[j] + xs[i]) * ((double)ys[j] - ys[i]);
        }

        if (-a * 0.5 < 0)
        {
            xs[..len].Reverse();
            ys[..len].Reverse();
        }

        const double ArcTolerance = 0.25;
        double tolerance = ArcTolerance > delta * ArcTolerance ? delta * ArcTolerance : ArcTolerance;
        double steps = Math.PI / Math.Acos(1 - (tolerance / delta));
        if (steps > delta * Math.PI)
        {
            steps = delta * Math.PI;
        }

        double sin = Math.Sin(2 * Math.PI / steps), cos = Math.Cos(2 * Math.PI / steps);
        double stepsPerRad = steps / (2 * Math.PI);

        Span<double> nx = stackalloc double[8];
        Span<double> ny = stackalloc double[8];
        for (int j = 0; j < len; j++)
        {
            int next = j + 1 == len ? 0 : j + 1;
            double dx = xs[next] - xs[j], dy = ys[next] - ys[j];
            double f = 1 / Math.Sqrt((dx * dx) + (dy * dy));
            nx[j] = dy * f;
            ny[j] = -dx * f;
        }

        // Clipper's OffsetPoint moves k to j at the end, except when it returns early for an almost straight
        // joint; the next joint then still measures against the older edge.
        int k = len - 1;
        for (int j = 0; j < len; j++)
        {
            double sinA = (nx[k] * ny[j]) - (nx[j] * ny[k]);
            double cosA = (nx[k] * nx[j]) + (ny[j] * ny[k]);
            if (Math.Abs(sinA * delta) < 1.0)
            {
                if (cosA > 0)
                {
                    output.Add(new PointD(Round(xs[j] + (nx[k] * delta)), Round(ys[j] + (ny[k] * delta))));
                    continue;
                }
            }
            else if (sinA > 1.0)
            {
                sinA = 1.0;
            }
            else if (sinA < -1.0)
            {
                sinA = -1.0;
            }

            if (sinA * delta < 0)
            {
                output.Add(new PointD(Round(xs[j] + (nx[k] * delta)), Round(ys[j] + (ny[k] * delta))));
                output.Add(new PointD(xs[j], ys[j]));
                output.Add(new PointD(Round(xs[j] + (nx[j] * delta)), Round(ys[j] + (ny[j] * delta))));
            }
            else
            {
                // DoRound: an arc from the previous edge's normal to this edge's normal.
                double angle = Math.Atan2(sinA, cosA);
                int arcSteps = Math.Max((int)Round(stepsPerRad * Math.Abs(angle)), 1);
                double X = nx[k], Y = ny[k];
                for (int i = 0; i < arcSteps; i++)
                {
                    output.Add(new PointD(Round(xs[j] + (X * delta)), Round(ys[j] + (Y * delta))));
                    double x2 = X;
                    X = (X * cos) - (sin * Y);
                    Y = (x2 * sin) + (Y * cos);
                }

                output.Add(new PointD(Round(xs[j] + (nx[j] * delta)), Round(ys[j] + (ny[j] * delta))));
            }

            k = j;
        }

        return output.Count;
    }

    // Clipper's Round: half away from zero, then truncated to an integer.
    private static long Round(double value) => value < 0 ? (long)(value - 0.5) : (long)(value + 0.5);

    /// <summary>
    /// The mean of <paramref name="map"/> inside a quadrilateral, as rapidocr's <c>box_score_fast</c> computes it:
    /// the quad's bounding box is clipped to the map, the corners are truncated to integers relative to it, and the
    /// pixels OpenCV's <c>fillPoly</c> would set (the scanline fill plus the 8-connected outline) are averaged.
    /// </summary>
    public static double MeanInsideQuad(ReadOnlySpan<float> map, int width, int height, ReadOnlySpan<PointD> box)
    {
        double minX = double.MaxValue, maxX = double.MinValue, minY = double.MaxValue, maxY = double.MinValue;
        for (int i = 0; i < 4; i++)
        {
            float x = (float)box[i].X, y = (float)box[i].Y;
            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
        }

        int xmin = Math.Clamp((int)Math.Floor(minX), 0, width - 1);
        int xmax = Math.Clamp((int)Math.Ceiling(maxX), 0, width - 1);
        int ymin = Math.Clamp((int)Math.Floor(minY), 0, height - 1);
        int ymax = Math.Clamp((int)Math.Ceiling(maxY), 0, height - 1);
        int w = xmax - xmin + 1, h = ymax - ymin + 1;

        Span<int> px = stackalloc int[4];
        Span<int> py = stackalloc int[4];
        for (int i = 0; i < 4; i++)
        {
            // astype(np.int32) truncates toward zero.
            px[i] = (int)((float)box[i].X - xmin);
            py[i] = (int)((float)box[i].Y - ymin);
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(w * h);
        try
        {
            Span<byte> mask = rented.AsSpan(0, w * h);
            mask.Clear();
            FillConvexPolygon(mask, w, h, px, py);

            double sum = 0;
            long count = 0;
            for (int y = 0; y < h; y++)
            {
                ReadOnlySpan<byte> m = mask.Slice(y * w, w);
                ReadOnlySpan<float> row = map.Slice(((ymin + y) * width) + xmin, w);
                for (int x = 0; x < w; x++)
                {
                    if (m[x] != 0)
                    {
                        sum += row[x];
                        count++;
                    }
                }
            }

            return count == 0 ? 0 : sum / count;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Sets the pixels OpenCV's <c>fillPoly</c> sets for a polygon with integer vertices and no sub-pixel shift: each
    /// edge drawn as an 8-connected line, and for every row from the top vertex to just above the bottom one, the span
    /// between the active edges, whose x positions advance in 16.16 fixed point.
    /// </summary>
    public static void FillConvexPolygon(Span<byte> mask, int w, int h, ReadOnlySpan<int> px, ReadOnlySpan<int> py)
    {
        int n = px.Length;
        Span<long> ex = stackalloc long[n];
        Span<long> edx = stackalloc long[n];
        Span<int> ey0 = stackalloc int[n];
        Span<int> ey1 = stackalloc int[n];
        int edges = 0;
        for (int i = 0; i < n; i++)
        {
            int j = i == 0 ? n - 1 : i - 1;
            DrawLine(mask, w, h, px[j], py[j], px[i], py[i]);
            if (py[j] == py[i])
            {
                continue;
            }

            long x0 = (long)px[j] << 16, x1 = (long)px[i] << 16;
            edx[edges] = (x1 - x0) / (py[i] - py[j]);
            if (py[j] < py[i])
            {
                ey0[edges] = py[j];
                ey1[edges] = py[i];
                ex[edges] = x0;
            }
            else
            {
                ey0[edges] = py[i];
                ey1[edges] = py[j];
                ex[edges] = x1;
            }

            edges++;
        }

        if (edges < 2)
        {
            return;
        }

        int top = int.MaxValue, bottom = int.MinValue;
        for (int e = 0; e < edges; e++)
        {
            top = Math.Min(top, ey0[e]);
            bottom = Math.Max(bottom, ey1[e]);
        }

        bottom = Math.Min(bottom, h);
        for (int y = Math.Max(top, 0); y < bottom; y++)
        {
            long left = long.MaxValue, right = long.MinValue;
            int active = 0;
            for (int e = 0; e < edges; e++)
            {
                if (ey0[e] <= y && y < ey1[e])
                {
                    long x = ex[e] + ((y - ey0[e]) * edx[e]);
                    left = Math.Min(left, x);
                    right = Math.Max(right, x);
                    active++;
                }
            }

            if (active < 2)
            {
                continue;
            }

            // The left end rounds up and the right end down, which is what OpenCV's spans come to.
            int x1 = (int)((left + 65535) >> 16), x2 = (int)(right >> 16);
            if (x1 < w && x2 >= 0)
            {
                x1 = Math.Max(x1, 0);
                x2 = Math.Min(x2, w - 1);
                if (x1 <= x2)
                {
                    mask.Slice((y * w) + x1, x2 - x1 + 1).Fill(1);
                }
            }
        }
    }

    /// <summary>
    /// Draws an 8-connected line as OpenCV's <c>LineIterator</c> does (left to right, Bresenham with its error term),
    /// clipping the end points to the image first with OpenCV's <c>clipLine</c>.
    /// </summary>
    public static void DrawLine(Span<byte> mask, int w, int h, int x1, int y1, int x2, int y2)
    {
        if ((uint)x1 >= (uint)w || (uint)x2 >= (uint)w || (uint)y1 >= (uint)h || (uint)y2 >= (uint)h)
        {
            if (!ClipLine(w, h, ref x1, ref y1, ref x2, ref y2))
            {
                return;
            }
        }

        if (x2 < x1)
        {
            (x1, x2) = (x2, x1);
            (y1, y2) = (y2, y1);
        }

        int dx = x2 - x1, dy = y2 - y1;
        int sy = dy < 0 ? -1 : 1;
        dy = Math.Abs(dy);
        int majorX = 1, majorY = 0, minorX = 0, minorY = sy;
        if (dy > dx)
        {
            (dx, dy) = (dy, dx);
            majorX = 0;
            majorY = sy;
            minorX = 1;
            minorY = 0;
        }

        int err = dx - (dy + dy);
        int x = x1, y = y1;
        for (int i = 0; i <= dx; i++)
        {
            if ((uint)x < (uint)w && (uint)y < (uint)h)
            {
                mask[(y * w) + x] = 1;
            }

            if (err < 0)
            {
                err += (dx + dx) - (dy + dy);
                x += majorX + minorX;
                y += majorY + minorY;
            }
            else
            {
                err -= dy + dy;
                x += majorX;
                y += majorY;
            }
        }
    }

    private static bool ClipLine(int w, int h, ref int x1, ref int y1, ref int x2, ref int y2)
    {
        long right = w - 1, bottom = h - 1;
        long lx1 = x1, ly1 = y1, lx2 = x2, ly2 = y2;
        int c1 = Code(lx1, ly1, right, bottom), c2 = Code(lx2, ly2, right, bottom);
        if ((c1 & c2) == 0 && (c1 | c2) != 0)
        {
            if ((c1 & 12) != 0)
            {
                long a = c1 < 8 ? 0 : bottom;
                lx1 += (long)((double)(a - ly1) * (lx2 - lx1) / (ly2 - ly1));
                ly1 = a;
                c1 = (lx1 < 0 ? 1 : 0) + (lx1 > right ? 2 : 0);
            }

            if ((c2 & 12) != 0)
            {
                long a = c2 < 8 ? 0 : bottom;
                lx2 += (long)((double)(a - ly2) * (lx2 - lx1) / (ly2 - ly1));
                ly2 = a;
                c2 = (lx2 < 0 ? 1 : 0) + (lx2 > right ? 2 : 0);
            }

            if ((c1 & c2) == 0 && (c1 | c2) != 0)
            {
                if (c1 != 0)
                {
                    long a = c1 == 1 ? 0 : right;
                    ly1 += (long)((double)(a - lx1) * (ly2 - ly1) / (lx2 - lx1));
                    lx1 = a;
                    c1 = 0;
                }

                if (c2 != 0)
                {
                    long a = c2 == 1 ? 0 : right;
                    ly2 += (long)((double)(a - lx2) * (ly2 - ly1) / (lx2 - lx1));
                    lx2 = a;
                    c2 = 0;
                }
            }
        }

        x1 = (int)lx1;
        y1 = (int)ly1;
        x2 = (int)lx2;
        y2 = (int)ly2;
        return (c1 | c2) == 0;
    }

    private static int Code(long x, long y, long right, long bottom) =>
        (x < 0 ? 1 : 0) + (x > right ? 2 : 0) + (y < 0 ? 4 : 0) + (y > bottom ? 8 : 0);
}
