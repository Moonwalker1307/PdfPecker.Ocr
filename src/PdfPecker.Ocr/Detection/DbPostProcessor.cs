using System.Numerics;

namespace PdfPecker.Ocr.Detection;

/// <summary>
/// Turns the detection network's probability map into text boxes: Differentiable Binarization post-processing as in
/// rapidocr's <c>DBPostProcess</c> (Apache-2.0, https://github.com/RapidAI/RapidOCR, from PaddleOCR): threshold,
/// 2 x 2 dilation, connected regions, a minimum-area rectangle per region, the mean probability inside it, an offset
/// that grows the shrunk kernel back by its area over its perimeter, and a final rectangle scaled to the page.
/// </summary>
/// <remarks>
/// <para>
/// rapidocr finds regions with OpenCV's <c>findContours</c>. Only the convex hull of a contour matters for the
/// rectangle, and that is the hull of the region's pixels, so this works on runs of pixels: one pass over the map
/// finds the runs of each row and joins them with 8-connectivity in a union-find, and each region's hull is built
/// from the two ends of its runs on each row. The boxes are listed in the order OpenCV returns their contours (reverse
/// raster order of their first pixel), which decides ties in the final sort.
/// </para>
/// <para>
/// rapidocr looks at the first <c>max_candidates</c> contours whatever their size, from the bottom of the page up, so a
/// dense page lost the lines at its top and the specks of a noisy scan used up the budget. Here only regions of at
/// least <c>MinSize</c> pixels each way count, the budget grows with the map (one region per 256 of its pixels, and at
/// least <see cref="DetectionOptions.MaxCandidates"/>), and regions are looked at from the top, so a page that still
/// has more loses its bottom.
/// </para>
/// <para>
/// <c>RETR_LIST</c> also returns the inner contours of holes in a region, whose rectangles lie inside the region's own
/// box and, when they pass the score threshold, duplicate text the outer box already holds. They are left out on
/// purpose; no hole appeared in the probability maps of the test pages.
/// </para>
/// <para>An instance reuses its buffers between pages and must be used by one thread at a time.</para>
/// </remarks>
internal sealed class DbPostProcessor
{
    private const int MinSize = 3;

    private int[] _runX0 = new int[1024];
    private int[] _runX1 = new int[1024];
    private int[] _runY = new int[1024];
    private int[] _parent = new int[1024];
    private int[] _rowStart = new int[64];
    private int[] _component = new int[1024];
    private int[] _componentStart = new int[64];
    private int[] _componentRuns = new int[1024];
    private byte[] _cur = [];
    private byte[] _prev = [];
    private PointD[] _points = new PointD[64];
    private PointD[] _hull = new PointD[128];
    private readonly List<PointD> _offset = [];
    private int _runCount;

    /// <summary>The number of connected regions found in the last map, for tests and diagnostics.</summary>
    public int LastRegionCount { get; private set; }

    /// <summary>Whether the last map had more regions of text size than were looked at, for tests and diagnostics.</summary>
    public bool LastCandidatesCapped { get; private set; }

    /// <summary>
    /// Finds the text boxes in a <paramref name="width"/> x <paramref name="height"/> probability map, scaled to a
    /// destination of <paramref name="destWidth"/> x <paramref name="destHeight"/> pixels, in rapidocr's output order
    /// (top to bottom in rows of 10 pixels, then left to right).
    /// </summary>
    public List<DetectedQuad> Process(ReadOnlySpan<float> map, int width, int height, int destWidth, int destHeight, DetectionOptions options, CancellationToken cancellationToken = default)
    {
        if (map.Length < (long)width * height)
        {
            throw new ArgumentException("The probability map is smaller than width * height.", nameof(map));
        }

        FindRuns(map, width, height, options.Threshold, options.UseDilation, cancellationToken);
        int regions = LabelRegions();
        LastRegionCount = regions;

        var boxes = new List<DetectedQuad>();
        Span<PointD> corners = stackalloc PointD[4];
        Span<PointD> box = stackalloc PointD[4];
        int cap = (int)Math.Max(options.MaxCandidates, Math.Min(int.MaxValue, (long)width * height / 256));
        int counted = 0;
        LastCandidatesCapped = false;
        for (int region = 0; region < regions; region++)
        {
            if ((region & 63) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            // Specks cost a few points of hull each and are dropped below; only regions that could be text count.
            if (!IsSpeck(region))
            {
                if (counted == cap)
                {
                    LastCandidatesCapped = true;
                    break;
                }

                counted++;
            }

            ReadOnlySpan<PointD> hull = RegionHull(region);
            float side = Geometry.MinAreaRect(hull, corners);
            if (side < MinSize)
            {
                continue;
            }

            Geometry.OrderMiniBox(corners, box);
            double score = Geometry.MeanInsideQuad(map, width, height, box);
            if (options.BoxThreshold > score)
            {
                continue;
            }

            double distance = Geometry.UnclipDistance(box, options.UnclipRatio);
            if (Geometry.OffsetPolygonRound(box, distance, _offset) == 0)
            {
                continue;
            }

            hull = OffsetHull();
            side = Geometry.MinAreaRect(hull, corners);
            if (side < MinSize + 2)
            {
                continue;
            }

            Geometry.OrderMiniBox(corners, box);
            if (ToDestination(box, width, height, destWidth, destHeight) is { } quad)
            {
                boxes.Add(new DetectedQuad(quad, (float)score));
            }
        }

        // OpenCV lists contours in reverse raster order of their first pixel; the sort below keeps that order for ties.
        boxes.Reverse();
        SortLikeRapidocr(boxes);
        return boxes;
    }

    /// <summary>
    /// Scales a box to the destination (rounded half to even, clipped to the destination), then applies rapidocr's
    /// <c>filter_det_res</c>: corners ordered clockwise from the top-left, clipped to the last pixel, and boxes with a
    /// side of 3 pixels or less dropped.
    /// </summary>
    private static Quad? ToDestination(ReadOnlySpan<PointD> box, int width, int height, int destWidth, int destHeight)
    {
        Span<Vector2> p = stackalloc Vector2[4];
        for (int i = 0; i < 4; i++)
        {
            // rapidocr computes this in float32: np.round(box / width * dest_width), then clips and casts to int32.
            float x = MathF.Round((float)box[i].X / width * destWidth, MidpointRounding.ToEven);
            float y = MathF.Round((float)box[i].Y / height * destHeight, MidpointRounding.ToEven);
            p[i] = new Vector2((int)Math.Clamp(x, 0, destWidth), (int)Math.Clamp(y, 0, destHeight));
        }

        OrderPointsClockwise(p);
        for (int i = 0; i < 4; i++)
        {
            p[i] = new Vector2(Math.Clamp(p[i].X, 0, destWidth - 1), Math.Clamp(p[i].Y, 0, destHeight - 1));
        }

        int rectWidth = (int)Vector2.Distance(p[0], p[1]);
        int rectHeight = (int)Vector2.Distance(p[0], p[3]);
        if (rectWidth <= 3 || rectHeight <= 3)
        {
            return null;
        }

        return new Quad(p[0], p[1], p[2], p[3]);
    }

    /// <summary>
    /// rapidocr's <c>order_points_clockwise</c>: the two leftmost points (by a stable sort on x) become top-left and
    /// bottom-left by their y, the other two top-right and bottom-right.
    /// </summary>
    internal static void OrderPointsClockwise(Span<Vector2> p)
    {
        for (int i = 1; i < 4; i++)
        {
            Vector2 v = p[i];
            int j = i - 1;
            while (j >= 0 && p[j].X > v.X)
            {
                p[j + 1] = p[j];
                j--;
            }

            p[j + 1] = v;
        }

        (Vector2 tl, Vector2 bl) = p[1].Y < p[0].Y ? (p[1], p[0]) : (p[0], p[1]);
        (Vector2 tr, Vector2 br) = p[3].Y < p[2].Y ? (p[3], p[2]) : (p[2], p[3]);
        p[0] = tl;
        p[1] = tr;
        p[2] = br;
        p[3] = bl;
    }

    /// <summary>
    /// rapidocr's <c>sorted_boxes</c>: a stable sort on the top-left y, rows started wherever the next box is 10 pixels
    /// or more lower than the previous one, and a stable sort by row, then top-left x. Layout puts lines into reading
    /// order later; this order keeps the detector's output comparable with rapidocr's.
    /// </summary>
    internal static void SortLikeRapidocr(List<DetectedQuad> boxes)
    {
        if (boxes.Count < 2)
        {
            return;
        }

        var byY = new DetectedQuad[boxes.Count];
        var index = new int[boxes.Count];
        for (int i = 0; i < index.Length; i++)
        {
            index[i] = i;
        }

        // A stable sort by y: ties keep their order through the index.
        Array.Sort(index, (a, b) =>
        {
            int c = boxes[a].Quad.P0.Y.CompareTo(boxes[b].Quad.P0.Y);
            return c != 0 ? c : a.CompareTo(b);
        });

        var row = new int[boxes.Count];
        for (int i = 0; i < index.Length; i++)
        {
            byY[i] = boxes[index[i]];
            row[i] = i == 0 ? 0 : row[i - 1] + (byY[i].Quad.P0.Y - byY[i - 1].Quad.P0.Y >= 10 ? 1 : 0);
        }

        for (int i = 0; i < index.Length; i++)
        {
            index[i] = i;
        }

        Array.Sort(index, (a, b) =>
        {
            int c = row[a].CompareTo(row[b]);
            if (c == 0)
            {
                c = byY[a].Quad.P0.X.CompareTo(byY[b].Quad.P0.X);
            }

            return c != 0 ? c : a.CompareTo(b);
        });

        for (int i = 0; i < index.Length; i++)
        {
            boxes[i] = byY[index[i]];
        }
    }

    private void FindRuns(ReadOnlySpan<float> map, int width, int height, float threshold, bool dilate, CancellationToken cancellationToken)
    {
        if (_cur.Length < width)
        {
            _cur = new byte[width];
            _prev = new byte[width];
        }

        EnsureCapacity(ref _rowStart, height + 1);
        Span<byte> cur = _cur.AsSpan(0, width);
        Span<byte> prev = _prev.AsSpan(0, width);
        prev.Clear();
        _runCount = 0;
        for (int y = 0; y < height; y++)
        {
            if ((y & 127) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            _rowStart[y] = _runCount;
            bool any = Binarize(map.Slice(y * width, width), threshold, cur);
            if (!dilate)
            {
                if (any)
                {
                    AddRuns(cur, y, extendRight: false);
                }

                continue;
            }

            // The 2 x 2 dilation with OpenCV's centre anchor sets a pixel when it or its left, upper or upper-left
            // neighbour is set: OR this row with the one above, then extend each run one pixel to the right.
            if (any)
            {
                for (int x = 0; x < width; x++)
                {
                    prev[x] |= cur[x];
                }

                AddRuns(prev, y, extendRight: true);
            }
            else
            {
                AddRuns(prev, y, extendRight: true);
            }

            cur.CopyTo(prev);
        }

        _rowStart[height] = _runCount;
    }

    private static bool Binarize(ReadOnlySpan<float> row, float threshold, Span<byte> dst)
    {
        bool any = false;
        int x = 0;
        if (Vector.IsHardwareAccelerated && row.Length >= Vector<float>.Count)
        {
            var t = new Vector<float>(threshold);
            for (; x <= row.Length - Vector<float>.Count; x += Vector<float>.Count)
            {
                var v = new Vector<float>(row.Slice(x));
                if (!Vector.GreaterThanAny(v, t))
                {
                    dst.Slice(x, Vector<float>.Count).Clear();
                    continue;
                }

                any = true;
                for (int k = 0; k < Vector<float>.Count; k++)
                {
                    dst[x + k] = row[x + k] > threshold ? (byte)1 : (byte)0;
                }
            }
        }

        for (; x < row.Length; x++)
        {
            bool on = row[x] > threshold;
            dst[x] = on ? (byte)1 : (byte)0;
            any |= on;
        }

        return any;
    }

    private void AddRuns(ReadOnlySpan<byte> bits, int y, bool extendRight)
    {
        int width = bits.Length;
        int x = bits.IndexOf((byte)1);
        if (x < 0)
        {
            return;
        }

        int lastIndex = -1;
        while (x >= 0 && x < width)
        {
            int end = bits[x..].IndexOf((byte)0);
            int x1 = end < 0 ? width - 1 : x + end - 1;
            if (extendRight)
            {
                x1 = Math.Min(x1 + 1, width - 1);
            }

            if (lastIndex >= 0 && x <= _runX1[lastIndex] + 1)
            {
                // The extended run touches the previous one: they are one run of the dilated row.
                _runX1[lastIndex] = x1;
            }
            else
            {
                lastIndex = AddRun(x, x1, y);
            }

            if (end < 0)
            {
                break;
            }

            int next = bits[(x + end)..].IndexOf((byte)1);
            x = next < 0 ? -1 : x + end + next;
        }
    }

    private int AddRun(int x0, int x1, int y)
    {
        if (_runCount == _runX0.Length)
        {
            int size = _runCount * 2;
            Array.Resize(ref _runX0, size);
            Array.Resize(ref _runX1, size);
            Array.Resize(ref _runY, size);
        }

        _runX0[_runCount] = x0;
        _runX1[_runCount] = x1;
        _runY[_runCount] = y;
        return _runCount++;
    }

    /// <summary>
    /// Joins runs that touch with 8-connectivity and groups them by region, in raster order of each region's first
    /// run. Returns the number of regions.
    /// </summary>
    private int LabelRegions()
    {
        int n = _runCount;
        EnsureCapacity(ref _parent, n);
        EnsureCapacity(ref _component, n);
        EnsureCapacity(ref _componentRuns, n);
        int[] parent = _parent;
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
        }

        // For each row, sweep its runs against the runs of the row above.
        int rows = _runCount == 0 ? 0 : _runY[n - 1] + 1;
        for (int y = 1; y < rows; y++)
        {
            int p = _rowStart[y - 1], pEnd = _rowStart[y];
            int cEnd = _rowStart[y + 1];
            for (int c = _rowStart[y]; c < cEnd; c++)
            {
                int a = _runX0[c] - 1, b = _runX1[c] + 1;
                while (p < pEnd && _runX1[p] < a)
                {
                    p++;
                }

                for (int k = p; k < pEnd && _runX0[k] <= b; k++)
                {
                    Union(parent, c, k);
                }
            }
        }

        // Number regions in the raster order of their first run; the root is always the region's first run.
        int regions = 0;
        for (int i = 0; i < n; i++)
        {
            int root = Find(parent, i);
            _component[i] = root == i ? regions++ : _component[root];
        }

        EnsureCapacity(ref _componentStart, regions + 1);
        Array.Clear(_componentStart, 0, regions + 1);
        for (int i = 0; i < n; i++)
        {
            _componentStart[_component[i] + 1]++;
        }

        for (int r = 0; r < regions; r++)
        {
            _componentStart[r + 1] += _componentStart[r];
        }

        // A counting sort keeps each region's runs in raster order. _parent is free again and serves as cursors.
        Span<int> cursor = _parent.AsSpan(0, regions);
        _componentStart.AsSpan(0, regions).CopyTo(cursor);
        for (int i = 0; i < n; i++)
        {
            _componentRuns[cursor[_component[i]]++] = i;
        }

        return regions;
    }

    private static int Find(int[] parent, int i)
    {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }

        return i;
    }

    private static void Union(int[] parent, int a, int b)
    {
        int ra = Find(parent, a), rb = Find(parent, b);
        if (ra == rb)
        {
            return;
        }

        // The smaller index becomes the root, so a region's root is its first run in raster order.
        if (ra < rb)
        {
            parent[rb] = ra;
        }
        else
        {
            parent[ra] = rb;
        }
    }

    /// <summary>Whether a region is less than <see cref="MinSize"/> pixels wide or high, judged by its runs.</summary>
    private bool IsSpeck(int region)
    {
        int start = _componentStart[region], end = _componentStart[region + 1];
        // The runs are in raster order, so the first and last give the rows.
        if (_runY[_componentRuns[end - 1]] - _runY[_componentRuns[start]] + 1 < MinSize)
        {
            return true;
        }

        int x0 = int.MaxValue, x1 = int.MinValue;
        for (int i = start; i < end; i++)
        {
            int run = _componentRuns[i];
            x0 = Math.Min(x0, _runX0[run]);
            x1 = Math.Max(x1, _runX1[run]);
            if (x1 - x0 + 1 >= MinSize)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>The convex hull of a region: only the ends of its runs on each row can be hull vertices.</summary>
    private ReadOnlySpan<PointD> RegionHull(int region)
    {
        int start = _componentStart[region], end = _componentStart[region + 1];
        EnsureCapacity(ref _points, 2 * (end - start));
        int count = 0;
        int i = start;
        while (i < end)
        {
            int run = _componentRuns[i];
            int y = _runY[run];
            int x0 = _runX0[run], x1 = _runX1[run];
            while (i + 1 < end && _runY[_componentRuns[i + 1]] == y)
            {
                i++;
                x1 = _runX1[_componentRuns[i]];
            }

            _points[count++] = new PointD(x0, y);
            if (x1 != x0)
            {
                _points[count++] = new PointD(x1, y);
            }

            i++;
        }

        EnsureCapacity(ref _hull, 2 * count);
        int h = Geometry.ConvexHullSorted(_points.AsSpan(0, count), _hull);
        Geometry.OrderHullLikeOpenCv(_hull.AsSpan(0, h), leftmostStart: true);
        return _hull.AsSpan(0, h);
    }

    /// <summary>The convex hull of the offset polygon, whose points are first sorted by (y, x).</summary>
    private ReadOnlySpan<PointD> OffsetHull()
    {
        int count = _offset.Count;
        EnsureCapacity(ref _points, count);
        _offset.CopyTo(_points);
        Span<PointD> points = _points.AsSpan(0, count);
        points.Sort(static (a, b) => a.Y != b.Y ? a.Y.CompareTo(b.Y) : a.X.CompareTo(b.X));
        EnsureCapacity(ref _hull, 2 * count);
        int h = Geometry.ConvexHullSorted(points, _hull);
        Geometry.OrderHullLikeOpenCv(_hull.AsSpan(0, h), leftmostStart: false);
        return _hull.AsSpan(0, h);
    }

    private static void EnsureCapacity<T>(ref T[] array, int size)
    {
        if (array.Length < size)
        {
            array = new T[Math.Max(size, array.Length * 2)];
        }
    }
}
