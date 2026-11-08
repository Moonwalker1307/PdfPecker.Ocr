using System.Numerics;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Layout;

/// <summary>
/// Puts recognised lines into reading order: top to bottom and left to right, with columns of text read one after
/// the other, on pages turned by 90, 180 or 270 degrees as well.
/// </summary>
/// <remarks>
/// <para>
/// The page is first turned so its dominant text direction (the length-weighted direction of the lines, to the
/// nearest quarter turn) reads left to right; an upside-down scan is then read from its real top. In that frame, the
/// lines are split recursively at the white space between them (an XY cut): bands of rows where a horizontal gap
/// crosses the whole block, columns where a vertical gap runs through it, the widest gaps first.
/// </para>
/// <para>
/// When a block can be cut both ways, the choice decides between columns of prose and the rows of a table or form,
/// whose cells also line up in columns. Columns win only when every column holds at least two lines and its lines are
/// long, at least five times their height on median, as lines of running text are and table cells are not;
/// otherwise rows win, so tables and forms are read row by row.
/// </para>
/// <para>
/// rapidocr orders boxes by their top-left corner in bands of 10 pixels, which interleaves columns and depends on the
/// resolution.
/// </para>
/// </remarks>
internal static class ReadingOrder
{
    private const int MaxDepth = 64;

    /// <summary>A line in the turned frame: its box, height and position in the input.</summary>
    private readonly record struct Item(int Index, float X0, float Y0, float X1, float Y1, float Height)
    {
        public float Width => X1 - X0;
    }

    /// <summary>Returns the lines in reading order, grouped into blocks of lines that belong together.</summary>
    public static List<List<RecognizedLine>> Order(IReadOnlyList<RecognizedLine> lines)
    {
        var blocks = new List<List<RecognizedLine>>();
        if (lines.Count == 0)
        {
            return blocks;
        }

        int turn = DominantQuarterTurn(lines);
        var items = new List<Item>(lines.Count);
        for (int i = 0; i < lines.Count; i++)
        {
            Quad q = lines[i].Quad;
            Vector2 a = Turn(q.P0, turn), b = Turn(q.P1, turn), c = Turn(q.P2, turn), d = Turn(q.P3, turn);
            float x0 = MathF.Min(MathF.Min(a.X, b.X), MathF.Min(c.X, d.X)), x1 = MathF.Max(MathF.Max(a.X, b.X), MathF.Max(c.X, d.X));
            float y0 = MathF.Min(MathF.Min(a.Y, b.Y), MathF.Min(c.Y, d.Y)), y1 = MathF.Max(MathF.Max(a.Y, b.Y), MathF.Max(c.Y, d.Y));
            float h = MathF.Max(1, MathF.Min(q.Height, y1 - y0));
            if (!float.IsFinite(x0 + x1 + y0 + y1 + h))
            {
                (x0, y0, x1, y1, h) = (0, 0, 0, 0, 1);
            }

            items.Add(new Item(i, x0, y0, x1, y1, h));
        }

        var orderedBlocks = new List<List<Item>>();
        Cut(items, orderedBlocks, 0);
        foreach (List<Item> block in orderedBlocks)
        {
            var list = new List<RecognizedLine>(block.Count);
            foreach (Item item in block)
            {
                list.Add(lines[item.Index]);
            }

            blocks.Add(list);
        }

        return blocks;
    }

    /// <summary>
    /// The quarter turn (0 to 3, counter-clockwise on the page) that makes the dominant text direction read left to
    /// right, from the lines' P0 to P1 directions weighted by their length.
    /// </summary>
    internal static int DominantQuarterTurn(IReadOnlyList<RecognizedLine> lines)
    {
        Span<double> weight = stackalloc double[4];
        foreach (RecognizedLine line in lines)
        {
            Vector2 d = line.Quad.P1 - line.Quad.P0;
            double length = d.Length();
            if (!(length > 0) || !double.IsFinite(length))
            {
                continue;
            }

            // The direction's angle on the page (y down), to the nearest quarter: 0 reads right, 1 up, 2 left, 3 down.
            double angle = Math.Atan2(-d.Y, d.X);
            int quarter = ((int)Math.Round(angle / (Math.PI / 2)) % 4 + 4) % 4;
            weight[quarter] += length;
        }

        int best = 0;
        for (int i = 1; i < 4; i++)
        {
            if (weight[i] > weight[best])
            {
                best = i;
            }
        }

        return best;
    }

    // Turns a page point so that text in the given quarter direction reads left to right (y stays down).
    private static Vector2 Turn(Vector2 p, int quarter) => quarter switch
    {
        1 => new Vector2(-p.Y, p.X),
        2 => new Vector2(-p.X, -p.Y),
        3 => new Vector2(p.Y, -p.X),
        _ => p,
    };

    private static void Cut(List<Item> items, List<List<Item>> output, int depth)
    {
        if (items.Count <= 1 || depth >= MaxDepth)
        {
            output.Add(SortLeaf(items));
            return;
        }

        List<List<Item>> rows = Split(items, horizontal: true);
        List<List<Item>> columns = Split(items, horizontal: false);
        List<List<Item>> parts;
        if (rows.Count > 1 && columns.Count > 1)
        {
            parts = IsProseColumns(columns) ? columns : rows;
        }
        else if (rows.Count > 1)
        {
            parts = rows;
        }
        else if (columns.Count > 1)
        {
            parts = columns;
        }
        else
        {
            output.Add(SortLeaf(items));
            return;
        }

        foreach (List<Item> part in parts)
        {
            Cut(part, output, depth + 1);
        }
    }

    /// <summary>
    /// Splits items at the widest gaps of their projection onto one axis: rows (by y) or columns (by x). Only gaps of
    /// at least 80% of the widest one cut, as in the classic recursive XY cut, so a heading or a footer is cut off
    /// before the evenly spaced rows of two columns would be, and the columns get their own turn. A line's vertical
    /// extent is taken without the top and bottom quarter of its height, because detected boxes are grown around the
    /// text and boxes of tightly set lines overlap.
    /// </summary>
    private static List<List<Item>> Split(List<Item> items, bool horizontal)
    {
        var sorted = new List<Item>(items);
        if (horizontal)
        {
            sorted.Sort(static (a, b) => Start(a, true).CompareTo(Start(b, true)) is var c && c != 0 ? c : a.Index.CompareTo(b.Index));
        }
        else
        {
            sorted.Sort(static (a, b) => Start(a, false).CompareTo(Start(b, false)) is var c && c != 0 ? c : a.Index.CompareTo(b.Index));
        }

        // The gap before each item (0 where it overlaps what came before), and the widest one.
        var gaps = new float[sorted.Count];
        float end = float.NegativeInfinity, widest = 0;
        for (int i = 0; i < sorted.Count; i++)
        {
            float s = Start(sorted[i], horizontal);
            gaps[i] = i > 0 && s > end ? s - end : 0;
            widest = MathF.Max(widest, gaps[i]);
            end = MathF.Max(end, End(sorted[i], horizontal));
        }

        var parts = new List<List<Item>>();
        var current = new List<Item>();
        for (int i = 0; i < sorted.Count; i++)
        {
            if (current.Count > 0 && gaps[i] > 0 && gaps[i] >= 0.8f * widest)
            {
                parts.Add(current);
                current = [];
            }

            current.Add(sorted[i]);
        }

        parts.Add(current);
        return parts;
    }

    private static float Start(Item item, bool horizontal) => horizontal ? item.Y0 + (0.25f * item.Height) : item.X0;

    private static float End(Item item, bool horizontal) => horizontal ? MathF.Max(item.Y0 + (0.25f * item.Height), item.Y1 - (0.25f * item.Height)) : item.X1;

    private static bool IsProseColumns(List<List<Item>> columns)
    {
        foreach (List<Item> column in columns)
        {
            if (column.Count < 2)
            {
                return false;
            }

            var ratios = new float[column.Count];
            for (int i = 0; i < ratios.Length; i++)
            {
                ratios[i] = column[i].Width / column[i].Height;
            }

            Array.Sort(ratios);
            if (ratios[ratios.Length / 2] < 5)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Orders lines that no gap separates: top to bottom, where a line that overlaps the previous one by more than half
    /// the smaller height joins its row, and each row left to right.
    /// </summary>
    private static List<Item> SortLeaf(List<Item> items)
    {
        var sorted = new List<Item>(items);
        sorted.Sort(static (a, b) => (a.Y0 + a.Y1).CompareTo(b.Y0 + b.Y1) is var c && c != 0 ? c : a.Index.CompareTo(b.Index));
        var result = new List<Item>(sorted.Count);
        int i = 0;
        while (i < sorted.Count)
        {
            var row = new List<Item> { sorted[i] };
            float top = sorted[i].Y0, bottom = sorted[i].Y1;
            int j = i + 1;
            while (j < sorted.Count)
            {
                Item next = sorted[j];
                float overlap = MathF.Min(bottom, next.Y1) - MathF.Max(top, next.Y0);
                if (overlap <= 0.5f * MathF.Min(bottom - top, next.Y1 - next.Y0))
                {
                    break;
                }

                row.Add(next);
                j++;
            }

            row.Sort(static (a, b) => a.X0.CompareTo(b.X0) is var c && c != 0 ? c : a.Index.CompareTo(b.Index));
            result.AddRange(row);
            i = j;
        }

        return result;
    }
}
