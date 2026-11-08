using System.Numerics;
using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>Polygon helpers for comparing boxes in tests.</summary>
internal static class QuadMath
{
    public static Quad FromArray(float[] b) =>
        new(new Vector2(b[0], b[1]), new Vector2(b[2], b[3]), new Vector2(b[4], b[5]), new Vector2(b[6], b[7]));

    /// <summary>Intersection over union of two convex quads (Sutherland-Hodgman clipping).</summary>
    public static double IoU(Quad a, Quad b)
    {
        var pa = new List<Vector2> { a.P0, a.P1, a.P2, a.P3 };
        var pb = new List<Vector2> { b.P0, b.P1, b.P2, b.P3 };
        double areaA = Math.Abs(Area(pa)), areaB = Math.Abs(Area(pb));
        if (areaA == 0 || areaB == 0)
        {
            return a == b ? 1 : 0;
        }

        // Clip requires the clipping polygon to run counter-clockwise in its own sign convention.
        if (Area(pb) < 0)
        {
            pb.Reverse();
        }

        List<Vector2> clipped = pa;
        for (int i = 0; i < pb.Count && clipped.Count > 0; i++)
        {
            Vector2 c0 = pb[i], c1 = pb[(i + 1) % pb.Count];
            var input = clipped;
            clipped = [];
            for (int j = 0; j < input.Count; j++)
            {
                Vector2 p = input[j], q = input[(j + 1) % input.Count];
                bool pin = Side(c0, c1, p) >= 0, qin = Side(c0, c1, q) >= 0;
                if (pin)
                {
                    clipped.Add(p);
                }

                if (pin != qin)
                {
                    clipped.Add(Intersect(c0, c1, p, q));
                }
            }
        }

        double inter = clipped.Count < 3 ? 0 : Math.Abs(Area(clipped));
        return inter / (areaA + areaB - inter);
    }

    private static double Area(List<Vector2> p)
    {
        double a = 0;
        for (int i = 0, j = p.Count - 1; i < p.Count; j = i++)
        {
            a += ((double)p[j].X * p[i].Y) - ((double)p[i].X * p[j].Y);
        }

        return a / 2;
    }

    private static double Side(Vector2 a, Vector2 b, Vector2 p) => ((double)(b.X - a.X) * (p.Y - a.Y)) - ((double)(b.Y - a.Y) * (p.X - a.X));

    private static Vector2 Intersect(Vector2 a, Vector2 b, Vector2 p, Vector2 q)
    {
        double a1 = Side(a, b, p), a2 = Side(a, b, q);
        double t = a1 / (a1 - a2);
        return new Vector2((float)(p.X + ((q.X - p.X) * t)), (float)(p.Y + ((q.Y - p.Y) * t)));
    }
}
