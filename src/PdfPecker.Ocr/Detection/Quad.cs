using System.Numerics;

namespace PdfPecker.Ocr.Detection;

/// <summary>
/// Four corners in page pixels (origin top-left). For detected boxes the order is rapidocr's: top-left, top-right,
/// bottom-right, bottom-left as seen on the page. For recognised lines and words it follows the text direction, so
/// P0 to P1 runs along the baseline the way the text reads, whatever the rotation.
/// </summary>
internal readonly record struct Quad(Vector2 P0, Vector2 P1, Vector2 P2, Vector2 P3)
{
    public Vector2 this[int index] => index switch
    {
        0 => P0,
        1 => P1,
        2 => P2,
        3 => P3,
        _ => throw new ArgumentOutOfRangeException(nameof(index)),
    };

    /// <summary>The smallest axis-aligned rectangle holding the four corners, as left, top, right and bottom.</summary>
    public (float Left, float Top, float Right, float Bottom) Bounds()
    {
        float left = MathF.Min(MathF.Min(P0.X, P1.X), MathF.Min(P2.X, P3.X));
        float right = MathF.Max(MathF.Max(P0.X, P1.X), MathF.Max(P2.X, P3.X));
        float top = MathF.Min(MathF.Min(P0.Y, P1.Y), MathF.Min(P2.Y, P3.Y));
        float bottom = MathF.Max(MathF.Max(P0.Y, P1.Y), MathF.Max(P2.Y, P3.Y));
        return (left, top, right, bottom);
    }

    /// <summary>The mean length of the two edges along the text (P0-P1 and P3-P2).</summary>
    public float Width => (Vector2.Distance(P0, P1) + Vector2.Distance(P3, P2)) * 0.5f;

    /// <summary>The mean length of the two edges across the text (P0-P3 and P1-P2).</summary>
    public float Height => (Vector2.Distance(P0, P3) + Vector2.Distance(P1, P2)) * 0.5f;

    public Vector2 Center => (P0 + P1 + P2 + P3) * 0.25f;
}

/// <summary>A detected text region and the mean text probability inside it.</summary>
internal readonly record struct DetectedQuad(Quad Quad, float Score);
