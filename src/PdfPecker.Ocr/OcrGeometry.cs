namespace PdfPecker.Ocr;

/// <summary>A point on a page or image, in pixels from its top-left corner.</summary>
/// <param name="X">The distance from the left edge, in pixels.</param>
/// <param name="Y">The distance from the top edge, in pixels.</param>
public readonly record struct OcrPoint(float X, float Y);

/// <summary>
/// The four corners of a piece of text, in pixels from the top-left corner of the page. The text runs from
/// <see cref="TopLeft"/> to <see cref="TopRight"/>, so for a line turned upside down <see cref="TopLeft"/> is the corner
/// furthest down and right on the page. The corners are those of the text as read, not sorted by position.
/// </summary>
/// <param name="TopLeft">The corner where the text starts, above its baseline.</param>
/// <param name="TopRight">The corner where the text ends, above its baseline.</param>
/// <param name="BottomRight">The corner where the text ends, below its baseline.</param>
/// <param name="BottomLeft">The corner where the text starts, below its baseline.</param>
public readonly record struct OcrQuad(OcrPoint TopLeft, OcrPoint TopRight, OcrPoint BottomRight, OcrPoint BottomLeft);

/// <summary>An upright rectangle, in pixels from the top-left corner of the page.</summary>
/// <param name="X">The left edge, in pixels.</param>
/// <param name="Y">The top edge, in pixels.</param>
/// <param name="Width">The width, in pixels.</param>
/// <param name="Height">The height, in pixels.</param>
public readonly record struct OcrBox(float X, float Y, float Width, float Height);

/// <summary>Measurements of quads that the result types, the hOCR writer and the text layer share.</summary>
internal static class OcrGeometry
{
    /// <summary>
    /// The smallest upright rectangle around the quad. Corners that are not finite numbers are left out, so a quad
    /// from a broken recognizer still gets a usable box; with no finite corner the box is empty at the origin.
    /// </summary>
    public static OcrBox Bounds(in OcrQuad quad)
    {
        float minX = float.PositiveInfinity, minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity, maxY = float.NegativeInfinity;
        Include(quad.TopLeft, ref minX, ref minY, ref maxX, ref maxY);
        Include(quad.TopRight, ref minX, ref minY, ref maxX, ref maxY);
        Include(quad.BottomRight, ref minX, ref minY, ref maxX, ref maxY);
        Include(quad.BottomLeft, ref minX, ref minY, ref maxX, ref maxY);
        if (minX > maxX || minY > maxY) return default;
        return new OcrBox(minX, minY, maxX - minX, maxY - minY);
    }

    /// <summary>Whether all four corners are finite numbers.</summary>
    public static bool IsFinite(in OcrQuad quad)
        => IsFinite(quad.TopLeft) && IsFinite(quad.TopRight) && IsFinite(quad.BottomRight) && IsFinite(quad.BottomLeft);

    /// <summary>Whether both coordinates are finite numbers.</summary>
    public static bool IsFinite(OcrPoint p) => float.IsFinite(p.X) && float.IsFinite(p.Y);

    /// <summary>A quad for an upright rectangle, the text running left to right.</summary>
    public static OcrQuad FromBox(float x, float y, float width, float height)
        => new(new OcrPoint(x, y), new OcrPoint(x + width, y), new OcrPoint(x + width, y + height), new OcrPoint(x, y + height));

    /// <summary>
    /// The angle of the text's direction in degrees, counter-clockwise as seen on the page (0 for upright text, 90 for
    /// text that reads upwards), from the average of the top and bottom edges, so a slightly uneven quad still gets the
    /// angle of its middle.
    /// </summary>
    public static double AngleDegrees(in OcrQuad quad)
    {
        double dx = (quad.TopRight.X - quad.TopLeft.X) + (quad.BottomRight.X - quad.BottomLeft.X);
        double dy = (quad.TopRight.Y - quad.TopLeft.Y) + (quad.BottomRight.Y - quad.BottomLeft.Y);
        if (dx == 0 && dy == 0) return 0;
        // The page's y axis points down, so a rising line has a negative dy.
        return Math.Atan2(-dy, dx) * (180 / Math.PI);
    }

    private static void Include(OcrPoint p, ref float minX, ref float minY, ref float maxX, ref float maxY)
    {
        if (!IsFinite(p)) return;
        if (p.X < minX) minX = p.X;
        if (p.Y < minY) minY = p.Y;
        if (p.X > maxX) maxX = p.X;
        if (p.Y > maxY) maxY = p.Y;
    }
}
