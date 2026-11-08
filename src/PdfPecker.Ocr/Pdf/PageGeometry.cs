namespace PdfPecker.Ocr.Pdf;

/// <summary>
/// Maps positions in the pixels a page was read at to points on the page as displayed, with the origin at its
/// bottom-left corner as PDF content streams expect.
/// </summary>
/// <remarks>
/// PdfPecker renders a page as it is displayed: its crop box, turned by its <c>/Rotate</c>. The text layer is a stamp
/// page of exactly that size, and <see cref="PdfStamper"/> places a stamp as the page is displayed, mapping it through
/// the page's rotation and crop box origin to the page's own coordinates. So the only mapping needed here is from
/// pixels to displayed points, and the scale comes from the page and pixel sizes rather than the resolution alone:
/// the renderer rounds each side to whole pixels and stretches the page a fraction of a pixel to fill them.
/// </remarks>
internal readonly struct PageGeometry
{
    /// <summary>
    /// Makes the mapping for a page of <paramref name="widthPt"/> × <paramref name="heightPt"/> points read as
    /// <paramref name="page"/>: the pixels cover the page, or with <paramref name="byDpi"/> they are placed at the
    /// page's resolution from its top-left corner.
    /// </summary>
    public PageGeometry(double widthPt, double heightPt, OcrPage page, bool byDpi = false)
    {
        WidthPt = widthPt;
        HeightPt = heightPt;
        bool dpiKnown = float.IsFinite(page.Dpi) && page.Dpi > 0;
        double perPixel = dpiKnown ? 72.0 / page.Dpi : 72.0 / 300;
        ScaleX = page.Width > 0 && !(byDpi && dpiKnown) ? widthPt / page.Width : perPixel;
        ScaleY = page.Height > 0 && !(byDpi && dpiKnown) ? heightPt / page.Height : perPixel;
    }

    /// <summary>The page's displayed width in points.</summary>
    public double WidthPt { get; }

    /// <summary>The page's displayed height in points.</summary>
    public double HeightPt { get; }

    /// <summary>Points per pixel across the page.</summary>
    public double ScaleX { get; }

    /// <summary>Points per pixel down the page.</summary>
    public double ScaleY { get; }

    /// <summary>A position in pixels from the top-left corner as points from the bottom-left corner.</summary>
    public (double X, double Y) ToPoints(OcrPoint p) => (p.X * ScaleX, HeightPt - p.Y * ScaleY);
}
