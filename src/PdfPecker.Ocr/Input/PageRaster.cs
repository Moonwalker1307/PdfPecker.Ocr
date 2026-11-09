namespace PdfPecker.Ocr.Input;

/// <summary>A page rendered for recognition, with the resolution it was rendered at.</summary>
/// <param name="Pixels">The pixels; the receiver disposes them.</param>
/// <param name="Dpi">The resolution the page was rendered at, lowered from the one asked for when the page had more pixels than allowed.</param>
internal readonly record struct RenderedPage(PagePixels Pixels, float Dpi);

/// <summary>Renders pages of a PDF to <see cref="PagePixels"/> through PdfPecker's renderer, within a pixel budget.</summary>
internal static class PageRaster
{
    /// <summary>
    /// The most pixels PdfPecker can hand a page over in: its RGBA bytes, four per pixel, must fit in one array. Above
    /// it the renderer's int arithmetic overflows, so a page that would be larger fails with PdfLimitException first.
    /// </summary>
    public static long MaxRenderPixels => Array.MaxLength / 4;

    /// <summary>
    /// The resolution at which a page of <paramref name="widthPt"/> × <paramref name="heightPt"/> points has at most
    /// <paramref name="maxPixels"/> pixels: <paramref name="dpi"/> itself when it fits.
    /// </summary>
    public static float FitDpi(double widthPt, double heightPt, float dpi, long maxPixels)
    {
        for (int attempt = 0; attempt < 64; attempt++)
        {
            if (PixelCount(widthPt, heightPt, dpi) <= maxPixels) return dpi;
            double w = widthPt * dpi / 72, h = heightPt * dpi / 72;
            // Scaling by the square root of the ratio lands within a pixel's rounding of the limit; the loop takes the
            // last steps down when rounding up overshoots.
            double ratio = Math.Sqrt(maxPixels / (Math.Max(w, 1) * Math.Max(h, 1)));
            float next = (float)(dpi * Math.Min(ratio, 0.999));
            if (!(next > 0) || next >= dpi) next = dpi * 0.5f;
            dpi = next;
        }
        return dpi;
    }

    /// <summary>The pixels PdfPecker renders a page into at a resolution: each side rounded, and at least 1.</summary>
    public static long PixelCount(double widthPt, double heightPt, float dpi)
    {
        double w = Math.Max(1, Math.Round(widthPt * dpi / 72)), h = Math.Max(1, Math.Round(heightPt * dpi / 72));
        return w * h >= long.MaxValue / 2 ? long.MaxValue / 2 : (long)(w * h);
    }

    /// <summary>Renders a page at <paramref name="dpi"/>, or lower when it would have more than <paramref name="maxPixels"/> pixels.</summary>
    /// <exception cref="InvalidDataException">The page cannot be rendered.</exception>
    /// <exception cref="PdfLimitException">The page needs more than <see cref="MaxRenderPixels"/> pixels at the resolution chosen.</exception>
    public static RenderedPage Render(PdfRenderer renderer, int pageIndex, float dpi, long maxPixels, CancellationToken cancellationToken)
    {
        var (w, h) = renderer.GetPageSize(pageIndex);
        float fit = FitDpi(w, h, dpi, maxPixels);
        // MaxPixels has no upper bound of its own, and FitDpi gives up on pages it cannot bring within the budget.
        long pixels = PixelCount(w, h, fit);
        if (pixels > MaxRenderPixels)
            throw new PdfLimitException(FormattableString.Invariant($"A page of {pixels} pixels at {fit} dpi needs more than 2 GB of pixels."));
        var options = new ImageOptions
        {
            Dpi = fit,
            // The pixel budget is the limit here; PdfPecker's own cap on a side (20,000 by default) would shrink a long,
            // narrow page that fits the budget.
            MaxDimension = int.MaxValue,
        };
        var image = renderer.RenderPage(pageIndex, options, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return new RenderedPage(PagePixels.FromRgba(image.Rgba, image.Width, image.Height), fit);
    }
}
