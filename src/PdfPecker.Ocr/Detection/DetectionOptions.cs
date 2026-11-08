namespace PdfPecker.Ocr.Detection;

/// <summary>How the page is scaled to the detection network's input.</summary>
internal enum DetectionResize
{
    /// <summary>
    /// The default for documents: scale down so the longer side fits <see cref="DetectionOptions.MaxSideLength"/>,
    /// and scale small images up towards <see cref="DetectionOptions.MinSideLength"/> on the shorter side, but never
    /// past the longer-side limit.
    /// </summary>
    Fit,

    /// <summary>rapidocr's <c>limit_type: min</c>: scale up until the shorter side reaches the minimum, never down.</summary>
    Min,

    /// <summary>rapidocr's <c>limit_type: max</c>: scale down until the longer side fits the maximum, never up.</summary>
    Max,
}

/// <summary>
/// Settings of the text detector. The post-processing values are rapidocr's defaults for PP-OCR detection models; the
/// input size defaults differ, for the reason given on <see cref="MaxSideLength"/>.
/// </summary>
internal sealed record DetectionOptions
{
    /// <summary>
    /// The longer side of the network input in <see cref="DetectionResize.Fit"/> mode. rapidocr scales pages so the
    /// shorter side is at least 736 pixels after first shrinking anything over 2,000 pixels, which detects an A4 page at
    /// about 1,408 x 1,984. Measured with rapidocr on rendered A4 pages at 150, 200 and 300 dpi with 6 to 10 point text,
    /// a longer side of 1,536 found every line and read them as accurately as 2,048 (with orientation windows, see
    /// <c>TextRecognizer</c>), at 40% of the detection time; 1,024 started to lose 6 point text at 300 dpi. Lines are
    /// cropped from the full-resolution page, so a smaller detection input does not cost recognition accuracy.
    /// </summary>
    public int MaxSideLength { get; init; } = 1536;

    /// <summary>The shorter side small images are scaled up towards (rapidocr's <c>limit_side_len</c>).</summary>
    public int MinSideLength { get; init; } = 736;

    public DetectionResize Resize { get; init; } = DetectionResize.Fit;

    /// <summary>
    /// A hard bound on the network input in pixels, whatever the resize mode asks for, so an extreme aspect ratio cannot
    /// make the input (and the network's work) unbounded.
    /// </summary>
    public long MaxInputPixels { get; init; } = 2048L * 2048;

    /// <summary>Pixels with a text probability above this belong to text (rapidocr's <c>thresh</c>).</summary>
    public float Threshold { get; init; } = 0.3f;

    /// <summary>Boxes whose mean probability is below this are dropped (rapidocr's <c>box_thresh</c>).</summary>
    public float BoxThreshold { get; init; } = 0.5f;

    /// <summary>At most this many connected regions are looked at (rapidocr's <c>max_candidates</c>).</summary>
    public int MaxCandidates { get; init; } = 1000;

    /// <summary>How far a shrunk text kernel is grown back, relative to its area over its perimeter.</summary>
    public float UnclipRatio { get; init; } = 1.6f;

    /// <summary>Whether the binary map is dilated with a 2 x 2 kernel before finding regions.</summary>
    public bool UseDilation { get; init; } = true;

    /// <summary>
    /// Whether a short or very wide image (a single line) gets black bands above and below it before detection, as
    /// rapidocr's <c>apply_vertical_padding</c> does, so the network sees a line with room around it.
    /// </summary>
    public bool VerticalPadding { get; init; } = true;

    /// <summary>Images at most this high are padded (rapidocr's <c>min_height</c>).</summary>
    public int PaddingMinHeight { get; init; } = 30;

    /// <summary>Images wider than this times their height are padded (rapidocr's <c>width_height_ratio</c>).</summary>
    public float PaddingWidthHeightRatio { get; init; } = 8;

    /// <summary>The settings rapidocr 3.10 uses, for comparisons with it.</summary>
    public static DetectionOptions Rapidocr { get; } = new() { Resize = DetectionResize.Min, MinSideLength = 736 };
}
