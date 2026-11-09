namespace PdfPecker.Ocr.Inference;

/// <summary>
/// Output-changing experiments, all off by default, so that the package's output stays what the goldens and the
/// ONNX Runtime references record. They are read once, from the environment variable <c>PDFPECKER_OCR_EXPERIMENT</c>
/// (a comma-separated list), before the models load; tests and benchmarks that see the internals may set them in code
/// before the first model is used.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><c>int8-rec</c>, <c>int8-det</c>, <c>int8-cls</c>: the 1x1 convolutions of the recogniser, detector or
/// orientation classifier in 8-bit integers (<see cref="Layers.Int8PointwiseConvLayer"/>), on processors with
/// AVX-VNNI; elsewhere the switch does nothing.</item>
/// <item><c>det-dpi</c> or <c>det-dpi:N</c>: pages below N dpi (300 by default) are detected at a proportionally
/// smaller longer side than <see cref="Detection.DetectionOptions.MaxSideLength"/>, so text keeps the size in the
/// detector's input it has on a page at N dpi.</item>
/// </list>
/// </remarks>
internal static class Experiments
{
    private static readonly string[] s_parts = Parse(Environment.GetEnvironmentVariable("PDFPECKER_OCR_EXPERIMENT"));

    /// <summary>The recogniser's 1x1 convolutions in 8 bits.</summary>
    public static bool Int8Recognition { get; set; } = Has("int8-rec");

    /// <summary>The detector's 1x1 convolutions in 8 bits.</summary>
    public static bool Int8Detection { get; set; } = Has("int8-det");

    /// <summary>The orientation classifier's 1x1 convolutions in 8 bits.</summary>
    public static bool Int8Orientation { get; set; } = Has("int8-cls");

    /// <summary>The resolution at and above which pages are detected at the full longer side; 0 turns the experiment off.</summary>
    public static float DetectionReferenceDpi { get; set; } = ReferenceDpi();

    private static string[] Parse(string? value) =>
        string.IsNullOrEmpty(value) ? [] : value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

    private static bool Has(string name) => Array.IndexOf(s_parts, name) >= 0;

    private static float ReferenceDpi()
    {
        foreach (string part in s_parts)
        {
            if (part == "det-dpi") return 300;
            if (part.StartsWith("det-dpi:", StringComparison.Ordinal)
                && float.TryParse(part.AsSpan(8), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float dpi)
                && dpi > 0)
                return dpi;
        }

        return 0;
    }

    /// <summary>A short description of what is on, for benchmark reports.</summary>
    public static string Describe()
    {
        var parts = new List<string>();
        if (Int8Recognition) parts.Add("int8-rec");
        if (Int8Detection) parts.Add("int8-det");
        if (Int8Orientation) parts.Add("int8-cls");
        if (DetectionReferenceDpi > 0) parts.Add(string.Create(System.Globalization.CultureInfo.InvariantCulture, $"det-dpi:{DetectionReferenceDpi}"));
        return parts.Count == 0 ? "none" : string.Join(",", parts);
    }
}
