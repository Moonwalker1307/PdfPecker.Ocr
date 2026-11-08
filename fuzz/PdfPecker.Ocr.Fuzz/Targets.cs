using System.Text;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// The registry of fuzz targets. A target lives in a file of its own under Targets/ as a static class with a
/// <c>Target</c> property, and is added to <see cref="All"/> with one line, so that people adding targets at the same
/// time touch different files and one line each here. The order of <see cref="All"/> is the order of --target all.
/// </summary>
internal static class Targets
{
    /// <summary>Every target that --list prints and --target all runs.</summary>
    public static IReadOnlyList<Target> All { get; } = new[]
    {
        OcrImageTarget.Target,
        OcrPixelsTarget.Target,
        OcrPdfTarget.Target,
        SearchableLayerTarget.Target,
    };

    // Targets that fail on purpose, to check that the supervisor copes with a fatal crash and with a thread that cannot be
    // cancelled. They are not listed and not part of "all".
    private static readonly Target[] SelfTests =
    {
        new()
        {
            Name = "selftest-crash", Extension = ".bin", Text = true, MaxSize = 4096, Dictionary = new[] { "crash" },
            Seeds = _ => new[] { ("seed", Encoding.UTF8.GetBytes("some text with a trigger word")) },
            Run = (d, ct) => { if (Encoding.UTF8.GetString(d).Contains("crash", StringComparison.Ordinal)) Environment.FailFast("selftest crash"); },
        },
        new()
        {
            Name = "selftest-hang", Extension = ".bin", Text = true, MaxSize = 4096, Dictionary = new[] { "hang" },
            Seeds = _ => new[] { ("seed", Encoding.UTF8.GetBytes("some text with a trigger word")) },
            Run = (d, ct) => { if (Encoding.UTF8.GetString(d).Contains("hang", StringComparison.Ordinal)) while (true) Thread.SpinWait(1000); },
        },
    };

    public static Target? Find(string name)
        => All.Concat(SelfTests).FirstOrDefault(t => string.Equals(t.Name, name, StringComparison.OrdinalIgnoreCase));
}
