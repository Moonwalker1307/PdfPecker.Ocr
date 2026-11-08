using System.Text;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// A placeholder that runs the whole pipeline (supervisor, workers, seeds, mutations, findings) against the one public
/// member the package has today, so the fuzzer and fuzz.yml can be built and checked before the real targets exist. It
/// never finds anything; remove it once ocr-image and the other targets are in.
/// </summary>
internal static class OcrSmokeTarget
{
    public static Target Target { get; } = new()
    {
        Name = "ocr-smoke",
        Extension = ".txt",
        Text = true,
        MaxSize = 64 * 1024,
        Dictionary = new[] { "0.1.0", "-dev", "\0", "�" },
        Seeds = _ => new[] { ("generated:version", Encoding.UTF8.GetBytes("PdfPecker.Ocr " + OcrInfo.Version)) },
        Run = (input, ct) =>
        {
            ct.ThrowIfCancellationRequested();
            // Comparing the input with the version keeps the call from being optimised away and reads every byte.
            _ = Encoding.UTF8.GetString(input).Contains(OcrInfo.Version, StringComparison.Ordinal);
        },
    };
}
