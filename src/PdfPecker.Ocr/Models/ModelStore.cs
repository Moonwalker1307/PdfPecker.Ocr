using System.IO.Compression;
using PdfPecker.Ocr.Inference;

namespace PdfPecker.Ocr.Models;

/// <summary>
/// The bundled models, loaded from the assembly's resources on first use and then shared by every thread. The files
/// are written by tools/ModelBuilder from the PP-OCRv6 small detection and recognition models and the PP-OCR mobile
/// v2.0 text direction classifier; see THIRD-PARTY-NOTICES.md.
/// </summary>
internal static class ModelStore
{
    private static readonly Lazy<OcrModel> s_detection = new(() => Load("det"));
    private static readonly Lazy<OcrModel> s_recognition = new(() => Load("rec"));
    private static readonly Lazy<OcrModel> s_classifier = new(() => Load("cls"));

    /// <summary>The text detector: input [N, 3, H, W] with H and W multiples of 32, output the text probability [N, 1, H, W].</summary>
    public static OcrModel Detection => s_detection.Value;

    /// <summary>The line recogniser: input [N, 3, 48, W], output [N, W / 8, classes], a softmax over <see cref="Characters"/>.</summary>
    public static OcrModel Recognition => s_recognition.Value;

    /// <summary>The 0/180 degree line classifier: input [N, 3, 48, 192], output [N, 2] probabilities.</summary>
    public static OcrModel Classifier => s_classifier.Value;

    /// <summary>
    /// The recogniser's classes by index: index 0 is the CTC blank (an empty string), the last is the space that
    /// rapidocr appends, and each other entry is one character (which may be a surrogate pair).
    /// </summary>
    public static string[] Characters => Recognition.Characters;

    /// <summary>Loads a bundled model by its short name (det, rec or cls).</summary>
    internal static OcrModel Load(string name)
    {
        string resource = "PdfPecker.Ocr.Models." + name + ".bin.br";
        using var stream = typeof(ModelStore).Assembly.GetManifestResourceStream(resource)
            ?? throw new InvalidOperationException($"The model resource {resource} is missing from the assembly.");
        using var brotli = new BrotliStream(stream, CompressionMode.Decompress);
        using var data = new MemoryStream(capacity: (int)Math.Min(stream.Length * 2, int.MaxValue));
        brotli.CopyTo(data);
        return OcrModel.Load(data.GetBuffer().AsSpan(0, (int)data.Length));
    }
}
