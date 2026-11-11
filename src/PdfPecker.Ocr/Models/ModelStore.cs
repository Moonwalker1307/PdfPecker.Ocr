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
    // The detector as loaded, before its stem is fused, which the gray variant starts from.
    private static readonly LoadOnce<OcrModel> s_detectionLayers = new(() => Load("det"));
    private static readonly LoadOnce<OcrModel> s_detection = new(() => s_detectionLayers.Value.WithFusedStem());
    private static readonly LoadOnce<OcrModel> s_recognition = new(() => LoadRecognition().WithFusedStem());
    private static readonly LoadOnce<OcrModel> s_convertedRecognition = new(() => Load("rec").WithFusedStem());
    private static readonly LoadOnce<OcrModel> s_classifier = new(() => Load("cls"));
    private static readonly LoadOnce<OcrModel> s_recognitionArgMax = new(() => Recognition.WithArgMaxOutput());
    private static readonly LoadOnce<OcrModel> s_detectionGray = new(() => s_detectionLayers.Value.WithGrayInput().WithFusedStem());

    /// <summary>The text detector: input [N, 3, H, W] with H and W multiples of 32, output the text probability [N, 1, H, W].</summary>
    public static OcrModel Detection => s_detection.Value;

    /// <summary>
    /// The text detector on a gray page: input [N, 1, H, W], the one plane that stands for the three equal colour planes
    /// (see <see cref="OcrModel.WithGrayInput"/>), output as <see cref="Detection"/>'s, bit for bit. It shares its layers.
    /// </summary>
    public static OcrModel DetectionGray => s_detectionGray.Value;

    /// <summary>
    /// The line recogniser: input [N, 3, 48, W], output [N, W / 8, classes], a softmax over <see cref="Characters"/>. Its
    /// last layer has the retrained columns of <see cref="HeadPatch"/>; <c>Load("rec")</c> is the converted model alone.
    /// </summary>
    public static OcrModel Recognition => s_recognition.Value;

    /// <summary>
    /// The line recogniser as converted from the ONNX model, without the retrained columns of <see cref="HeadPatch"/>:
    /// what ONNX Runtime and rapidocr compute, for the tests that compare with them.
    /// </summary>
    public static OcrModel ConvertedRecognition => s_convertedRecognition.Value;

    /// <summary>
    /// The line recogniser with the output greedy decoding reads, [N, W / 8, 2]: the most probable class at each time
    /// step and its probability (see <see cref="OcrModel.WithArgMaxOutput"/>). It shares the recogniser's layers.
    /// </summary>
    public static OcrModel RecognitionArgMax => s_recognitionArgMax.Value;

    /// <summary>The 0/180 degree line classifier: input [N, 3, 48, W] (rapidocr uses W = 192), output [N, 2] probabilities.</summary>
    public static OcrModel Classifier => s_classifier.Value;

    /// <summary>
    /// The recogniser's classes by index: index 0 is the CTC blank (an empty string), the last is the space that
    /// rapidocr appends, and each other entry is one character (which may be a surrogate pair).
    /// </summary>
    public static string[] Characters => Recognition.Characters;

    /// <summary>Loads a bundled model by its short name (det, rec or cls).</summary>
    internal static OcrModel Load(string name)
    {
        using var data = Inflate(name);
        return OcrModel.Load(data.GetBuffer().AsSpan(0, (int)data.Length));
    }

    /// <summary>The bundled recogniser with the retrained output weights of <see cref="HeadPatch"/> in its last layer.</summary>
    internal static OcrModel LoadRecognition()
    {
        ModelDefinition definition;
        using (var data = Inflate("rec"))
        {
            definition = ModelDefinition.Deserialize(data.GetBuffer().AsSpan(0, (int)data.Length));
        }

        RecognitionHeadPatch.ApplyTo(definition);
        return OcrModel.FromDefinition(definition);
    }

    /// <summary>The retrained output weights the recogniser loads with, read from the assembly's resources.</summary>
    internal static HeadPatch RecognitionHeadPatch
    {
        get
        {
            const string Resource = "PdfPecker.Ocr.Models.rec-head.bin";
            var assembly = typeof(ModelStore).Assembly;
            using var stream = assembly.GetManifestResourceStream(Resource)
                ?? throw new InvalidOperationException($"The model resource {Resource} is missing from {assembly.FullName}.");
            var data = new byte[stream.Length];
            stream.ReadExactly(data);
            return HeadPatch.Read(data);
        }
    }

    /// <summary>The model file of a bundled model, inflated; the model data is the stream's buffer up to its length.</summary>
    internal static MemoryStream Inflate(string name)
    {
        string resource = "PdfPecker.Ocr.Models." + name + ".bin.deflate";
        var assembly = typeof(ModelStore).Assembly;
        using var stream = assembly.GetManifestResourceStream(resource)
            ?? throw new InvalidOperationException(
                $"The model resource {resource} is missing from {assembly.FullName}, which has the resources " +
                $"[{string.Join(", ", assembly.GetManifestResourceNames())}].");
        using var deflate = new DeflateStream(stream, CompressionMode.Decompress);
        // The files inflate to less than twice their size, so the buffer is allocated once.
        var data = new MemoryStream(capacity: (int)Math.Min(stream.Length * 2, int.MaxValue));
        deflate.CopyTo(data);
        return data;
    }
}
