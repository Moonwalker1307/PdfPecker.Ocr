using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Recognition;
using PdfPecker.Ocr.Tests.Pipeline;
using Xunit.Abstractions;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>
/// The pages rapidocr 3.10 read with ONNX Runtime (tools/PipelineReference), read again with this package's own
/// runtime and the real models, and the text compared with rapidocr's.
/// </summary>
[Collection("EndToEnd")]
public sealed class RapidocrAgreementTests(ITestOutputHelper output)
{
    public static TheoryData<string> Cases => [.. PipelineFixture.Cases];

    /// <summary>
    /// With rapidocr's settings the pipeline reads every line rapidocr read, with the same text, whatever their
    /// order (the reading order is this package's own).
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void RapidocrSettingsReadRapidocrsText(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        PageResult page = Run(fx, DetectionOptions.Rapidocr, RecognitionOptions.Rapidocr);
        string[] expected = [.. Final(fx).Order(StringComparer.Ordinal)];
        string[] actual = [.. page.Lines.Select(l => l.Text).Order(StringComparer.Ordinal)];
        output.WriteLine(string.Join("\n", page.Lines.Select(l => l.Text)));
        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// With the package's defaults the text differs from rapidocr's only where the defaults set out to: a space at a
    /// word gap that rapidocr runs together (its word boxes split there too), lines that rapidocr's orientation step
    /// turned upside down by mistake, a line that the larger detection input splits at a wide gap, and the reading
    /// order. So the words, as rapidocr's word boxes have them, are the same, leaving out the wrongly turned lines.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void DefaultSettingsDifferOnlyByDesign(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        PageResult page = Run(fx, OcrEngine.DetectionOptions, new RecognitionOptions());
        output.WriteLine(string.Join("\n", page.Lines.Select(l => l.Text)));
        string[] expected = [.. fx.Root.GetProperty("final").EnumerateArray()
            .Where(f => !WronglyTurned.Contains(f.GetProperty("text").GetString()!))
            .SelectMany(f => f.GetProperty("words").EnumerateArray().Select(w => w.GetProperty("text").GetString()!))
            .Order(StringComparer.Ordinal)];
        string[] actual = [.. page.Lines.Where(l => !CorrectedTurns.Contains(l.Text)).SelectMany(l => l.Words.Select(w => w.Text)).Order(StringComparer.Ordinal)];
        Assert.Equal(expected, actual);
    }

    /// <summary>What rapidocr read on lines that it turned upside down although they were upright.</summary>
    private static readonly HashSet<string> WronglyTurned = new(StringComparer.Ordinal) { "%6ε" };

    /// <summary>The same lines read upright.</summary>
    private static readonly HashSet<string> CorrectedTurns = new(StringComparer.Ordinal) { "39%" };

    private static IEnumerable<string> Final(PipelineFixture fx)
        => fx.Root.GetProperty("final").EnumerateArray().Select(f => f.GetProperty("text").GetString()!);

    private static PageResult Run(PipelineFixture fx, DetectionOptions detection, RecognitionOptions recognition)
    {
        using var context = new InferenceContext(4);
        var pipeline = new OcrPipeline(
            new ModelNetwork(ModelStore.Detection, context),
            new ModelNetwork(ModelStore.Recognition, context),
            new ModelNetwork(ModelStore.Classifier, context),
            OcrEngine.Characters,
            detection,
            recognition);
        return pipeline.Run(new PixelView(fx.Image(), fx.Width, fx.Height, fx.Channels), 150);
    }
}
