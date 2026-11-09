using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Models;
using Xunit.Abstractions;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// The bundled models against ONNX Runtime on the original ONNX files (fixtures from tools/OnnxReference/generate.py):
/// every stored output value within 1e-3, and the recogniser's and classifier's argmax equal for every row.
/// </summary>
public sealed class ReferenceTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-3f;

    public static TheoryData<string> Fixtures => new() { "det_96x128", "det_320x320", "rec_1x48x320", "rec_2x48x160", "cls_2x48x192" };

    [Theory]
    [MemberData(nameof(Fixtures))]
    public void MatchesOnnxRuntime(string name)
    {
        var fixture = ReferenceFixture.Load(name);
        var model = fixture.Model switch
        {
            "det" => ModelStore.Detection,
            "rec" => ModelStore.Recognition,
            _ => ModelStore.Classifier,
        };
        using var context = new InferenceContext(4);
        var result = model.Run(fixture.Input(), context);
        Assert.Equal(fixture.OutputShape, result.Shape);

        var values = result.Span;
        float worst = 0;
        int worstIndex = -1;
        foreach (var (index, expected) in fixture.Samples)
        {
            float error = Math.Abs(values[index] - expected);
            if (error > worst)
            {
                worst = error;
                worstIndex = index;
            }
        }
        double mean = 0;
        foreach (float v in values) mean += v;
        mean /= values.Length;
        output.WriteLine($"{name}: {fixture.Samples.Length} values compared, max abs error {worst:E3} at {worstIndex}, mean {mean:F6} (reference {fixture.Mean:F6})");
        Assert.True(worst <= Tolerance, $"max abs error {worst} at index {worstIndex} exceeds {Tolerance}");
        Assert.True(Math.Abs(mean - fixture.Mean) <= 1e-4, $"mean {mean} differs from {fixture.Mean}");

        int classes = result.Shape[^1];
        int mismatches = 0;
        for (int row = 0; row < fixture.RowMaxima.Length; row++)
        {
            var span = values.Slice(row * classes, classes);
            int best = 0;
            for (int i = 1; i < classes; i++)
                if (span[i] > span[best]) best = i;
            if (best != fixture.RowMaxima[row].Index) mismatches++;
            Assert.True(Math.Abs(span[best] - fixture.RowMaxima[row].Value) <= Tolerance);
        }
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void CharactersFollowRapidOcr()
    {
        var characters = ModelStore.Characters;
        Assert.Equal(18710, characters.Length);
        Assert.Equal("", characters[0]);
        Assert.Equal(" ", characters[^1]);
        Assert.Equal("!", characters[1]);
        Assert.All(characters.AsSpan(1, characters.Length - 2).ToArray(), c => Assert.True(c.Length is 1 or 2));
        Assert.Contains("ø", characters);
        Assert.Contains("ß", characters);
    }
}
