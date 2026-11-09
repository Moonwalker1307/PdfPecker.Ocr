using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// The layers that skip the equal columns of a batch's zero padding (see <see cref="ColumnRuns"/>) must give every bit of
/// the full computation, on every matrix kernel and at any number of threads, and the runs must really be found, or the
/// comparison proves nothing.
/// </summary>
public sealed class ColumnRunsTests
{
    /// <summary>
    /// A recognition batch of lines of different lengths, each padded with zeros to the widest as the recogniser pads
    /// them: the reference fixture's two lines, one cut short, and one that fills the whole width.
    /// </summary>
    private static Tensor PaddedBatch(int width)
    {
        var lines = ReferenceFixture.Load("rec_2x48x160").Input();
        int w = lines.Shape[3];
        var batch = new Tensor(4, 3, 48, width);
        int[] lengths = [w, 37, 101, width];
        for (int n = 0; n < 4; n++)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < 48; y++)
                {
                    for (int x = 0; x < lengths[n]; x++)
                        batch.Span[(((n * 3) + c) * 48 + y) * width + x] = lines[n % 2, c, y, x % w];
                }
            }
        }

        return batch;
    }

    private static float[] Run(OcrModel model, Tensor input, int threads, bool skip)
    {
        // The switch belongs to the context, so tests of other classes that run at the same time keep the skip on.
        using var context = new InferenceContext(threads) { SkipsEqualColumns = skip };
        return model.Run(input, context).Span.ToArray();
    }

    [Theory]
    [InlineData(400, 1)]
    [InlineData(400, 3)]
    [InlineData(321, 4)]
    [InlineData(1000, 2)]
    public void SkippingThePaddingGivesTheSameBits(int width, int threads)
    {
        var input = PaddedBatch(width);
        foreach (var model in new[] { ModelStore.RecognitionArgMax, ModelStore.Recognition })
        {
            float[] full = Run(model, input, threads, skip: false);
            float[] skipped = Run(model, input, threads, skip: true);
            Assert.True(full.AsSpan().SequenceEqual(skipped), $"{model.Name} differs at width {width} with {threads} threads");
        }
    }

    [Fact]
    public void EveryMatrixKernelGivesTheSameBitsWhenSkipping()
    {
        byte[] data = ModelStore.Inflate("rec").ToArray();
        var input = PaddedBatch(480);
        float[] full = Run(ModelStore.Recognition, input, 2, skip: false);
        foreach (var kernel in GemmKernel.Supported)
        {
            float[] skipped = Run(OcrModel.Load(data, kernel), input, 2, skip: true);
            Assert.True(full.AsSpan().SequenceEqual(skipped), $"{kernel.Name} differs");
        }
    }

    [Fact]
    public void RunsAreFoundThroughTheWholeRecognitionNetwork()
    {
        var input = PaddedBatch(640);
        using var context = new InferenceContext(1);
        var model = ModelStore.RecognitionArgMax;
        model.Run(input, context);
        var steps = context.GetPlan(model, input.Shape).Steps;

        // The short line's padding, 603 of 640 pixels, is one run up to the decoder's input: the last step's output has
        // 80 time steps, of which those beyond the line's reach are equal.
        var last = steps[^1].OutputRuns;
        Assert.Equal(1, last.Axis);
        Assert.True(last.End[1] - last.Start[1] > 40, $"run [{last.Start[1]}, {last.End[1]})");
        Assert.False(last.HasRun(3));
        Assert.Contains(steps, s => s.Layer is PdfPecker.Ocr.Inference.Layers.SqueezeExciteLayer && s.OutputRuns.HasRun(1));

        // A context with the skip off finds none, so that the comparisons above compare two different computations.
        using var full = new InferenceContext(1) { SkipsEqualColumns = false };
        model.Run(input, full);
        Assert.All(full.GetPlan(model, input.Shape).Steps, s => Assert.Equal(-1, s.OutputRuns.Axis));
    }

    [Fact]
    public void ADetectionInputPaddedWithZerosGivesTheSameBits()
    {
        // The detector's graph has layers that know nothing of runs; the runs must stop there, not reach a wrong place.
        var reference = ReferenceFixture.Load("det_96x128").Input();
        var input = new Tensor(1, 3, 96, 192);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 96; y++)
                for (int x = 0; x < 128; x++)
                    input.Span[((c * 96) + y) * 192 + x] = reference[0, c, y, x];
        Assert.True(Run(ModelStore.Detection, input, 2, skip: false).AsSpan().SequenceEqual(Run(ModelStore.Detection, input, 2, skip: true)));
        Assert.True(Run(ModelStore.Classifier, PaddedBatch(192), 2, skip: false).AsSpan().SequenceEqual(Run(ModelStore.Classifier, PaddedBatch(192), 2, skip: true)));
    }
}
