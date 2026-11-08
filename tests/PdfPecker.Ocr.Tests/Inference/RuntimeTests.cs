using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;
using static PdfPecker.Ocr.Tests.Inference.TestModels;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// What callers rely on besides the numbers: bitwise identical results for any degree of parallelism, the context's
/// ownership of outputs, cancellation, the documented exceptions, and models shared by threads.
/// </summary>
public sealed class RuntimeTests
{
    [Theory]
    [InlineData("det_320x320")]
    [InlineData("rec_2x48x160")]
    [InlineData("cls_2x48x192")]
    public void ResultsDoNotDependOnTheDegreeOfParallelism(string fixture)
    {
        var reference = ReferenceFixture.Load(fixture);
        var model = reference.Model switch
        {
            "det" => ModelStore.Detection,
            "rec" => ModelStore.Recognition,
            _ => ModelStore.Classifier,
        };
        var input = reference.Input();
        float[]? first = null;
        foreach (int degree in new[] { 1, 2, 4, 3 })
        {
            using var context = new InferenceContext(degree);
            float[] result = model.Run(input, context).Span.ToArray();
            if (first is null) first = result;
            else Assert.True(first.AsSpan().SequenceEqual(result), $"degree {degree} differs from degree 1");
        }
    }

    [Fact]
    public void LayersAreDeterministicAcrossDegrees()
    {
        // The detector and recogniser above cover the layers they use at their shapes; this covers the tiling of a
        // matrix product and a convolution large enough to give every worker several tiles.
        var x = RandomTensor(1, 3, 64, 37, 53);
        var graph = new GraphBuilder(4);
        int a = graph.Add(OpKind.Conv, [0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1, act: ActivationKind.HardSwish), [1 / 6f, 0.5f], Weight([96, 64, 3, 3], 1), Weight([96], 2));
        int b = graph.Add(OpKind.Conv, [a], ConvInts(1, 1, act: ActivationKind.Relu), [0, 0], Weight([80, 96, 1, 1], 3), Weight([80], 4));
        int c = graph.Add(OpKind.Conv, [b], ConvInts(5, 5, 2, 2, pt: 2, pl: 2, pb: 2, pr: 2, group: 80), [0, 0], Weight([80, 1, 5, 5], 5), Weight([80], 6));
        int t = graph.Add(OpKind.Transpose, [c], [0, 2, 3, 1]);
        int r = graph.Add(OpKind.Reshape, [t], [0, -1, 80]);
        int l = graph.Add(OpKind.Linear, [r], [(int)ActivationKind.Gelu, 0], [0, 0], Weight([80, 120], 7), Weight([120], 8));
        graph.Add(OpKind.LayerNorm, [l], [], [1e-5f], Weight([120], 9), Weight([120], 10));
        foreach (var kernel in GemmKernel.Supported)
        {
            float[] one = graph.Run(x, kernel, degree: 1).Span.ToArray();
            foreach (int degree in new[] { 2, 4 })
                Assert.True(one.AsSpan().SequenceEqual(graph.Run(x, kernel, degree).Span), $"{kernel.Name} with {degree} threads");
        }
    }

    [Fact]
    public void OutputBelongsToTheContextUntilItsNextRun()
    {
        var model = ModelStore.Classifier;
        using var context = new InferenceContext(2);
        var upright = ReferenceFixture.Load("cls_2x48x192").Input();
        var noise = RandomTensor(7, 2, 3, 48, 192);
        var first = model.Run(upright, context);
        var kept = first.Clone();
        var second = model.Run(noise, context);
        Assert.True(first.Span.SequenceEqual(second.Span));
        Assert.False(kept.Span.SequenceEqual(second.Span));
        Assert.True(kept.Span.SequenceEqual(model.Run(upright, context).Span));
    }

    [Fact]
    public void ShapeChangesBetweenRunsReplanTheContext()
    {
        var model = ModelStore.Recognition;
        var small = ReferenceFixture.Load("rec_2x48x160");
        var large = ReferenceFixture.Load("rec_1x48x320");
        using var shared = new InferenceContext(2);
        float[] a = model.Run(small.Input(), shared).Span.ToArray();
        float[] b = model.Run(large.Input(), shared).Span.ToArray();
        float[] c = model.Run(small.Input(), shared).Span.ToArray();
        Assert.Equal(a, c);
        using var fresh = new InferenceContext(2);
        Assert.Equal(b, model.Run(large.Input(), fresh).Span.ToArray());
    }

    [Fact]
    public void BadInputsThrowArgumentException()
    {
        var model = ModelStore.Detection;
        using var context = new InferenceContext(1);
        Assert.Throws<ArgumentException>(() => model.Run(new Tensor(1, 4, 32, 32), context));
        Assert.Throws<ArgumentException>(() => model.Run(new Tensor(3, 32, 32), context));
        Assert.Throws<ArgumentException>(() => ModelStore.Recognition.Run(new Tensor(1, 3, 32, 320), context));
        Assert.Throws<ArgumentException>(() => ModelStore.Classifier.Run(new Tensor(1, 3, 47, 192), context));
        Assert.Throws<ArgumentNullException>(() => model.Run(null!, context));
        Assert.Throws<ArgumentException>(() => Tensor.FromBuffer(new float[5], 2, 3));
        Assert.Throws<ArgumentException>(() => new Tensor(-1, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new InferenceContext(0));
    }

    [Fact]
    public void CancellationStopsARun()
    {
        using var cancel = new CancellationTokenSource();
        cancel.Cancel();
        using var context = new InferenceContext(2) { CancellationToken = cancel.Token };
        Assert.Throws<OperationCanceledException>(() => ModelStore.Classifier.Run(new Tensor(1, 3, 48, 192), context));
        context.CancellationToken = CancellationToken.None;
        Assert.Equal([1, 2], ModelStore.Classifier.Run(new Tensor(1, 3, 48, 192), context).Shape);
    }

    [Fact]
    public void DisposedContextThrows()
    {
        var context = new InferenceContext(1);
        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ModelStore.Classifier.Run(new Tensor(1, 3, 48, 192), context));
    }

    [Fact]
    public void ModelsAreSharedByThreads()
    {
        var input = ReferenceFixture.Load("rec_1x48x320").Input();
        float[] expected;
        using (var context = new InferenceContext(1)) expected = ModelStore.Recognition.Run(input, context).Span.ToArray();
        var results = new float[4][];
        Parallel.For(0, 4, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
        {
            using var context = new InferenceContext(1 + i % 2);
            for (int run = 0; run < 2; run++) results[i] = ModelStore.Recognition.Run(input, context).Span.ToArray();
        });
        Assert.All(results, r => Assert.Equal(expected, r));
    }

    [Fact]
    public void BundledModelsCarryTheirNotices()
    {
        foreach (var model in new[] { ModelStore.Detection, ModelStore.Recognition, ModelStore.Classifier })
        {
            Assert.Contains("Apache License, Version 2.0", model.Notice, StringComparison.Ordinal);
            Assert.Contains("Changed by PdfPecker.Ocr", model.Notice, StringComparison.Ordinal);
        }
        Assert.Equal([-1, 3, -1, -1], ModelStore.Detection.InputShape);
        Assert.Equal([-1, 3, 48, -1], ModelStore.Recognition.InputShape);
        Assert.Equal([-1, 3, 48, -1], ModelStore.Classifier.InputShape);
        Assert.Empty(ModelStore.Detection.Characters);
        Assert.Empty(ModelStore.Classifier.Characters);
    }
}
