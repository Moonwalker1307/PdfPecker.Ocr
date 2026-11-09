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

    [Theory]
    [InlineData("det", "det_96x128")]
    [InlineData("rec", "rec_2x48x160")]
    public void EveryMatrixKernelGivesTheSameBits(string name, string fixture)
    {
        // Each kernel sums a value's products in the same order with fused multiply-adds, whatever its block size, so
        // the AVX-512, AVX2 and portable kernels agree to the bit where the processor runs them all.
        byte[] data = ModelStore.Inflate(name).ToArray();
        var input = ReferenceFixture.Load(fixture).Input();
        float[]? first = null;
        foreach (var kernel in GemmKernel.Supported)
        {
            using var context = new InferenceContext(2);
            float[] result = OcrModel.Load(data, kernel).Run(input, context).Span.ToArray();
            if (first is null) first = result;
            else Assert.True(first.AsSpan().SequenceEqual(result), $"{kernel.Name} differs");
        }
    }

    [Theory]
    [InlineData("rec_1x48x320")]
    [InlineData("rec_2x48x160")]
    public void ArgMaxOutputIsTheFullOutputsBestClass(string fixture)
    {
        // The engine's recogniser returns only the best class of each time step and its probability; they must be the
        // bits a search of the full softmax finds, with every matrix kernel and degree of parallelism.
        byte[] data = ModelStore.Inflate("rec").ToArray();
        var input = ReferenceFixture.Load(fixture).Input();
        foreach (var kernel in GemmKernel.Supported)
        {
            var full = OcrModel.Load(data, kernel);
            var argMax = full.WithArgMaxOutput();
            float[] probabilities;
            int[] shape;
            using (var context = new InferenceContext(2))
            {
                var result = full.Run(input, context);
                probabilities = result.Span.ToArray();
                shape = result.Shape;
            }

            int classes = shape[^1], rows = probabilities.Length / classes;
            foreach (int degree in new[] { 1, 2, 4, 3 })
            {
                using var context = new InferenceContext(degree);
                var best = argMax.Run(input, context);
                Assert.Equal([.. shape[..^1], 2], best.Shape);
                for (int row = 0; row < rows; row++)
                {
                    var (index, value) = PdfPecker.Ocr.Recognition.CtcDecoder.ArgMax(probabilities.AsSpan(row * classes, classes));
                    Assert.Equal(index, (int)best.Span[2 * row]);
                    Assert.Equal(BitConverter.SingleToInt32Bits(value), BitConverter.SingleToInt32Bits(best.Span[(2 * row) + 1]));
                }
            }
        }
    }

    [Fact]
    public void ArgMaxOutputOfASmallHead()
    {
        // Rows with a few close logits and classes that are not a multiple of any vector width or panel.
        foreach (int classes in new[] { 1, 3, 17, 300, 1031 })
        {
            var graph = new GraphBuilder(3);
            int l = graph.Add(OpKind.Linear, [0], [(int)ActivationKind.None, 0], [0, 0], Weight([24, classes], classes, 4f), Weight([classes], classes + 1));
            graph.Add(OpKind.Softmax, [l], []);
            var x = RandomTensor(classes, 2, 29, 24);
            foreach (var kernel in GemmKernel.Supported)
            {
                var model = OcrModel.FromDefinition(graph.Model, kernel);
                float[] probabilities;
                using (var context = new InferenceContext(1)) probabilities = model.Run(x, context).Span.ToArray();
                foreach (int degree in new[] { 1, 3 })
                {
                    using var context = new InferenceContext(degree);
                    var best = model.WithArgMaxOutput().Run(x, context);
                    Assert.Equal([2, 29, 2], best.Shape);
                    for (int row = 0; row < 58; row++)
                    {
                        var (index, value) = PdfPecker.Ocr.Recognition.CtcDecoder.ArgMax(probabilities.AsSpan(row * classes, classes));
                        Assert.Equal(index, (int)best.Span[2 * row]);
                        Assert.Equal(value, best.Span[(2 * row) + 1]);
                    }
                }
            }
        }
    }

    [Fact]
    public void ArgMaxOutputNeedsALinearSoftmaxHead()
    {
        var graph = new GraphBuilder(3);
        int l = graph.Add(OpKind.Linear, [0], [(int)ActivationKind.Gelu, 0], [0, 0], Weight([8, 5], 1), Weight([5], 2));
        graph.Add(OpKind.Softmax, [l], []);
        Assert.Throws<InvalidOperationException>(() => OcrModel.FromDefinition(graph.Model).WithArgMaxOutput());
        Assert.Throws<InvalidOperationException>(() => ModelStore.Detection.WithArgMaxOutput());
    }

    [Fact]
    public void GrayInputIsTheRepeatedPlaneBitForBit()
    {
        // A gray page's detector input is one plane read as all three channels; the output must be the bits of the
        // full model on the plane written three times, with every matrix kernel and degree of parallelism.
        byte[] data = ModelStore.Inflate("det").ToArray();
        var fixture = ReferenceFixture.Load("det_96x128").Input();
        int plane = 96 * 128;
        var gray = fixture.Span[..plane].ToArray();
        var repeated = new float[3 * plane];
        for (int c = 0; c < 3; c++) gray.CopyTo(repeated, c * plane);
        foreach (var kernel in GemmKernel.Supported)
        {
            var full = OcrModel.Load(data, kernel);
            var grayModel = full.WithGrayInput();
            Assert.Equal([-1, 1, -1, -1], grayModel.InputShape);
            float[] expected;
            using (var context = new InferenceContext(2))
                expected = full.Run(repeated, [1, 3, 96, 128], context, out _).ToArray();
            foreach (int degree in new[] { 1, 3 })
            {
                using var context = new InferenceContext(degree);
                var result = grayModel.Run(gray, [1, 1, 96, 128], context, out int[] shape);
                Assert.Equal([1, 1, 96, 128], shape);
                Assert.True(expected.AsSpan().SequenceEqual(result), $"{kernel.Name} with {degree} threads");
            }
        }
    }

    [Fact]
    public void GrayInputNeedsAFirstConvolutionThatAloneReadsTheInput()
    {
        var plain = new GraphBuilder(4);
        plain.Model.InputShape = [-1, 3, -1, -1];
        plain.Add(OpKind.Conv, [0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1), [0, 0], Weight([4, 3, 3, 3], 1), Weight([4], 2));
        Assert.Equal([-1, 1, -1, -1], OcrModel.FromDefinition(plain.Model).WithGrayInput().InputShape);
        var graph = new GraphBuilder(4);
        graph.Model.InputShape = [-1, 3, -1, -1];
        int a = graph.Add(OpKind.Conv, [0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1), [0, 0], Weight([3, 3, 3, 3], 1), Weight([3], 2));
        graph.Add(OpKind.Add, [a, 0], [1, 1]);
        Assert.Throws<InvalidOperationException>(() => OcrModel.FromDefinition(graph.Model).WithGrayInput());
        var pointwise = new GraphBuilder(4);
        pointwise.Model.InputShape = [-1, 3, -1, -1];
        pointwise.Add(OpKind.Conv, [0], ConvInts(1, 1), [0, 0], Weight([4, 3, 1, 1], 1), Weight([4], 2));
        Assert.Throws<InvalidOperationException>(() => OcrModel.FromDefinition(pointwise.Model).WithGrayInput());
        Assert.Throws<InvalidOperationException>(() => ModelStore.DetectionGray.WithGrayInput());
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
