using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Inference.Layers;
using PdfPecker.Ocr.Models;
using static PdfPecker.Ocr.Tests.Inference.TestModels;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// The experimental 8-bit pointwise convolution (<see cref="Int8Gemm"/>): the AVX-VNNI kernel against its scalar
/// reference bit for bit, the layer against the float convolution within the error 8-bit quantisation allows, and the
/// same bits at every degree of parallelism. Skipped in effect on processors without AVX-VNNI and on .NET 8.
/// </summary>
public sealed class Int8Tests
{
    [Fact]
    public void KernelMatchesItsScalarReferenceBitForBit()
    {
        if (!Int8Gemm.IsSupported) return;
        foreach (var (rows, k, cols) in new[] { (1, 1, 1), (6, 4, 16), (7, 5, 17), (13, 19, 33), (48, 96, 160), (96, 192, 100), (25, 383, 70) })
        {
            float[] w = Random(rows * k, rows + k, 0.4f);
            float[] b = Random(k * cols, k * 3 + cols, 2f);
            // A column of zeros, which must quantise to zeros rather than NaN.
            for (int kk = 0; kk < k; kk++) b[kk * cols] = 0;
            float[] bias = Random(rows, rows * 7, 1f);
            var weights = Int8Gemm.QuantizeWeights(w, rows, k);
            var expected = new float[rows * cols];
            Int8Gemm.Reference(weights, b, k, cols, expected, bias);
            var actual = Multiply(weights, b, k, cols, bias, rows);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]), $"{rows}x{k}x{cols} at {i}: {expected[i]} != {actual[i]}");
        }
    }

    [Fact]
    public unsafe void ActivationAndResidualInRegistersMatchTheSeparatePass()
    {
        if (!Int8Gemm.IsSupported) return;
        foreach (var (rows, k, cols) in new[] { (12, 40, 32), (13, 19, 45), (96, 192, 100) })
        {
            float[] w = Random(rows * k, rows + k, 0.4f);
            float[] b = Random(k * cols, k * 3 + cols, 2f);
            float[] residual = Random(rows * cols, rows * cols, 1f);
            float[] bias = Random(rows, rows * 7, 1f);
            var weights = Int8Gemm.QuantizeWeights(w, rows, k);
            var expected = new float[rows * cols];
            Int8Gemm.Reference(weights, b, k, cols, expected, bias);
            for (int i = 0; i < expected.Length; i++)
                expected[i] = VectorMath.Activate(expected[i], ActivationKind.HardSwish, 1 / 6f, 0.5f) + residual[i];
            fixed (float* r = residual)
            {
                var actual = Multiply(weights, b, k, cols, bias, rows, new Epilogue(ActivationKind.HardSwish, 1 / 6f, 0.5f, r));
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]), $"{rows}x{k}x{cols} at {i}: {expected[i]} != {actual[i]}");
            }
        }
    }

    [Fact]
    public void LayerIsCloseToTheFloatConvolution()
    {
        if (!Int8Gemm.IsSupported) return;
        foreach (var (cin, cout, h, w) in new[] { (37, 21, 9, 13), (96, 48, 7, 30), (192, 384, 3, 41) })
        {
            var x = RandomTensor(cin * 7 + h, 2, cin, h, w);
            var weight = Weight([cout, cin, 1, 1], cout);
            var bias = Weight([cout], cout + 3);
            var graph = new GraphBuilder(4);
            graph.Add(OpKind.Conv, [0], ConvInts(1, 1), [0, 0], weight, bias);
            var expected = Conv(x, weight.Data, bias.Data, cout, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, ActivationKind.None, 0, 0, out var shape);
            var model = OcrModel.FromDefinition(graph.Model, int8Pointwise: true);
            Assert.IsType<Int8PointwiseConvLayer>(model.Layers[0]);
            using var context = new InferenceContext(2);
            var result = model.Run(x, context).Clone();
            Assert.Equal(shape, result.Shape);
            // Inputs in [-1, 1] and weights in [-0.5, 0.5]: one quantisation step of each is about 1/254 of its range,
            // so a sum of cin products errs by a few hundredths at most.
            double worst = 0;
            for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - result.Span[i]));
            Assert.True(worst < 0.01 * Math.Sqrt(cin), $"{cin}->{cout}: worst error {worst}");
        }
    }

    [Fact]
    public void LayerGivesTheSameBitsAtEveryDegree()
    {
        if (!Int8Gemm.IsSupported) return;
        var x = RandomTensor(5, 3, 96, 6, 101);
        var weight = Weight([192, 96, 1, 1], 9);
        var bias = Weight([192], 10);
        var graph = new GraphBuilder(4);
        graph.Add(OpKind.Conv, [0], ConvInts(1, 1, act: ActivationKind.HardSwish), [1 / 6f, 0.5f], weight, bias);
        var model = OcrModel.FromDefinition(graph.Model, int8Pointwise: true);
        float[]? first = null;
        foreach (int degree in new[] { 1, 2, 3, 4 })
        {
            using var context = new InferenceContext(degree);
            float[] output = model.Run(x, context).Span.ToArray();
            first ??= output;
            Assert.Equal(first, output);
        }
    }

    [Fact]
    public void OnlyPointwiseConvolutionsRunInEightBits()
    {
        var graph = new GraphBuilder(4);
        int a = graph.Add(OpKind.Conv, [0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1), [0, 0], Weight([32, 4, 3, 3], 1), Weight([32], 2));
        int b = graph.Add(OpKind.Conv, [a], ConvInts(1, 1), [0, 0], Weight([8, 32, 1, 1], 3), Weight([8], 4));
        graph.Add(OpKind.Conv, [b], ConvInts(1, 1), [0, 0], Weight([4, 8, 1, 1], 5), Weight([4], 6));
        var model = OcrModel.FromDefinition(graph.Model, int8Pointwise: true);
        Assert.IsType<GemmConvLayer>(model.Layers[0]);
        if (Int8Gemm.IsSupported) Assert.IsType<Int8PointwiseConvLayer>(model.Layers[1]);
        else Assert.IsType<GemmConvLayer>(model.Layers[1]);
        // Eight input channels are too few to gain from 8 bits.
        Assert.IsType<GemmConvLayer>(model.Layers[2]);
        Assert.IsType<GemmConvLayer>(OcrModel.FromDefinition(graph.Model).Layers[1]);
    }

    private static unsafe float[] Multiply(Int8Gemm.Weights weights, float[] b, int k, int cols, float[] bias, int rows, Epilogue epilogue = default)
    {
        var packed = new byte[Int8Gemm.PackedBytes(k, cols) + 64];
        var padded = new float[GemmKernel.RoundUp(rows, Int8Gemm.MR) + Int8Gemm.MR];
        bias.CopyTo(padded, 0);
        var c = new float[rows * cols];
        fixed (float* src = b, pb = padded, pc = c)
        fixed (byte* dst = packed)
        {
            Int8Gemm.QuantizeColumns(src, cols, k, cols, dst);
            Int8Gemm.Multiply(weights, 0, rows, dst, cols, pc, cols, pb, epilogue);
        }

        return c;
    }
}
