using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;
using static PdfPecker.Ocr.Tests.Inference.TestModels;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// Every layer against a naive double-precision implementation of the ONNX operator it replaces, with every matrix
/// multiplication kernel the processor can run. Shapes are chosen to leave partial register blocks and tiles at the
/// edges, which is where blocked kernels go wrong.
/// </summary>
public sealed class KernelTests
{
    private const double Tolerance = 2e-5;

    public static TheoryData<string> Kernels
    {
        get
        {
            var data = new TheoryData<string>();
            foreach (var kernel in GemmKernel.Supported) data.Add(kernel.Name);
            return data;
        }
    }

    private static GemmKernel Kernel(string name) => GemmKernel.Supported.Single(k => k.Name == name);

    [Theory]
    [MemberData(nameof(Kernels))]
    public void MatrixProductMatchesNaive(string kernel)
    {
        // Linear layers are the plainest use of the kernel: rows x K activations times packed K x N weights.
        foreach (var (rows, k, n) in new[] { (1, 1, 1), (5, 3, 7), (13, 17, 33), (12, 32, 32), (25, 64, 100), (61, 120, 360), (7, 600, 45) })
        {
            var x = RandomTensor(rows * 31 + k, 1, rows, k);
            var w = Weight([k, n], n);
            var b = Weight([n], n + 1);
            var graph = new GraphBuilder(3);
            graph.Add(OpKind.Linear, [0], [0, 0], [0, 0], w, b);
            var expected = TestModels.Linear(x, w.Data, b.Data, n, ActivationKind.None, 0, 0);
            AssertClose(expected, graph.Run(x, Kernel(kernel)).Span, Tolerance, $"{rows}x{k}x{n}");
        }
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void LinearAppliesActivationAndResidual(string kernel)
    {
        var x = RandomTensor(3, 2, 19, 48);
        var w = Weight([48, 48], 4);
        var b = Weight([48], 5);
        var graph = new GraphBuilder(3);
        graph.Add(OpKind.Linear, [0, 0], [(int)ActivationKind.Gelu, 1], [0, 0], w, b);
        var expected = TestModels.Linear(x, w.Data, b.Data, 48, ActivationKind.Gelu, 0, 0, residual: x);
        AssertClose(expected, graph.Run(x, Kernel(kernel)).Span, Tolerance, "linear gelu residual");
    }

    public static TheoryData<string, string> ConvCases
    {
        get
        {
            var data = new TheoryData<string, string>();
            string[] cases = ["pointwise", "pointwise-stride2", "stem3x3s2", "3x3", "3x3-dilated", "2x2-same", "1x7", "3x3-stride2x1", "5x5-asymmetric", "3x3-stride2-wide", "large-k"];
            foreach (var kernel in GemmKernel.Supported)
                foreach (string c in cases) data.Add(kernel.Name, c);
            return data;
        }
    }

    [Theory]
    [MemberData(nameof(ConvCases))]
    public void ConvolutionMatchesNaive(string kernel, string name)
    {
        // (cin, cout, h, w, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr)
        var (cin, cout, h, w, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr) = name switch
        {
            "pointwise" => (19, 21, 9, 13, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
            "pointwise-stride2" => (8, 12, 9, 14, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0),
            "stem3x3s2" => (3, 16, 17, 23, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1),
            "3x3" => (10, 7, 11, 37, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
            "3x3-dilated" => (6, 9, 15, 12, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2),
            "2x2-same" => (5, 11, 10, 15, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1),
            "1x7" => (4, 6, 5, 40, 1, 7, 1, 1, 1, 1, 0, 3, 0, 3),
            "3x3-stride2x1" => (7, 5, 12, 21, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1),
            "5x5-asymmetric" => (3, 4, 13, 11, 5, 5, 1, 2, 1, 1, 2, 1, 3, 0),
            // Output rows of 38 pixels: whole panels of every kernel read with stride 2, and runs split across rows.
            "3x3-stride2-wide" => (5, 13, 9, 75, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1),
            _ => (96, 40, 7, 30, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1),
        };
        var x = RandomTensor(cin * 7 + h, 2, cin, h, w);
        var weight = Weight([cout, cin, kh, kw], cout);
        var bias = Weight([cout], cout + 3);
        var graph = new GraphBuilder(4);
        graph.Add(OpKind.Conv, [0], ConvInts(kh, kw, sh, sw, dh, dw, pt, pl, pb, pr), [0, 0], weight, bias);
        var expected = Conv(x, weight.Data, bias.Data, cout, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, 1, ActivationKind.None, 0, 0, out var shape);
        var result = graph.Run(x, Kernel(kernel));
        Assert.Equal(shape, result.Shape);
        AssertClose(expected, result.Span, Tolerance, name);
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void ConvolutionPanelsReadInPlaceEqualPackedOnes(string kernel)
    {
        // The same input once more with one column in front (two for a stride of 2): the output columns move by one,
        // so the panels fall on other pixels, and pixels whose panel was read in place in one run are packed in the
        // other (across the end of an output row). Without padding both runs sum the same values in the same order,
        // so every output must match to the bit, after an activation too; 26 output channels leave a partial block
        // of rows, and the vertical stride of 2 skips rows.
        foreach (var (kh, kw, dh, dw, sh, sw) in new[] { (3, 3, 1, 1, 1, 1), (2, 2, 1, 1, 1, 1), (1, 7, 1, 1, 1, 1), (3, 3, 2, 2, 1, 1), (3, 3, 1, 1, 2, 2), (3, 3, 1, 1, 1, 2), (3, 5, 2, 1, 2, 2) })
        {
            int cin = 6, cout = 26, h = 11, w = sw == 1 ? 150 : 299;
            var wide = RandomTensor(kh * 31 + kw + sw, 2, cin, h, w + sw);
            var narrow = new Tensor([2, cin, h, w]);
            for (int row = 0; row < 2 * cin * h; row++) wide.Span.Slice(row * (w + sw) + sw, w).CopyTo(narrow.Span.Slice(row * w, w));
            var graph = new GraphBuilder(4);
            graph.Add(OpKind.Conv, [0], ConvInts(kh, kw, sh, sw, dh, dw, act: ActivationKind.Gelu), [0, 0], Weight([cout, cin, kh, kw], cout + kh), Weight([cout], cout + 5));
            var a = graph.Run(wide, Kernel(kernel));
            var b = graph.Run(narrow, Kernel(kernel));
            int oh = b.Shape[2], ow = b.Shape[3];
            Assert.Equal(ow + 1, a.Shape[3]);
            for (int row = 0; row < 2 * cout * oh; row++)
            {
                for (int x = 0; x < ow; x++)
                {
                    float expected = a.Span[row * (ow + 1) + x + 1], actual = b.Span[row * ow + x];
                    Assert.True(BitConverter.SingleToUInt32Bits(expected) == BitConverter.SingleToUInt32Bits(actual), $"{kh}x{kw}/{sh}x{sw} row {row}, column {x}: {actual} against {expected}");
                }
            }
        }
    }

    public static TheoryData<string> DepthwiseCases => new()
    {
        "3x3", "3x3-stride2", "3x3-stride2x1", "5x5", "7x7", "1x7", "7x1-stride2", "3x3-asymmetric", "5x5-stride2-odd", "wide", "wide-stride2",
    };

    [Theory]
    [MemberData(nameof(DepthwiseCases))]
    public void DepthwiseConvolutionMatchesNaive(string name)
    {
        var (c, h, w, kh, kw, sh, sw, pt, pl, pb, pr) = name switch
        {
            "3x3" => (6, 9, 13, 3, 3, 1, 1, 1, 1, 1, 1),
            "3x3-stride2" => (5, 12, 18, 3, 3, 2, 2, 1, 1, 1, 1),
            "3x3-stride2x1" => (4, 12, 21, 3, 3, 2, 1, 1, 1, 1, 1),
            "5x5" => (3, 11, 19, 5, 5, 1, 1, 2, 2, 2, 2),
            "7x7" => (7, 16, 41, 7, 7, 1, 1, 3, 3, 3, 3),
            "1x7" => (2, 3, 25, 1, 7, 1, 1, 0, 3, 0, 3),
            "7x1-stride2" => (3, 21, 6, 7, 1, 2, 2, 3, 0, 3, 0),
            "3x3-asymmetric" => (4, 8, 9, 3, 3, 2, 2, 0, 0, 1, 1),
            "5x5-stride2-odd" => (3, 13, 15, 5, 5, 2, 2, 2, 2, 2, 2),
            "wide-stride2" => (3, 7, 163, 5, 5, 2, 2, 2, 2, 2, 2),
            _ => (2, 5, 133, 3, 3, 1, 1, 1, 1, 1, 1),
        };
        var x = RandomTensor(c * 11 + w, 2, c, h, w);
        var weight = Weight([c, 1, kh, kw], c * 3);
        var bias = Weight([c], c * 5);
        var graph = new GraphBuilder(4);
        graph.Add(OpKind.Conv, [0], ConvInts(kh, kw, sh, sw, 1, 1, pt, pl, pb, pr, group: c, act: ActivationKind.HardSwish), [1 / 6f, 0.5f], weight, bias);
        var expected = Conv(x, weight.Data, bias.Data, c, kh, kw, sh, sw, 1, 1, pt, pl, pb, pr, c, ActivationKind.HardSwish, 1 / 6f, 0.5f, out var shape);
        var result = graph.Run(x);
        Assert.Equal(shape, result.Shape);
        AssertClose(expected, result.Span, Tolerance, name);
    }

    public static TheoryData<int, float, float> Activations => new()
    {
        { (int)ActivationKind.Relu, 0, 0 },
        { (int)ActivationKind.Sigmoid, 0, 0 },
        { (int)ActivationKind.HardSigmoid, 0.2f, 0.5f },
        { (int)ActivationKind.HardSwish, 1 / 6f, 0.5f },
        { (int)ActivationKind.Swish, 0, 0 },
        { (int)ActivationKind.Gelu, 0, 0 },
    };

    [Theory]
    [MemberData(nameof(Activations))]
    public void ConvolutionsApplyActivationThenResidual(int activation, float alpha, float beta)
    {
        var act = (ActivationKind)activation;
        // Inputs large enough to reach the saturated ends of the sigmoid and GELU.
        var x = Tensor.FromBuffer(Random(2 * 8 * 7 * 19, activation, 4f), 2, 8, 7, 19);
        foreach (var kernel in GemmKernel.Supported)
        {
            var w = Weight([8, 8, 3, 3], 1);
            var b = Weight([8], 2);
            var graph = new GraphBuilder(4);
            graph.Add(OpKind.Conv, [0, 0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1, act: act, residual: true), [alpha, beta], w, b);
            var expected = Conv(x, w.Data, b.Data, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, act, alpha, beta, out _, residual: x);
            AssertClose(expected, graph.Run(x, kernel).Span, Tolerance, $"{act} {kernel.Name}");
        }

        var dw = Weight([8, 1, 3, 3], 3);
        var db = Weight([8], 4);
        var depthwise = new GraphBuilder(4);
        depthwise.Add(OpKind.Conv, [0, 0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1, group: 8, act: act, residual: true), [alpha, beta], dw, db);
        var dexpected = Conv(x, dw.Data, db.Data, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 8, act, alpha, beta, out _, residual: x);
        AssertClose(dexpected, depthwise.Run(x).Span, Tolerance, $"{act} depthwise");

        var standalone = new GraphBuilder(4);
        standalone.Add(OpKind.Activation, [0], [activation], [alpha, beta]);
        var values = x.Span.ToArray().Select(v => Activate(v, act, alpha, beta)).ToArray();
        AssertClose(values, standalone.Run(x).Span, Tolerance, $"{act} standalone");
    }

    [Fact]
    public void ActivationsAgreeAcrossVectorWidths()
    {
        // Rows run through 512-bit vectors where the processor has AVX-512 and through Vector<T> otherwise and for the
        // tail; a value must get the same bits either way, or results would depend on where a tile boundary falls.
        float[] special = [0f, -0f, 1e-30f, -1e-30f, 87f, -87f, 88.5f, -88.5f, 100f, -100f, 3f, -3f, 4f, -4f, 5.7f, -5.7f, 0.5f];
        float[] values = [.. special, .. Random(4096, 11, 30f), .. special];
        foreach (var kind in Enum.GetValues<ActivationKind>())
        {
            var graph = new GraphBuilder(2);
            graph.Add(OpKind.Activation, [0], [(int)kind], [0.2f, 0.5f]);
            var row = graph.Run(Tensor.FromBuffer((float[])values.Clone(), 1, values.Length)).Span;
            for (int i = 0; i < values.Length; i++)
            {
                float single = VectorMath.Activate(values[i], kind, 0.2f, 0.5f);
                Assert.True(BitConverter.SingleToUInt32Bits(single) == BitConverter.SingleToUInt32Bits(row[i]), $"{kind}({values[i]}) at {i}: row {row[i]}, single {single}");
            }
        }
    }

    [Fact]
    public void ErfIsAccurate()
    {
        double worst = 0;
        for (int i = -6000; i <= 6000; i++)
        {
            float x = i / 1000f;
            float erf = VectorMath.Erf(new System.Numerics.Vector<float>(x))[0];
            worst = Math.Max(worst, Math.Abs(erf - Erf(x)));
        }
        Assert.True(worst < 5e-7, $"max error {worst}");
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void TransposedConvolutionMatchesNaive(string kernel)
    {
        foreach (var (cin, cout, h, w, k) in new[] { (12, 5, 6, 9, 2), (3, 2, 4, 5, 3), (33, 16, 5, 17, 2) })
        {
            var x = RandomTensor(cin + h, 2, cin, h, w);
            var weight = Weight([cin, cout, k, k], cout);
            var bias = Weight([cout], cout + 1);
            var graph = new GraphBuilder(4);
            graph.Add(OpKind.ConvTranspose, [0], [k, k, (int)ActivationKind.Relu], [0, 0], weight, bias);
            var expected = TestModels.ConvTranspose(x, weight.Data, bias.Data, cout, k, k, ActivationKind.Relu, 0, 0);
            var result = graph.Run(x, Kernel(kernel));
            Assert.Equal([2, cout, h * k, w * k], result.Shape);
            AssertClose(expected, result.Span, Tolerance, $"{cin}->{cout} k{k}");
        }
    }

    [Theory]
    [InlineData(true, 2, 2, 1, 1, 0, 0, 1, 1, false)]
    [InlineData(true, 2, 2, 1, 1, 0, 0, 0, 0, false)]
    [InlineData(true, 3, 3, 2, 2, 1, 1, 1, 1, false)]
    [InlineData(true, 2, 2, 2, 2, 0, 0, 0, 0, false)]
    [InlineData(false, 2, 2, 2, 2, 0, 0, 0, 0, false)]
    [InlineData(false, 3, 3, 2, 2, 1, 1, 1, 1, false)]
    [InlineData(false, 3, 3, 2, 2, 1, 1, 1, 1, true)]
    [InlineData(false, 3, 2, 3, 2, 1, 0, 2, 1, true)]
    [InlineData(false, 3, 2, 3, 2, 0, 0, 0, 0, false)]
    [InlineData(true, 3, 3, 1, 1, 0, 0, 0, 0, false)]
    [InlineData(false, 2, 3, 1, 2, 0, 0, 0, 0, false)]
    public void PoolingMatchesNaive(bool max, int kh, int kw, int sh, int sw, int pt, int pl, int pb, int pr, bool countIncludePad)
    {
        foreach (var (h, w) in new[] { (9, 13), (16, 40), (2, 3) })
        {
            if (h + pt + pb < kh || w + pl + pr < kw) continue;
            var x = RandomTensor(h * w, 2, 3, h, w);
            var graph = new GraphBuilder(4);
            graph.Add(max ? OpKind.MaxPool : OpKind.AveragePool, [0], [kh, kw, sh, sw, pt, pl, pb, pr, countIncludePad ? 1 : 0]);
            var expected = Pool(x, max, kh, kw, sh, sw, pt, pl, pb, pr, countIncludePad);
            AssertClose(expected, graph.Run(x).Span, Tolerance, $"{h}x{w}");
        }
    }

    [Fact]
    public void GlobalAveragePoolMatchesNaive()
    {
        var x = RandomTensor(5, 3, 7, 11, 13);
        var graph = new GraphBuilder(4);
        graph.Add(OpKind.GlobalAveragePool, [0], []);
        var result = graph.Run(x);
        Assert.Equal([3, 7, 1, 1], result.Shape);
        var expected = new double[21];
        for (int p = 0; p < 21; p++)
        {
            for (int i = 0; i < 143; i++) expected[p] += x.Span[p * 143 + i];
            expected[p] /= 143;
        }
        AssertClose(expected, result.Span, Tolerance, "global average");
    }

    [Theory]
    [InlineData(false, 16, 4)]
    [InlineData(true, 16, 4)]
    [InlineData(false, 21, 5)]
    [InlineData(true, 37, 10)]
    public void SqueezeExciteMatchesNaive(bool addInput, int channels, int hidden)
    {
        var x = RandomTensor(9, 2, channels, 9, 11);
        var w1 = Weight([hidden, channels], 1);
        var b1 = Weight([hidden], 2);
        var w2 = Weight([channels, hidden], 3);
        var b2 = Weight([channels], 4);
        var graph = new GraphBuilder(4);
        graph.Add(OpKind.SqueezeExcite, [0], [(int)ActivationKind.Relu, (int)ActivationKind.HardSigmoid, addInput ? 1 : 0], [0, 0, 0.2f, 0.5f], w1, b1, w2, b2);
        var expected = TestModels.SqueezeExcite(x, w1.Data, b1.Data, w2.Data, b2.Data, ActivationKind.Relu, ActivationKind.HardSigmoid, 0.2f, 0.5f, addInput);
        AssertClose(expected, graph.Run(x).Span, Tolerance, "squeeze-excite");
    }

    [Fact]
    public void LayerNormMatchesNaive()
    {
        foreach (int c in new[] { 7, 64, 120 })
        {
            var x = Tensor.FromBuffer(Random(2 * 13 * c, c, 3f).Select(v => v + 1.5f).ToArray(), 2, 13, c);
            var gamma = Weight([c], 1);
            var beta = Weight([c], 2);
            var graph = new GraphBuilder(3);
            graph.Add(OpKind.LayerNorm, [0], [], [1e-6f], gamma, beta);
            AssertClose(TestModels.LayerNorm(x, gamma.Data, beta.Data, 1e-6f), graph.Run(x).Span, Tolerance, $"layer norm {c}");
        }
    }

    [Fact]
    public void SoftmaxMatchesNaive()
    {
        foreach (int c in new[] { 1, 2, 9, 18710 })
        {
            var x = Tensor.FromBuffer(Random(3 * c, c, 20f), 3, c);
            var graph = new GraphBuilder(2);
            graph.Add(OpKind.Softmax, [0], []);
            AssertClose(TestModels.Softmax(x.Span, c), graph.Run(x).Span, 1e-6, $"softmax {c}");
        }
    }

    [Fact]
    public void AttentionMatchesNaive()
    {
        foreach (var (n, t, c, heads) in new[] { (1, 1, 8, 1), (2, 40, 120, 8), (3, 7, 24, 3) })
        {
            var x = Tensor.FromBuffer(Random(n * t * 3 * c, t, 2f), n, t, 3 * c);
            float scale = 1 / MathF.Sqrt(c / heads);
            var graph = new GraphBuilder(3);
            graph.Add(OpKind.Attention, [0], [heads], [scale]);
            var result = graph.Run(x);
            Assert.Equal([n, t, c], result.Shape);
            AssertClose(TestModels.Attention(x, heads, scale), result.Span, Tolerance, $"attention {n}x{t}x{c}/{heads}");
        }
    }

    [Fact]
    public void AttentionIsThePlainLoopsBitForBit()
    {
        // The vectorised layer computes each score and output as the plain loops do, sum by sum in the same order, so
        // the recogniser's results stay those of the scalar code it replaced.
        foreach (var (n, t, c, heads) in new[] { (1, 1, 8, 1), (2, 40, 120, 8), (3, 7, 24, 3), (1, 33, 45, 3), (2, 200, 120, 8) })
        {
            var x = Tensor.FromBuffer(Random(n * t * 3 * c, t + c, 3f), n, t, 3 * c);
            float scale = 1 / MathF.Sqrt(c / heads);
            var graph = new GraphBuilder(3);
            graph.Add(OpKind.Attention, [0], [heads], [scale]);
            var result = graph.Run(x).Span;
            int d = c / heads;
            var expected = new float[n * t * c];
            var scores = new float[t];
            var q = new float[d];
            var qkv = x.Span;
            for (int b = 0; b < n; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    for (int t1 = 0; t1 < t; t1++)
                    {
                        for (int i = 0; i < d; i++) q[i] = qkv[(b * t + t1) * 3 * c + h * d + i] * scale;
                        for (int t2 = 0; t2 < t; t2++)
                        {
                            float acc = 0f;
                            for (int i = 0; i < d; i++) acc += q[i] * qkv[(b * t + t2) * 3 * c + c + h * d + i];
                            scores[t2] = acc;
                        }

                        VectorMath.Softmax(scores);
                        for (int i = 0; i < d; i++)
                        {
                            float acc = 0f;
                            for (int t2 = 0; t2 < t; t2++) acc += scores[t2] * qkv[(b * t + t2) * 3 * c + 2 * c + h * d + i];
                            expected[(b * t + t1) * c + h * d + i] = acc;
                        }
                    }
                }
            }

            Assert.True(expected.AsSpan().SequenceEqual(result), $"attention {n}x{t}x{c}/{heads}");
        }
    }

    [Theory]
    [MemberData(nameof(Kernels))]
    public void ConcatOfConvolutionsWithUpsampling(string kernel)
    {
        // Two convolutions write straight into the concatenation's buffer, and the second one's half-size map is
        // upsampled as it is concatenated; an Add then reads an upsampled map too. Both are the detector's neck.
        var x = RandomTensor(1, 2, 6, 8, 12);
        var w1 = Weight([5, 6, 1, 1], 2);
        var b1 = Weight([5], 3);
        var w2 = Weight([4, 6, 3, 3], 4);
        var b2 = Weight([4], 5);
        var graph = new GraphBuilder(4);
        int a = graph.Add(OpKind.Conv, [0], ConvInts(1, 1), [0, 0], w1, b1);
        int b = graph.Add(OpKind.Conv, [0], ConvInts(3, 3, 2, 2, pt: 1, pl: 1, pb: 1, pr: 1), [0, 0], w2, b2);
        graph.Add(OpKind.Concat, [a, b, 0], [1, 1, 1, 2, 2, 1, 1]);
        var result = graph.Run(x, Kernel(kernel));
        Assert.Equal([2, 15, 8, 12], result.Shape);

        var first = Conv(x, w1.Data, b1.Data, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, ActivationKind.None, 0, 0, out _);
        var second = Conv(x, w2.Data, b2.Data, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, ActivationKind.None, 0, 0, out _);
        var expected = new double[2 * 15 * 96];
        for (int n = 0; n < 2; n++)
            for (int c = 0; c < 15; c++)
                for (int y = 0; y < 8; y++)
                    for (int xx = 0; xx < 12; xx++)
                        expected[((n * 15 + c) * 8 + y) * 12 + xx] = c < 5 ? first[((n * 5 + c) * 8 + y) * 12 + xx]
                            : c < 9 ? second[((n * 4 + c - 5) * 4 + y / 2) * 6 + xx / 2]
                            : x.Span[((n * 6 + c - 9) * 8 + y) * 12 + xx];
        AssertClose(expected, result.Span, Tolerance, "concat");

        var sum = new GraphBuilder(4);
        int small = sum.Add(OpKind.Conv, [0], ConvInts(3, 3, 2, 2, pt: 1, pl: 1, pb: 1, pr: 1), [0, 0], Weight([6, 6, 3, 3], 7), Weight([6], 8));
        sum.Add(OpKind.Add, [0, small], [2, 2]);
        var added = sum.Run(x, Kernel(kernel));
        var smallExpected = Conv(x, sum.Model.Weights[0].Data, sum.Model.Weights[1].Data, 6, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, ActivationKind.None, 0, 0, out _);
        var addExpected = new double[x.Length];
        for (int p = 0; p < 12; p++)
            for (int y = 0; y < 8; y++)
                for (int xx = 0; xx < 12; xx++)
                    addExpected[(p * 8 + y) * 12 + xx] = x.Span[(p * 8 + y) * 12 + xx] + smallExpected[(p * 4 + y / 2) * 6 + xx / 2];
        AssertClose(addExpected, added.Span, Tolerance, "upsampled add");
    }

    [Fact]
    public void StemPoolWritesOverItsInputInsideAConcatenation()
    {
        // The detector's stem: a 2x2, stride 1 maximum with bottom and right padding writes over the convolution it
        // reads, and that memory and a second branch become slices of the concatenation without copies. An upsampled
        // input before them checks that the slices after it start where its enlarged slice ends.
        var x = RandomTensor(31, 1, 3, 10, 36);
        var wa = Weight([4, 3, 1, 1], 1);
        var ba = Weight([4], 2);
        var wb = Weight([2, 4, 1, 1], 3);
        var bb = Weight([2], 4);
        var wc = Weight([4, 2, 1, 1], 5);
        var bc = Weight([4], 6);
        var ws = Weight([3, 3, 3, 3], 7);
        var bs = Weight([3], 8);
        var wf = Weight([5, 11, 1, 1], 9);
        var bf = Weight([5], 10);
        var graph = new GraphBuilder(4);
        int a = graph.Add(OpKind.Conv, [0], ConvInts(1, 1), [0, 0], wa, ba);
        int b = graph.Add(OpKind.Conv, [a], ConvInts(1, 1), [0, 0], wb, bb);
        int c = graph.Add(OpKind.Conv, [b], ConvInts(1, 1), [0, 0], wc, bc);
        int d = graph.Add(OpKind.MaxPool, [a], [2, 2, 1, 1, 0, 0, 1, 1, 0]);
        int small = graph.Add(OpKind.Conv, [0], ConvInts(3, 3, 2, 2, pt: 1, pl: 1, pb: 1, pr: 1), [0, 0], ws, bs);
        int concat = graph.Add(OpKind.Concat, [small, d, c], [1, 2, 2, 1, 1, 1, 1]);
        graph.Add(OpKind.Conv, [concat], ConvInts(1, 1), [0, 0], wf, bf);

        static Tensor ToTensor(double[] values, params int[] shape) => Tensor.FromBuffer(values.Select(v => (float)v).ToArray(), shape);
        var ea = ToTensor(Conv(x, wa.Data, ba.Data, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, ActivationKind.None, 0, 0, out _), 1, 4, 10, 36);
        var eb = ToTensor(Conv(ea, wb.Data, bb.Data, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, ActivationKind.None, 0, 0, out _), 1, 2, 10, 36);
        var ec = Conv(eb, wc.Data, bc.Data, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, ActivationKind.None, 0, 0, out _);
        var ed = Pool(ea, true, 2, 2, 1, 1, 0, 0, 1, 1, false);
        var es = Conv(x, ws.Data, bs.Data, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, ActivationKind.None, 0, 0, out _);
        var joined = new double[11 * 360];
        for (int ch = 0; ch < 11; ch++)
            for (int y = 0; y < 10; y++)
                for (int xx = 0; xx < 36; xx++)
                    joined[(ch * 10 + y) * 36 + xx] = ch < 3 ? es[(ch * 5 + y / 2) * 18 + xx / 2]
                        : ch < 7 ? ed[((ch - 3) * 10 + y) * 36 + xx]
                        : ec[((ch - 7) * 10 + y) * 36 + xx];
        var expected = Conv(ToTensor(joined, 1, 11, 10, 36), wf.Data, bf.Data, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, ActivationKind.None, 0, 0, out _);

        var one = graph.Run(x, degree: 1);
        AssertClose(expected, one.Span, Tolerance, "stem");
        Assert.True(one.Span.SequenceEqual(graph.Run(x, degree: 3).Span), "threads");
    }

    [Fact]
    public void ShapeOperatorsMoveValues()
    {
        var x = RandomTensor(2, 2, 3, 4, 5);
        var graph = new GraphBuilder(4);
        int t = graph.Add(OpKind.Transpose, [0], [0, 2, 3, 1]);
        int r = graph.Add(OpKind.Reshape, [t], [0, -1, 3]);
        graph.Add(OpKind.Mul, [r, r], [1, 1]);
        var result = graph.Run(x);
        Assert.Equal([2, 20, 3], result.Shape);
        var expected = new double[result.Length];
        for (int n = 0; n < 2; n++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < 4; y++)
                    for (int xx = 0; xx < 5; xx++)
                    {
                        double v = x.Span[((n * 3 + c) * 4 + y) * 5 + xx];
                        expected[((n * 4 + y) * 5 + xx) * 3 + c] = v * v;
                    }
        AssertClose(expected, result.Span, 1e-7, "transpose reshape mul");

        var up = new GraphBuilder(4);
        up.Add(OpKind.Upsample, [0], [2, 3]);
        var upsampled = up.Run(x);
        Assert.Equal([2, 3, 8, 15], upsampled.Shape);
        Assert.Equal(x[1, 2, 3, 4], upsampled[1, 2, 7, 14]);
        Assert.Equal(x[0, 1, 2, 0], upsampled[0, 1, 5, 2]);
    }

    [Fact]
    public void BroadcastingFollowsNumpyRules()
    {
        var x = RandomTensor(4, 2, 3, 4, 5);
        var row = new GraphBuilder(4);
        // A [1, 3, 1, 1] operand made from the input by global pooling, as the classifier's channel scaling does.
        int pooled = row.Add(OpKind.GlobalAveragePool, [0], []);
        row.Add(OpKind.Mul, [0, pooled], [1, 1]);
        var result = row.Run(x);
        var expected = new double[x.Length];
        for (int p = 0; p < 6; p++)
        {
            double mean = 0;
            for (int i = 0; i < 20; i++) mean += x.Span[p * 20 + i];
            mean /= 20;
            for (int i = 0; i < 20; i++) expected[p * 20 + i] = x.Span[p * 20 + i] * mean;
        }
        AssertClose(expected, result.Span, Tolerance, "broadcast mul");
    }
}
