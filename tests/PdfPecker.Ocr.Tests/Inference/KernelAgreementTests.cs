using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;
using static PdfPecker.Ocr.Tests.Inference.TestModels;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// Every matrix kernel the processor runs against the portable one, bit for bit. Each value of a product is one chain of
/// multiply-adds over k from the same seed, so kernels that fuse them as the portable kernel does on this processor (all
/// of them where it has FMA or NEON) must give the same bits whatever their tiles; otherwise a page would read
/// differently on a processor that picks another kernel. On Arm64 this is what holds the NEON kernel to the portable
/// one; on x64 the NEON kernel is not in <see cref="KernelTests.Kernels"/>, and the AVX kernels are held to the same
/// rule.
/// </summary>
public sealed unsafe class KernelAgreementTests
{
    private static GemmKernel Kernel(string name) => GemmKernel.Supported.Single(k => k.Name == name);

    // The portable kernel fuses its multiply-adds where the processor can (see GemmKernel.MultiplyAdd).
    private static bool Fused => Fma.IsSupported || AdvSimd.IsSupported;

    private static float MultiplyAdd(float a, float b, float c) => Fused ? MathF.FusedMultiplyAdd(a, b, c) : (a * b) + c;

    private static readonly ActivationKind[] AllActivations = Enum.GetValues<ActivationKind>();

    /// <summary>Shapes that leave every remainder of rows and columns for tiles of 6, 8 and 12 rows and 8 to 32 columns.</summary>
    private static IEnumerable<(int Rows, int K, int Cols)> Shapes()
    {
        var random = new Random(17);
        int[] ks = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 27, 31, 48, 96, 100];
        foreach (int rows in new[] { 1, 5, 6, 7, 8, 9, 11, 12, 13, 24, 25, 48 })
            foreach (int cols in new[] { 1, 3, 4, 7, 8, 11, 12, 13, 16, 23, 24, 25, 31, 32, 33, 47, 96, 100 })
                yield return (rows, ks[random.Next(ks.Length)], cols);
    }

    private static void AssertSameBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(expected[i]) != BitConverter.SingleToUInt32Bits(actual[i]))
                Assert.Fail($"{what}: value {i} is {actual[i]:R}, expected {expected[i]:R}");
        }
    }

    /// <summary>C = A x B through <see cref="GemmKernel.Multiply"/> with one kernel, C laid out with a leading dimension wider than its columns.</summary>
    private static float[] Multiply(GemmKernel gemm, float[] a, float[] b, int rows, int k, int cols, int biasMode, float[] bias, ActivationKind act, float[]? residual, nint ldc)
    {
        float[] packedA = gemm.PackA(a, rows, k);
        float[] packedB = gemm.PackB(b, k, cols);
        float[] rowBias = gemm.PadBias(bias, rows);
        float[] colBias = gemm.PadBias(bias, cols);
        var c = new float[rows * ldc];
        fixed (float* pa = packedA, pb = packedB, pc = c, rb = rowBias, cb = colBias, res = residual)
        {
            var epilogue = new Epilogue(act, 0.2f, 0.5f, res);
            gemm.Multiply(k, pa, rows, pb, cols, pc, ldc, biasMode == 1 ? rb : null, biasMode == 2 ? cb : null, epilogue);
        }

        return c;
    }

    [Theory]
    [MemberData(nameof(KernelTests.Kernels), MemberType = typeof(KernelTests))]
    public void ProductsHaveThePortableKernelsBits(string kernel)
    {
        var gemm = Kernel(kernel);
        int shape = 0;
        foreach (var (rows, k, cols) in Shapes())
        {
            shape++;
            float[] a = Random(rows * k, shape, 2f), b = Random(k * cols, shape + 1000, 2f);
            float[] bias = Random(Math.Max(rows, cols), shape + 2000);
            nint ldc = cols + 5;
            int biasMode = shape % 3;
            var act = AllActivations[shape % AllActivations.Length];
            float[]? residual = shape % 2 == 0 ? Random(rows * (int)ldc, shape + 3000) : null;

            // The reference: one chain of multiply-adds per value, in the order of k, then the activation and the
            // residual, value by value.
            var expected = new float[rows * ldc];
            for (int r = 0; r < rows; r++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = biasMode == 1 ? bias[r] : biasMode == 2 ? bias[j] : 0f;
                    for (int kk = 0; kk < k; kk++) sum = MultiplyAdd(a[(r * k) + kk], b[(kk * cols) + j], sum);
                    if (act != ActivationKind.None) sum = VectorMath.Activate(sum, act, 0.2f, 0.5f);
                    if (residual != null) sum += residual[(r * ldc) + j];
                    expected[(r * ldc) + j] = sum;
                }
            }

            string what = $"{kernel} {rows}x{k}x{cols} bias {biasMode} {act} residual {residual != null}";
            var portable = Multiply(GemmKernel.Portable, a, b, rows, k, cols, biasMode, bias, act, residual, ldc);
            var result = Multiply(gemm, a, b, rows, k, cols, biasMode, bias, act, residual, ldc);
            AssertSameBits(expected, portable, "portable " + what);
            AssertSameBits(portable, result, what);
        }
    }

    [Theory]
    [MemberData(nameof(KernelTests.Kernels), MemberType = typeof(KernelTests))]
    public void ProductsOfSpecialValuesHaveThePortableKernelsBits(string kernel)
    {
        // NaNs with different payloads (quiet and signalling, both signs), infinities, zeros of both signs, subnormals
        // and values whose products overflow, in A, B and the bias. Which NaN a multiply-add passes on depends on the
        // operand it is in, so this holds a kernel to the portable one's choice of operands, not only to its sums.
        float[] special =
        [
            BitConverter.UInt32BitsToSingle(0x7FC00001), BitConverter.UInt32BitsToSingle(0xFFC12345), BitConverter.UInt32BitsToSingle(0x7F800001),
            BitConverter.UInt32BitsToSingle(0x7FBFFFFF), float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, 1e-40f, -1e-40f, 3e38f, -3e38f,
        ];
        var gemm = Kernel(kernel);
        var random = new Random(31);
        float Pick() => random.Next(3) == 0 ? special[random.Next(special.Length)] : (float)((random.NextDouble() * 4) - 2);
        foreach (var (rows, k, cols) in new[] { (8, 5, 12), (13, 9, 25), (25, 17, 37), (6, 3, 8), (17, 1, 13) })
        {
            float[] a = Enumerable.Range(0, rows * k).Select(_ => Pick()).ToArray();
            float[] b = Enumerable.Range(0, k * cols).Select(_ => Pick()).ToArray();
            float[] bias = Enumerable.Range(0, Math.Max(rows, cols)).Select(_ => Pick()).ToArray();
            for (int biasMode = 0; biasMode < 3; biasMode++)
            {
                var portable = Multiply(GemmKernel.Portable, a, b, rows, k, cols, biasMode, bias, ActivationKind.None, null, cols);
                var result = Multiply(gemm, a, b, rows, k, cols, biasMode, bias, ActivationKind.None, null, cols);
                AssertSameBits(portable, result, $"{kernel} special values {rows}x{k}x{cols} bias {biasMode}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(KernelTests.Kernels), MemberType = typeof(KernelTests))]
    public void ProductsReadInPlaceHaveThePortableKernelsBits(string kernel)
    {
        // MultiplyDirect reads one panel of B at an offset per step of k; panels of every width up to NR and partial
        // blocks of rows go through the edge buffer.
        var gemm = Kernel(kernel);
        var random = new Random(23);
        int shape = 0;
        foreach (int rows in new[] { 1, 5, 6, 7, 8, 12, 13, 26, 48 })
        {
            foreach (int k in new[] { 1, 3, 9, 27, 50 })
            {
                shape++;
                const int Source = 4096;
                float[] a = Random(rows * k, shape, 2f), b = Random(Source, shape + 1000, 2f);
                float[] bias = Random(rows, shape + 2000);
                var offsets = new int[k];
                for (int kk = 0; kk < k; kk++) offsets[kk] = random.Next(Source - 64);
                bool withBias = shape % 2 == 0;
                foreach (int cols in Enumerable.Range(1, 32))
                {
                    nint ldc = 40;
                    var expected = new float[rows * ldc];
                    for (int r = 0; r < rows; r++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            float sum = withBias ? bias[r] : 0f;
                            for (int kk = 0; kk < k; kk++) sum = MultiplyAdd(a[(r * k) + kk], b[offsets[kk] + j], sum);
                            expected[(r * ldc) + j] = sum;
                        }
                    }

                    // A width a kernel cannot take in one panel goes through it in panels, as the convolutions do.
                    var result = new float[rows * ldc];
                    float[] packedA = gemm.PackA(a, rows, k);
                    float[] padded = gemm.PadBias(bias, rows);
                    var edge = new float[gemm.MR * gemm.NR];
                    fixed (float* pa = packedA, pb = b, pc = result, rb = padded, pe = edge)
                    fixed (int* po = offsets)
                    {
                        for (int j0 = 0; j0 < cols; j0 += gemm.NR)
                            gemm.MultiplyDirect(k, pa, rows, pb + j0, po, pc + j0, ldc, withBias ? rb : null, pe, Math.Min(gemm.NR, cols - j0));
                    }

                    AssertSameBits(expected, result, $"{kernel} direct {rows}x{k}x{cols} bias {withBias}");
                }
            }
        }
    }

    public static TheoryData<string, string> LayerCases
    {
        get
        {
            var data = new TheoryData<string, string>();
            string[] cases = ["pointwise-residual", "pointwise-wide", "pointwise-stride2", "stem3x3s2", "stem-wide", "3x3", "3x3-residual", "2x2-same", "1x7", "3x3-stride2x1", "large-k", "linear", "linear-residual", "head"];
            foreach (var kernel in GemmKernel.Supported)
                foreach (string c in cases) data.Add(kernel.Name, c);
            return data;
        }
    }

    [Theory]
    [MemberData(nameof(LayerCases))]
    public void LayersHaveThePortableKernelsBits(string kernel, string name)
    {
        // The layers tile, pack and read panels in place by the kernel's MR and NR, and apply their epilogues per
        // block of MR rows or per tile, so each case is run on enough threads to split it, and compared bit for bit.
        var graph = new GraphBuilder(4);
        Tensor x;
        if (name is "linear" or "linear-residual" or "head")
        {
            graph = new GraphBuilder(3);
            int features = name == "head" ? 120 : 61;
            int classes = name == "head" ? 1031 : 61;
            x = RandomTensor(features, 3, 37, features);
            bool residual = name == "linear-residual";
            int l = graph.Add(OpKind.Linear, residual ? [0, 0] : [0], [(int)(name == "head" ? ActivationKind.None : ActivationKind.Gelu), residual ? 1 : 0], [0, 0],
                Weight([features, classes], classes, 2f), Weight([classes], classes + 1));
            if (name == "head") graph.Add(OpKind.Softmax, [l], []);
        }
        else
        {
            // (cin, cout, h, w, kh, kw, sh, sw, pad, act, residual)
            var (cin, cout, h, w, kh, kw, sh, sw, pad, act, residual) = name switch
            {
                "pointwise-residual" => (24, 24, 9, 37, 1, 1, 1, 1, 0, ActivationKind.None, true),
                "pointwise-wide" => (19, 53, 7, 131, 1, 1, 1, 1, 0, ActivationKind.HardSwish, false),
                "pointwise-stride2" => (8, 12, 9, 14, 1, 1, 2, 2, 0, ActivationKind.Relu, false),
                "stem3x3s2" => (3, 16, 17, 23, 3, 3, 2, 2, 1, ActivationKind.Relu, false),
                // Output rows of 300 pixels, read in place in split halves, with a partial block of 22 channels.
                "stem-wide" => (3, 22, 24, 600, 3, 3, 2, 2, 1, ActivationKind.Relu, false),
                "3x3" => (10, 7, 11, 37, 3, 3, 1, 1, 1, ActivationKind.Sigmoid, false),
                "3x3-residual" => (13, 13, 11, 70, 3, 3, 1, 1, 1, ActivationKind.Gelu, true),
                "2x2-same" => (5, 11, 10, 15, 2, 2, 1, 1, 0, ActivationKind.Swish, false),
                "1x7" => (4, 6, 5, 40, 1, 7, 1, 1, 0, ActivationKind.HardSigmoid, false),
                "3x3-stride2x1" => (7, 5, 12, 21, 3, 3, 2, 1, 1, ActivationKind.None, false),
                _ => (96, 40, 7, 30, 3, 3, 1, 1, 1, ActivationKind.Relu, false),
            };
            x = RandomTensor(cin * 7 + h, 2, cin, h, w);
            int pl = kw == 7 ? 3 : pad;
            int[] ints = ConvInts(kh, kw, sh, sw, 1, 1, pad, pl, name == "2x2-same" ? 1 : pad, name == "2x2-same" ? 1 : pl, act: act, residual: residual);
            graph.Add(OpKind.Conv, residual ? [0, 0] : [0], ints, [0.2f, 0.5f], Weight([cout, cin, kh, kw], cout, 2f), Weight([cout], cout + 3));
        }

        var expected = OcrModel.FromDefinition(graph.Model, GemmKernel.Portable);
        var model = OcrModel.FromDefinition(graph.Model, Kernel(kernel));
        float[] reference;
        using (var context = new InferenceContext(1)) reference = expected.Run(x, context).Span.ToArray();
        foreach (int degree in new[] { 1, 3 })
        {
            using var context = new InferenceContext(degree);
            AssertSameBits(reference, model.Run(x, context).Span, $"{kernel} {name} on {degree} threads");
        }

        if (name == "head")
        {
            float[] best;
            using (var context = new InferenceContext(1)) best = expected.WithArgMaxOutput().Run(x, context).Span.ToArray();
            using var other = new InferenceContext(3);
            AssertSameBits(best, model.WithArgMaxOutput().Run(x, other).Span, $"{kernel} argmax head");
        }
    }

    public static TheoryData<string, string> ModelCases
    {
        get
        {
            var data = new TheoryData<string, string>();
            foreach (var kernel in GemmKernel.Supported)
                foreach (string fixture in new[] { "det_96x128", "rec_2x48x160", "cls_2x48x192" }) data.Add(kernel.Name, fixture);
            return data;
        }
    }

    [Theory]
    [MemberData(nameof(ModelCases))]
    public void BundledModelsHaveThePortableKernelsBits(string kernel, string fixture)
    {
        string name = fixture[..3];
        var input = ReferenceFixture.Load(fixture).Input();
        OcrModel Build(GemmKernel gemm) => name == "rec" ? ModelStore.LoadRecognition(gemm).WithFusedStem() : name == "det" ? ModelStore.Load(name, gemm).WithFusedStem() : ModelStore.Load(name, gemm);
        float[] expected;
        using (var context = new InferenceContext(2)) expected = Build(GemmKernel.Portable).Run(input, context).Span.ToArray();
        using var other = new InferenceContext(3);
        AssertSameBits(expected, Build(Kernel(kernel)).Run(input, other).Span, $"{kernel} {fixture}");
    }
}
