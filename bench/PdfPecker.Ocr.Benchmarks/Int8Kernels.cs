#if OCR_INTERNALS
using System.Diagnostics;
using System.Globalization;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace OcrBench;

/// <summary>
/// The float and the experimental 8-bit products of a pointwise convolution's shapes on one thread, split into packing
/// (or quantising) the activations and multiplying, in GMAC/s:
///   dotnet run -c Release -p:OcrInternals=true -- int8-kernels
/// </summary>
static unsafe class Int8Kernels
{
    public static void Run()
    {
        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine($"GEMM {GemmKernel.Best.Name}, int8 supported {Int8Gemm.IsSupported}");
        Console.WriteLine($"{"M",5} {"K",5} {"N",5} {"fp32 pack",10} {"fp32 mul",10} {"GMAC/s",7} {"i8 quant",10} {"i8 mul",10} {"GMAC/s",7} {"speed-up",8}");
        foreach (var (m, k, n) in new[] { (48, 24, 1024), (96, 48, 1024), (48, 96, 1024), (192, 96, 1024), (96, 192, 1024), (384, 192, 1024), (192, 384, 1024), (768, 384, 1024), (384, 768, 1024) })
        {
            var gemm = GemmKernel.Best;
            var rng = new Random(1);
            float[] w = new float[m * k], x = new float[k * n];
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 1);
            float[] packedA = gemm.PackA(w, m, k);
            float[] bias = gemm.PadBias(new float[m], m);
            var q = Int8Gemm.QuantizeWeights(w, m, k);
            float[] qbias = new float[m + 64];
            float[] packedB = new float[gemm.PackedBSize(k, n) + 64];
            byte[] packedQ = new byte[Int8Gemm.PackedBytes(k, n) + 64];
            float[] c = new float[m * n];
            int reps = Math.Max(3, (int)(2e9 / ((double)m * k * n)));
            fixed (float* fa = packedA, fb = packedB, fx = x, fc = c, fbias = bias, fq = qbias)
            fixed (byte* fqb = packedQ)
            {
                float* pa = fa, pb = fb, px = fx, pc = fc, pbias = fbias, pq = fq;
                byte* pqb = fqb;
                double fpPack = Time(reps, () => gemm.PackBRows(px, n, k, n, pb));
                double fpMul = Time(reps, () => gemm.Multiply(k, pa, m, pb, n, pc, n, pbias, null));
                double qPack = Time(reps, () => Int8Gemm.QuantizeColumns(px, n, k, n, pqb));
                double qMul = Time(reps, () => Int8Gemm.Multiply(q, 0, m, pqb, n, pc, n, pq));
                double macs = (double)m * k * n;
                Console.WriteLine(string.Create(inv, $"{m,5} {k,5} {n,5} {fpPack * 1e6,8:F1}us {fpMul * 1e6,8:F1}us {macs / fpMul / 1e9,7:F1} {qPack * 1e6,8:F1}us {qMul * 1e6,8:F1}us {macs / qMul / 1e9,7:F1} {(fpPack + fpMul) / (qPack + qMul),8:F2}"));
            }
        }
    }

    /// <summary>Single 1x1 convolution layers of the recogniser's shapes, float and 8-bit in turns, on one thread.</summary>
    public static void Layers()
    {
        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine();
        Console.WriteLine("Whole layers on one thread ([N, Cin, H, W] in, activation), float and 8-bit in turns, best of 5:");
        Console.WriteLine($"{"layer",-36} {"float",10} {"8-bit",10} {"speed-up",8}");
        foreach (var (cin, cout, h, w, act) in new[] { (192, 384, 6, 1200, ActivationKind.Gelu), (384, 192, 6, 1200, ActivationKind.None), (384, 768, 3, 1200, ActivationKind.Gelu), (768, 384, 3, 1200, ActivationKind.None), (96, 192, 12, 1200, ActivationKind.Gelu), (384, 768, 3, 1200, ActivationKind.None), (48, 96, 384, 272, ActivationKind.Gelu), (24, 48, 384, 272, ActivationKind.Relu) })
        {
            var def = new ModelDefinition { Name = "layer", InputShape = [-1, cin, -1, -1] };
            var rng = new Random(cin + cout);
            float[] weights = new float[cout * cin], bias = new float[cout];
            for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
            def.Weights.Add(new WeightDefinition([cout, cin, 1, 1], weights, WeightStorage.Float32));
            def.Weights.Add(new WeightDefinition([cout], bias, WeightStorage.Float32));
            def.Nodes.Add(new NodeDefinition(OpKind.Conv, [0], 1, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, (int)act, 0], [1 / 6f, 0.5f], [0, 1]));
            var fp = OcrModel.FromDefinition(def);
            var q = OcrModel.FromDefinition(def, int8Pointwise: true);
            int n = 1;
            float[] x = new float[n * cin * h * w];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 1);
            using var context = new InferenceContext(1);
            int[] shape = [n, cin, h, w];
            double bestF = double.MaxValue, bestQ = double.MaxValue;
            for (int round = 0; round < 6; round++)
            {
                long s0 = Stopwatch.GetTimestamp();
                fp.Run(x, shape, context, out _);
                double tf = Stopwatch.GetElapsedTime(s0).TotalMilliseconds;
                s0 = Stopwatch.GetTimestamp();
                q.Run(x, shape, context, out _);
                double tq = Stopwatch.GetElapsedTime(s0).TotalMilliseconds;
                if (round == 0) continue;
                bestF = Math.Min(bestF, tf);
                bestQ = Math.Min(bestQ, tq);
            }

            Console.WriteLine(string.Create(inv, $"{$"{cin}->{cout} [{n},{cin},{h},{w}] {act}",-36} {bestF,8:F2}ms {bestQ,8:F2}ms {bestF / bestQ,8:F2}"));
        }
    }

    static double Time(int reps, Action action)
    {
        for (int i = 0; i < 3; i++) action();
        var best = double.MaxValue;
        for (int round = 0; round < 5; round++)
        {
            long s = Stopwatch.GetTimestamp();
            for (int i = 0; i < reps; i++) action();
            best = Math.Min(best, Stopwatch.GetElapsedTime(s).TotalSeconds / reps);
        }

        return best;
    }
}
#endif
