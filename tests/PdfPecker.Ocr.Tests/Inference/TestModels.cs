using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>Builds small models of one or a few layers, and naive reference implementations in double precision.</summary>
internal static class TestModels
{
    public static float[] Random(int count, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return data;
    }

    public static Tensor RandomTensor(int seed, params int[] shape) => Tensor.FromBuffer(Random(Tensor.ElementCount(shape), seed), shape);


    /// <summary>Runs a model definition on an input with a fresh context, and returns a copy of the output.</summary>
    public static Tensor Run(ModelDefinition model, Tensor input, GemmKernel? gemm = null, int degree = 2)
    {
        var loaded = OcrModel.FromDefinition(model, gemm);
        using var context = new InferenceContext(degree);
        return loaded.Run(input, context).Clone();
    }

    /// <summary>The integer attributes of a convolution node, in the order <see cref="OpKind.Conv"/> lists them.</summary>
    public static int[] ConvInts(int kh, int kw, int sh = 1, int sw = 1, int dh = 1, int dw = 1, int pt = 0, int pl = 0, int pb = 0, int pr = 0,
        int group = 1, ActivationKind act = ActivationKind.None, bool residual = false) =>
        [kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, group, (int)act, residual ? 1 : 0];

    public static WeightDefinition Weight(int[] shape, int seed, float scale = 0.5f) =>
        new(shape, Random(Tensor.ElementCount(shape), seed, scale), WeightStorage.Float32);

    public static double Activate(double x, ActivationKind kind, double alpha, double beta) => kind switch
    {
        ActivationKind.Relu => Math.Max(x, 0),
        ActivationKind.Sigmoid => 1 / (1 + Math.Exp(-x)),
        ActivationKind.HardSigmoid => Math.Clamp(alpha * x + beta, 0, 1),
        ActivationKind.HardSwish => x * Math.Clamp(alpha * x + beta, 0, 1),
        ActivationKind.Swish => x / (1 + Math.Exp(-x)),
        ActivationKind.Gelu => 0.5 * x * (1 + Erf(x / Math.Sqrt(2))),
        _ => x,
    };

    /// <summary>erf in double precision (series for small arguments, continued fraction for large ones).</summary>
    public static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double result;
        if (ax < 2.5)
        {
            double term = ax, sum = ax, x2 = ax * ax;
            for (int n = 1; n < 100; n++)
            {
                term *= -x2 / n;
                double add = term / (2 * n + 1);
                sum += add;
                if (Math.Abs(add) < 1e-17) break;
            }
            result = 2 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            // erfc by Lentz's continued fraction.
            double f = ax, c = ax, d = 0;
            for (int n = 1; n < 200; n++)
            {
                double an = n / 2.0;
                d = ax + an * d;
                d = 1 / d;
                c = ax + an / c;
                double delta = c * d;
                f *= delta;
                if (Math.Abs(delta - 1) < 1e-16) break;
            }
            result = 1 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / f;
        }
        return x < 0 ? -result : result;
    }

    /// <summary>Naive convolution in double precision with ONNX semantics.</summary>
    public static double[] Conv(Tensor x, float[] w, float[] b, int cout, int kh, int kw, int sh, int sw, int dh, int dw,
        int pt, int pl, int pb, int pr, int group, ActivationKind act, float alpha, float beta, out int[] shape, Tensor? residual = null)
    {
        int n = x.Shape[0], cin = x.Shape[1], h = x.Shape[2], wd = x.Shape[3];
        int oh = (h + pt + pb - dh * (kh - 1) - 1) / sh + 1;
        int ow = (wd + pl + pr - dw * (kw - 1) - 1) / sw + 1;
        shape = [n, cout, oh, ow];
        var span = x.Span;
        var result = new double[n * cout * oh * ow];
        int cinPerGroup = cin / group, coutPerGroup = cout / group;
        for (int b0 = 0; b0 < n; b0++)
            for (int co = 0; co < cout; co++)
            {
                int g = co / coutPerGroup;
                for (int y = 0; y < oh; y++)
                    for (int xx = 0; xx < ow; xx++)
                    {
                        double acc = b[co];
                        for (int ci = 0; ci < cinPerGroup; ci++)
                            for (int ky = 0; ky < kh; ky++)
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int iy = y * sh - pt + ky * dh, ix = xx * sw - pl + kx * dw;
                                    if (iy < 0 || iy >= h || ix < 0 || ix >= wd) continue;
                                    acc += (double)w[((co * cinPerGroup + ci) * kh + ky) * kw + kx] * span[((b0 * cin + g * cinPerGroup + ci) * h + iy) * wd + ix];
                                }
                        int o = ((b0 * cout + co) * oh + y) * ow + xx;
                        result[o] = Activate(acc, act, alpha, beta) + (residual is null ? 0 : residual.Span[o]);
                    }
            }
        return result;
    }

    /// <summary>A transposed convolution whose kernel equals its stride, in double precision.</summary>
    public static double[] ConvTranspose(Tensor x, float[] w, float[] b, int cout, int kh, int kw, ActivationKind act, float alpha, float beta)
    {
        int n = x.Shape[0], cin = x.Shape[1], h = x.Shape[2], wd = x.Shape[3];
        int oh = h * kh, ow = wd * kw;
        var span = x.Span;
        var result = new double[n * cout * oh * ow];
        for (int b0 = 0; b0 < n; b0++)
            for (int co = 0; co < cout; co++)
                for (int y = 0; y < oh; y++)
                    for (int xx = 0; xx < ow; xx++)
                    {
                        double acc = b[co];
                        for (int ci = 0; ci < cin; ci++)
                            acc += (double)w[((ci * cout + co) * kh + y % kh) * kw + xx % kw] * span[((b0 * cin + ci) * h + y / kh) * wd + xx / kw];
                        result[((b0 * cout + co) * oh + y) * ow + xx] = Activate(acc, act, alpha, beta);
                    }
        return result;
    }

    /// <summary>Max or average pooling with ONNX semantics, in double precision.</summary>
    public static double[] Pool(Tensor x, bool max, int kh, int kw, int sh, int sw, int pt, int pl, int pb, int pr, bool countIncludePad)
    {
        int planes = x.Shape[0] * x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        int oh = (h + pt + pb - kh) / sh + 1, ow = (w + pl + pr - kw) / sw + 1;
        var span = x.Span;
        var result = new double[planes * oh * ow];
        for (int p = 0; p < planes; p++)
            for (int y = 0; y < oh; y++)
                for (int xx = 0; xx < ow; xx++)
                {
                    double acc = max ? double.NegativeInfinity : 0;
                    int count = 0;
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int iy = y * sh - pt + ky, ix = xx * sw - pl + kx;
                            bool inside = iy >= 0 && iy < h && ix >= 0 && ix < w;
                            // Padding positions inside the padded input count for count_include_pad; positions past the
                            // padding (a window that hangs over the end) never do.
                            if (iy < h + pb && ix < w + pr) count += countIncludePad || inside ? 1 : 0;
                            if (!inside) continue;
                            double v = span[(p * h + iy) * w + ix];
                            acc = max ? Math.Max(acc, v) : acc + v;
                        }
                    result[(p * oh + y) * ow + xx] = max ? acc : acc / count;
                }
        return result;
    }

    /// <summary>x [rows, K] times W [K, N] plus bias, activation and an optional residual, in double precision.</summary>
    public static double[] Linear(Tensor x, float[] w, float[] b, int n, ActivationKind act, float alpha, float beta, Tensor? residual = null)
    {
        int k = x.Shape[^1], rows = x.Length / k;
        var span = x.Span;
        var result = new double[rows * n];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                double acc = b[j];
                for (int i = 0; i < k; i++) acc += (double)span[r * k + i] * w[i * n + j];
                result[r * n + j] = Activate(acc, act, alpha, beta) + (residual is null ? 0 : residual.Span[r * n + j]);
            }
        return result;
    }

    public static double[] LayerNorm(Tensor x, float[] gamma, float[] beta, double epsilon)
    {
        int c = gamma.Length, rows = x.Length / c;
        var span = x.Span;
        var result = new double[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, variance = 0;
            for (int i = 0; i < c; i++) mean += span[r * c + i];
            mean /= c;
            for (int i = 0; i < c; i++) variance += (span[r * c + i] - mean) * (span[r * c + i] - mean);
            variance /= c;
            for (int i = 0; i < c; i++) result[r * c + i] = (span[r * c + i] - mean) / Math.Sqrt(variance + epsilon) * gamma[i] + beta[i];
        }
        return result;
    }

    public static double[] Softmax(ReadOnlySpan<float> x, int c)
    {
        var result = new double[x.Length];
        for (int r = 0; r < x.Length / c; r++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int i = 0; i < c; i++) max = Math.Max(max, x[r * c + i]);
            for (int i = 0; i < c; i++) sum += result[r * c + i] = Math.Exp(x[r * c + i] - max);
            for (int i = 0; i < c; i++) result[r * c + i] /= sum;
        }
        return result;
    }

    /// <summary>Multi-head self-attention on packed q, k, v [N, T, 3C], in double precision.</summary>
    public static double[] Attention(Tensor x, int heads, double scale)
    {
        int n = x.Shape[0], t = x.Shape[1], c = x.Shape[2] / 3, d = c / heads;
        var span = x.Span;
        var result = new double[n * t * c];
        var scores = new double[t];
        for (int b = 0; b < n; b++)
            for (int h = 0; h < heads; h++)
                for (int t1 = 0; t1 < t; t1++)
                {
                    double max = double.NegativeInfinity, sum = 0;
                    for (int t2 = 0; t2 < t; t2++)
                    {
                        double acc = 0;
                        for (int i = 0; i < d; i++)
                            acc += span[(b * t + t1) * 3 * c + h * d + i] * scale * span[(b * t + t2) * 3 * c + c + h * d + i];
                        scores[t2] = acc;
                        max = Math.Max(max, acc);
                    }
                    for (int t2 = 0; t2 < t; t2++) sum += scores[t2] = Math.Exp(scores[t2] - max);
                    for (int i = 0; i < d; i++)
                    {
                        double acc = 0;
                        for (int t2 = 0; t2 < t; t2++) acc += scores[t2] / sum * span[(b * t + t2) * 3 * c + 2 * c + h * d + i];
                        result[(b * t + t1) * c + h * d + i] = acc;
                    }
                }
        return result;
    }

    /// <summary>Squeeze and excitation in double precision.</summary>
    public static double[] SqueezeExcite(Tensor x, float[] w1, float[] b1, float[] w2, float[] b2, ActivationKind act1, ActivationKind act2,
        float alpha2, float beta2, bool addInput)
    {
        int n = x.Shape[0], c = x.Shape[1], plane = x.Shape[2] * x.Shape[3], hidden = b1.Length;
        var span = x.Span;
        var result = new double[x.Length];
        for (int b = 0; b < n; b++)
        {
            var mean = new double[c];
            for (int ch = 0; ch < c; ch++)
            {
                for (int i = 0; i < plane; i++) mean[ch] += span[(b * c + ch) * plane + i];
                mean[ch] /= plane;
            }
            var mid = new double[hidden];
            for (int j = 0; j < hidden; j++)
            {
                double acc = b1[j];
                for (int ch = 0; ch < c; ch++) acc += w1[j * c + ch] * mean[ch];
                mid[j] = Activate(acc, act1, 0, 0);
            }
            for (int ch = 0; ch < c; ch++)
            {
                double acc = b2[ch];
                for (int j = 0; j < hidden; j++) acc += w2[ch * hidden + j] * mid[j];
                double s = Activate(acc, act2, alpha2, beta2);
                for (int i = 0; i < plane; i++)
                {
                    double v = span[(b * c + ch) * plane + i];
                    result[(b * c + ch) * plane + i] = addInput ? v + v * s : v * s;
                }
            }
        }
        return result;
    }

    public static void AssertClose(double[] expected, ReadOnlySpan<float> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1, Math.Abs(expected[i]));
            if (error > worst)
            {
                worst = error;
                at = i;
            }
        }
        Assert.True(worst <= tolerance, $"{what}: error {worst:E3} at {at} (expected {(at >= 0 ? expected[at] : 0)}, got {(at >= 0 ? actual[at] : 0)})");
    }
}

/// <summary>Builds a model definition node by node; each node writes the next value, and value 0 is the input.</summary>
internal sealed class GraphBuilder(int inputRank)
{
    public ModelDefinition Model { get; } = new() { Name = "test", InputShape = Enumerable.Repeat(-1, inputRank).ToArray() };

    /// <summary>Adds a node and returns the value it writes.</summary>
    public int Add(OpKind op, int[] inputs, int[] ints, float[]? floats = null, params WeightDefinition[] weights)
    {
        int first = Model.Weights.Count;
        Model.Weights.AddRange(weights);
        int output = Model.Nodes.Count + 1;
        Model.Nodes.Add(new NodeDefinition(op, inputs, output, ints, floats ?? [], Enumerable.Range(first, weights.Length).ToArray()));
        return output;
    }

    public Tensor Run(Tensor input, GemmKernel? gemm = null, int degree = 2) => TestModels.Run(Model, input, gemm, degree);
}
