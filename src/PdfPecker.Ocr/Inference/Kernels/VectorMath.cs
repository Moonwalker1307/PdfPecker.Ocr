using System.Numerics;
using System.Runtime.CompilerServices;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Kernels;

/// <summary>
/// Vectorised activations, exponentials and row reductions on <see cref="Vector{T}"/>, which is 256 bits on x64 with
/// AVX2 and 128 bits on Arm64. A row's tail goes through the same vector code on a zero-padded copy, so a value gets the
/// same result whichever position in a row it has; that keeps results independent of how work is split into tiles.
/// </summary>
internal static unsafe class VectorMath
{
    private static readonly int Width = Vector<float>.Count;

    /// <summary>Applies an activation to length values in place.</summary>
    public static void Activate(float* data, int length, ActivationKind kind, float alpha, float beta)
    {
        if (kind == ActivationKind.None || length <= 0) return;
        int i = 0;
        for (; i <= length - Width; i += Width)
            Store(Activate(Load(data + i), kind, alpha, beta), data + i);
        if (i < length)
        {
            float* tail = stackalloc float[Width];
            int rest = length - i;
            new Span<float>(tail, Width).Clear();
            new Span<float>(data + i, rest).CopyTo(new Span<float>(tail, rest));
            Store(Activate(Load(tail), kind, alpha, beta), tail);
            new Span<float>(tail, rest).CopyTo(new Span<float>(data + i, rest));
        }
    }

    /// <summary>The activation of one vector.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Activate(Vector<float> x, ActivationKind kind, float alpha, float beta) => kind switch
    {
        ActivationKind.Relu => Vector.Max(x, Vector<float>.Zero),
        ActivationKind.Sigmoid => Sigmoid(x),
        ActivationKind.HardSigmoid => HardSigmoid(x, alpha, beta),
        ActivationKind.HardSwish => x * HardSigmoid(x, alpha, beta),
        ActivationKind.Swish => x * Sigmoid(x),
        ActivationKind.Gelu => Gelu(x),
        _ => x,
    };

    /// <summary>The activation of one value, through the vector code so scalar and vector results agree.</summary>
    public static float Activate(float x, ActivationKind kind, float alpha, float beta) =>
        kind == ActivationKind.None ? x : Activate(new Vector<float>(x), kind, alpha, beta)[0];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Load(float* p) => Unsafe.ReadUnaligned<Vector<float>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(Vector<float> v, float* p) => Unsafe.WriteUnaligned(p, v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> HardSigmoid(Vector<float> x, float alpha, float beta) =>
        Vector.Min(Vector.Max(x * new Vector<float>(alpha) + new Vector<float>(beta), Vector<float>.Zero), Vector<float>.One);

    /// <summary>1 / (1 + exp(-x)).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Sigmoid(Vector<float> x) => Vector<float>.One / (Vector<float>.One + Exp(-x));

    /// <summary>0.5 x (1 + erf(x / sqrt 2)).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Gelu(Vector<float> x) =>
        x * (Vector<float>.One + Erf(x * new Vector<float>(0.70710678118654752f))) * new Vector<float>(0.5f);

    /// <summary>
    /// exp(x) with the Cephes single-precision polynomial: about one unit in the last place over the clamped range.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Exp(Vector<float> x)
    {
        x = Vector.Min(Vector.Max(x, new Vector<float>(-87.3365447505531f)), new Vector<float>(88.3762626647949f));
        var fx = Vector.Floor(x * new Vector<float>(1.44269504088896341f) + new Vector<float>(0.5f));
        x -= fx * new Vector<float>(0.693359375f);
        x -= fx * new Vector<float>(-2.12194440e-4f);
        var z = x * x;
        var y = new Vector<float>(1.9875691500E-4f);
        y = y * x + new Vector<float>(1.3981999507E-3f);
        y = y * x + new Vector<float>(8.3334519073E-3f);
        y = y * x + new Vector<float>(4.1665795894E-2f);
        y = y * x + new Vector<float>(1.6666665459E-1f);
        y = y * x + new Vector<float>(5.0000001201E-1f);
        y = y * z + x + Vector<float>.One;
        var pow2 = Vector.ShiftLeft(Vector.ConvertToInt32(fx) + new Vector<int>(127), 23);
        return y * Vector.AsVectorSingle(pow2);
    }

    /// <summary>
    /// erf(x) by Abramowitz and Stegun 7.1.26, with an absolute error below 1.5e-7, well inside the models' tolerance.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Erf(Vector<float> x)
    {
        var ax = Vector.Abs(x);
        var t = Vector<float>.One / (Vector<float>.One + new Vector<float>(0.3275911f) * ax);
        var p = new Vector<float>(1.061405429f);
        p = p * t + new Vector<float>(-1.453152027f);
        p = p * t + new Vector<float>(1.421413741f);
        p = p * t + new Vector<float>(-0.284496736f);
        p = p * t + new Vector<float>(0.254829592f);
        p *= t;
        var e = Vector<float>.One - p * Exp(-(ax * ax));
        return Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero), -e, e);
    }

    /// <summary>The sum of length values, lane by lane and then across lanes, in a fixed order.</summary>
    public static float Sum(float* data, int length)
    {
        var acc = Vector<float>.Zero;
        int i = 0;
        for (; i <= length - Width; i += Width) acc += Load(data + i);
        float sum = Vector.Sum(acc);
        for (; i < length; i++) sum += data[i];
        return sum;
    }

    /// <summary>The largest of length values (length at least 1).</summary>
    public static float Max(float* data, int length)
    {
        int i = 0;
        float max = float.NegativeInfinity;
        if (length >= Width)
        {
            var acc = Load(data);
            for (i = Width; i <= length - Width; i += Width) acc = Vector.Max(acc, Load(data + i));
            for (int j = 0; j < Width; j++) max = Math.Max(max, acc[j]);
        }
        for (; i < length; i++) max = Math.Max(max, data[i]);
        return max;
    }

    /// <summary>Softmax of one row in place: exp(x - max) / sum.</summary>
    public static void Softmax(float* row, int length)
    {
        float max = Max(row, length);
        var vmax = new Vector<float>(max);
        var acc = Vector<float>.Zero;
        int i = 0;
        for (; i <= length - Width; i += Width)
        {
            var e = Exp(Load(row + i) - vmax);
            Store(e, row + i);
            acc += e;
        }
        float sum = Vector.Sum(acc);
        for (; i < length; i++)
        {
            float e = Exp(new Vector<float>(row[i] - max))[0];
            row[i] = e;
            sum += e;
        }
        Scale(row, length, 1f / sum);
    }

    /// <summary>Multiplies length values in place by a factor.</summary>
    public static void Scale(float* data, int length, float factor)
    {
        var f = new Vector<float>(factor);
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(data + i) * f, data + i);
        for (; i < length; i++) data[i] *= factor;
    }

    /// <summary>dst[i] += src[i].</summary>
    public static void AddInPlace(float* dst, float* src, int length)
    {
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(dst + i) + Load(src + i), dst + i);
        for (; i < length; i++) dst[i] += src[i];
    }

    /// <summary>dst[i] = a[i] + b[i].</summary>
    public static void Add(float* a, float* b, float* dst, int length)
    {
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) + Load(b + i), dst + i);
        for (; i < length; i++) dst[i] = a[i] + b[i];
    }

    /// <summary>dst[i] = a[i] * b[i].</summary>
    public static void Multiply(float* a, float* b, float* dst, int length)
    {
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) * Load(b + i), dst + i);
        for (; i < length; i++) dst[i] = a[i] * b[i];
    }

    /// <summary>dst[i] = a[i] + s.</summary>
    public static void AddScalar(float* a, float s, float* dst, int length)
    {
        var v = new Vector<float>(s);
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) + v, dst + i);
        for (; i < length; i++) dst[i] = a[i] + s;
    }

    /// <summary>
    /// dst[i] = a[i] + a[i] * s, rounding the product first as the original graph's separate Mul and Add do. Each value
    /// is read before it is written, so dst may be a.
    /// </summary>
    public static void AddScaled(float* a, float s, float* dst, int length)
    {
        var v = new Vector<float>(s);
        int i = 0;
        for (; i <= length - Width; i += Width)
        {
            var x = Load(a + i);
            var product = x * v;
            Store(x + product, dst + i);
        }
        for (; i < length; i++)
        {
            float product = a[i] * s;
            dst[i] = a[i] + product;
        }
    }

    /// <summary>dst[i] = a[i] * s.</summary>
    public static void MultiplyScalar(float* a, float s, float* dst, int length)
    {
        var v = new Vector<float>(s);
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) * v, dst + i);
        for (; i < length; i++) dst[i] = a[i] * s;
    }
}
