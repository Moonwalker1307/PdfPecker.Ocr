using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Kernels;

/// <summary>
/// Vectorised activations, exponentials and row reductions on <see cref="Vector{T}"/>, which is 256 bits on x64 with
/// AVX2 and 128 bits on Arm64. A row's tail goes through the same vector code on a zero-padded copy, so a value gets the
/// same result whichever position in a row it has; that keeps results independent of how work is split into tiles.
/// </summary>
internal static unsafe partial class VectorMath
{
    private static readonly int Width = Vector<float>.Count;

    /// <summary>Applies an activation to length values in place.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Activate(float* data, int length, ActivationKind kind, float alpha, float beta)
    {
        if (kind == ActivationKind.None || length <= 0) return;
        int i = Wide ? ActivateWide(data, length, kind, alpha, beta) : 0;
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
    /// erf(x) as an odd polynomial over an even one in x, on x clamped to [-4, 4] where erf is 1 to single precision
    /// beyond (the rational approximation Eigen and XLA use). Its absolute error is below 5e-7, well inside the models'
    /// tolerance, and it needs neither an exponential nor a branch.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Erf(Vector<float> x)
    {
        x = Vector.Min(Vector.Max(x, new Vector<float>(-4f)), new Vector<float>(4f));
        var x2 = x * x;
        var p = x2 * new Vector<float>(ErfP13) + new Vector<float>(ErfP11);
        p = x2 * p + new Vector<float>(ErfP9);
        p = x2 * p + new Vector<float>(ErfP7);
        p = x2 * p + new Vector<float>(ErfP5);
        p = x2 * p + new Vector<float>(ErfP3);
        p = x2 * p + new Vector<float>(ErfP1);
        p = x * p;
        var q = x2 * new Vector<float>(ErfQ8) + new Vector<float>(ErfQ6);
        q = x2 * q + new Vector<float>(ErfQ4);
        q = x2 * q + new Vector<float>(ErfQ2);
        q = x2 * q + new Vector<float>(ErfQ0);
        return p / q;
    }

    private const float ErfP1 = -1.60960333262415e-02f, ErfP3 = -2.95459980854025e-03f, ErfP5 = -7.34990630326855e-04f,
        ErfP7 = -5.69250639462346e-05f, ErfP9 = -2.10102402082508e-06f, ErfP11 = 2.77068142495902e-08f, ErfP13 = -2.72614225801306e-10f;

    private const float ErfQ0 = -1.42647390514189e-02f, ErfQ2 = -7.37332916720468e-03f, ErfQ4 = -1.68282697438203e-03f,
        ErfQ6 = -2.13374055278905e-04f, ErfQ8 = -1.45660718464996e-05f;

    /// <summary>The sum of length values, lane by lane and then across lanes, in a fixed order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Softmax(float* row, int length)
    {
        float sum = ExpShiftedSum(row, length, Max(row, length), out _);
        Scale(row, length, 1f / sum);
    }

    /// <summary>Softmax of one row in place, for callers that hold the row in a span.</summary>
    public static void Softmax(Span<float> row)
    {
        fixed (float* p = row) Softmax(p, row.Length);
    }

    /// <summary>
    /// The largest probability of a softmax over one row and the first index that has it, exactly as if the row had
    /// gone through <see cref="Softmax(float*, int)"/> and then a search for the first largest value: the same exponentials, the same
    /// sum and the same products. The row is overwritten with the exponentials, and the probabilities are never stored.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int SoftmaxArgMax(float* row, int length, out float probability)
    {
        float scale = 1f / ExpShiftedSum(row, length, Max(row, length), out float largest);

        // Rounding a product to nearest never puts a smaller factor above a larger one, so the largest probability is
        // the largest exponential times the scale. Two exponentials can still round to the same probability, so the
        // index is the first one whose own product equals it.
        probability = largest * scale;
        int i = 0;
        var target = new Vector<float>(probability);
        var factor = new Vector<float>(scale);
        for (; i <= length - Width; i += Width)
        {
            if (Vector.EqualsAny(Load(row + i) * factor, target)) break;
        }

        for (; i < length; i++)
        {
            if (row[i] * scale == probability) return i;
        }

        return 0;
    }

    /// <summary>
    /// Replaces a row with exp(x - max) and returns the sum of the exponentials and the largest of them. The sum is
    /// taken in Vector&lt;T&gt; lanes in the order of the row and then over the tail, whether the exponentials come 512
    /// bits at a time (where AVX-512 is used) or not, so it does not depend on the vector width.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float ExpShiftedSum(float* row, int length, float max, out float largest)
    {
        var acc = Vector<float>.Zero;
        var top = Vector<float>.Zero;
        int i = 0;
        if (Wide)
        {
            var wmax = Vector512.Create(max);
            for (; i <= length - 16; i += 16)
            {
                Vector512.Store(ExpWide(Vector512.Load(row + i) - wmax), row + i);
                for (int j = 0; j < 16; j += Width)
                {
                    var e = Load(row + i + j);
                    acc += e;
                    top = Vector.Max(top, e);
                }
            }
        }

        var vmax = new Vector<float>(max);
        for (; i <= length - Width; i += Width)
        {
            var e = Exp(Load(row + i) - vmax);
            Store(e, row + i);
            acc += e;
            top = Vector.Max(top, e);
        }

        float sum = Vector.Sum(acc);
        largest = 0;
        for (int j = 0; j < Width; j++) largest = Math.Max(largest, top[j]);
        for (; i < length; i++)
        {
            float e = Exp(new Vector<float>(row[i] - max))[0];
            row[i] = e;
            sum += e;
            largest = Math.Max(largest, e);
        }

        return sum;
    }

    /// <summary>Multiplies length values in place by a factor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Scale(float* data, int length, float factor)
    {
        var f = new Vector<float>(factor);
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(data + i) * f, data + i);
        for (; i < length; i++) data[i] *= factor;
    }

    /// <summary>dst[i] += src[i].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddInPlace(float* dst, float* src, int length)
    {
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(dst + i) + Load(src + i), dst + i);
        for (; i < length; i++) dst[i] += src[i];
    }

    /// <summary>dst[i] = a[i] + b[i].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Add(float* a, float* b, float* dst, int length)
    {
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) + Load(b + i), dst + i);
        for (; i < length; i++) dst[i] = a[i] + b[i];
    }

    /// <summary>dst[i] = a[i] * b[i].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Multiply(float* a, float* b, float* dst, int length)
    {
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) * Load(b + i), dst + i);
        for (; i < length; i++) dst[i] = a[i] * b[i];
    }

    /// <summary>dst[i] = a[i] + s.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

    /// <summary>dst[i] = src[2 i] for i in [0, count); reads nothing past src[2 count - 2].</summary>
    /// <remarks>
    /// A vector block of n outputs reads 2 n values, the last of them odd and one past the block's last even one, so
    /// the blocks stop before the end and the last n outputs come from the 2 n values that end at src[2 count - 2]:
    /// that block starts one value early, so its odd values are the even ones of the row.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyEven(float* src, float* dst, int count)
    {
        int i = 0;
        if (Avx512F.IsSupported && count > 16)
        {
            var even = Vector512.Create(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            for (; i + 16 < count; i += 16)
            {
                float* s = src + (2 * i);
                Avx512F.Store(dst + i, Avx512F.PermuteVar16x32x2(Avx512F.LoadVector512(s), even, Avx512F.LoadVector512(s + 16)));
            }

            float* last = src + (2 * (count - 16)) - 1;
            var odd = Vector512.Create(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
            Avx512F.Store(dst + count - 16, Avx512F.PermuteVar16x32x2(Avx512F.LoadVector512(last), odd, Avx512F.LoadVector512(last + 16)));
            return;
        }

        if (Avx2.IsSupported && count > 8)
        {
            for (; i + 8 < count; i += 8)
            {
                // Shuffle picks elements 0 and 2 of each half-lane pair, giving (s0 s2 s8 s10 | s4 s6 s12 s14); the
                // permute puts the 64-bit pairs in order.
                float* s = src + (2 * i);
                var even = Avx.Shuffle(Avx.LoadVector256(s), Avx.LoadVector256(s + 8), 0b10_00_10_00);
                Avx.Store(dst + i, Avx2.Permute4x64(even.AsDouble(), 0b11_01_10_00).AsSingle());
            }

            float* last = src + (2 * (count - 8)) - 1;
            var odd = Avx.Shuffle(Avx.LoadVector256(last), Avx.LoadVector256(last + 8), 0b11_01_11_01);
            Avx.Store(dst + count - 8, Avx2.Permute4x64(odd.AsDouble(), 0b11_01_10_00).AsSingle());
            return;
        }

        for (; i < count; i++) dst[i] = src[2 * i];
    }

    /// <summary>dst[i] = a[i] * s.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MultiplyScalar(float* a, float s, float* dst, int length)
    {
        var v = new Vector<float>(s);
        int i = 0;
        for (; i <= length - Width; i += Width) Store(Load(a + i) * v, dst + i);
        for (; i < length; i++) dst[i] = a[i] * s;
    }
}
