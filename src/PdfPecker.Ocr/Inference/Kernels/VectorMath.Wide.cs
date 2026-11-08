using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Kernels;

// The activations again on 512-bit vectors, for processors with AVX-512, where Vector<T> stays 256 bits wide: the
// epilogue of a matrix product (GELU above all) costs half as much. Each function is the same sequence of operations
// as its Vector<T> twin in VectorMath.cs, lane for lane, so a value gets the same result on either path, which
// KernelTests checks. Change both together.
internal static unsafe partial class VectorMath
{
    /// <summary>True when the 512-bit activations run, on the same condition as the AVX-512 matrix kernel.</summary>
    internal static bool Wide { get; } = Avx512F.IsSupported;

    /// <summary>Activates the leading whole 512-bit vectors of a row and returns how many values it did.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int ActivateWide(float* data, int length, ActivationKind kind, float alpha, float beta)
    {
        int i = 0;
        for (; i <= length - 16; i += 16)
            Vector512.Store(ActivateWide(Vector512.Load(data + i), kind, alpha, beta), data + i);
        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> ActivateWide(Vector512<float> x, ActivationKind kind, float alpha, float beta) => kind switch
    {
        ActivationKind.Relu => Vector512.Max(x, Vector512<float>.Zero),
        ActivationKind.Sigmoid => SigmoidWide(x),
        ActivationKind.HardSigmoid => HardSigmoidWide(x, alpha, beta),
        ActivationKind.HardSwish => x * HardSigmoidWide(x, alpha, beta),
        ActivationKind.Swish => x * SigmoidWide(x),
        ActivationKind.Gelu => x * (Vector512<float>.One + ErfWide(x * Vector512.Create(0.70710678118654752f))) * Vector512.Create(0.5f),
        _ => x,
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> HardSigmoidWide(Vector512<float> x, float alpha, float beta) =>
        Vector512.Min(Vector512.Max(x * Vector512.Create(alpha) + Vector512.Create(beta), Vector512<float>.Zero), Vector512<float>.One);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> SigmoidWide(Vector512<float> x) => Vector512<float>.One / (Vector512<float>.One + ExpWide(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> ExpWide(Vector512<float> x)
    {
        x = Vector512.Min(Vector512.Max(x, Vector512.Create(-87.3365447505531f)), Vector512.Create(88.3762626647949f));
        var fx = Vector512.Floor(x * Vector512.Create(1.44269504088896341f) + Vector512.Create(0.5f));
        x -= fx * Vector512.Create(0.693359375f);
        x -= fx * Vector512.Create(-2.12194440e-4f);
        var z = x * x;
        var y = Vector512.Create(1.9875691500E-4f);
        y = y * x + Vector512.Create(1.3981999507E-3f);
        y = y * x + Vector512.Create(8.3334519073E-3f);
        y = y * x + Vector512.Create(4.1665795894E-2f);
        y = y * x + Vector512.Create(1.6666665459E-1f);
        y = y * x + Vector512.Create(5.0000001201E-1f);
        y = y * z + x + Vector512<float>.One;
        var pow2 = Vector512.ShiftLeft(Vector512.ConvertToInt32(fx) + Vector512.Create(127), 23);
        return y * pow2.AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> ErfWide(Vector512<float> x)
    {
        x = Vector512.Min(Vector512.Max(x, Vector512.Create(-4f)), Vector512.Create(4f));
        var x2 = x * x;
        var p = x2 * Vector512.Create(ErfP13) + Vector512.Create(ErfP11);
        p = x2 * p + Vector512.Create(ErfP9);
        p = x2 * p + Vector512.Create(ErfP7);
        p = x2 * p + Vector512.Create(ErfP5);
        p = x2 * p + Vector512.Create(ErfP3);
        p = x2 * p + Vector512.Create(ErfP1);
        p = x * p;
        var q = x2 * Vector512.Create(ErfQ8) + Vector512.Create(ErfQ6);
        q = x2 * q + Vector512.Create(ErfQ4);
        q = x2 * q + Vector512.Create(ErfQ2);
        q = x2 * q + Vector512.Create(ErfQ0);
        return p / q;
    }
}
