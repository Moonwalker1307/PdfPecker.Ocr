using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace PdfPecker.Ocr.Inference.Kernels;

/// <summary>
/// An experimental 8-bit matrix product for pointwise convolutions (see <see cref="Experiments"/>): weights quantised
/// symmetrically per output channel, activations per column (per pixel, over all input channels) when a tile is packed,
/// products summed exactly in 32-bit integers with AVX-VNNI's <c>vpdpbusd</c>, and each sum scaled back to float with
/// the bias added. It is off by default, because its output is close to the float product's but not equal to it.
/// </summary>
/// <remarks>
/// <para>
/// <c>vpdpbusd</c> multiplies unsigned by signed bytes, so a quantised activation q in [-127, 127] is stored as q + 128
/// and each output channel's sum starts at -128 times the sum of its quantised weights, which takes the offset out
/// again exactly. Four values of k go into each 32-bit lane: weights are packed in panels of MR rows with each row's
/// four bytes together (one broadcast per row and group of four), and activations in panels of NR columns with each
/// column's four bytes together (two 256-bit loads per group).
/// </para>
/// <para>
/// Every value depends only on its own column and output channel: the column's scale is computed from that column alone
/// and the integer sum is exact, so tiling and thread count do not change any bit, and <see cref="Reference"/>, the
/// scalar version the tests compare with, gives the same bits.
/// </para>
/// </remarks>
internal static unsafe class Int8Gemm
{
    /// <summary>Rows of C per kernel call: 12 accumulators, two loads and a broadcast fit the 16 registers VEX can name.</summary>
    public const int MR = 6;

    /// <summary>Columns of C per kernel call.</summary>
    public const int NR = 16;

    /// <summary>
    /// Whether the processor has the instructions of the kernel (AVX-VNNI with FMA). .NET 8 has AVX-VNNI only as a
    /// preview feature, which a package cannot ask its callers to turn on, so there the experiment does nothing.
    /// </summary>
    public static bool IsSupported =>
#if NET9_0_OR_GREATER
        AvxVnni.IsSupported && Fma.IsSupported;
#else
        false;
#endif

    /// <summary>Weights quantised per output channel and packed in MR-row panels of groups of four values of k.</summary>
    internal sealed class Weights
    {
        public Weights(sbyte[] packed, float[] scales, int[] offsets, int rows, int k)
        {
            Packed = packed;
            Scales = scales;
            Offsets = offsets;
            Rows = rows;
            K = k;
        }

        /// <summary>The panels: for each panel of MR rows, for each group g of four k, row r's weights 4g to 4g + 3.</summary>
        public sbyte[] Packed { get; }

        /// <summary>The scale of each row (output channel), padded with zeros to whole panels.</summary>
        public float[] Scales { get; }

        /// <summary>-128 times the sum of each row's quantised weights, the start of its integer sums.</summary>
        public int[] Offsets { get; }

        public int Rows { get; }

        public int K { get; }

        /// <summary>The number of groups of four values of k.</summary>
        public int Groups => (K + 3) / 4;
    }

    /// <summary>Quantises a row-major rows x k weight matrix: each row to [-127, 127] by its largest magnitude.</summary>
    public static Weights QuantizeWeights(ReadOnlySpan<float> w, int rows, int k)
    {
        int groups = (k + 3) / 4;
        int paddedRows = GemmKernel.RoundUp(rows, MR);
        // Pinned, as the float kernel's packed weights are: the kernels read them through pointers for the model's life.
        var packed = GC.AllocateArray<sbyte>(checked(paddedRows * groups * 4), pinned: true);
        var scales = GC.AllocateArray<float>(paddedRows + MR, pinned: true);
        var offsets = GC.AllocateArray<int>(paddedRows + MR, pinned: true);
        var q = new sbyte[k];
        for (int r = 0; r < rows; r++)
        {
            var row = w.Slice(r * k, k);
            float max = 0;
            foreach (float v in row) max = Math.Max(max, Math.Abs(v));
            float scale = max / 127f;
            int sum = 0;
            for (int i = 0; i < k; i++)
            {
                int value = max == 0 ? 0 : Math.Clamp((int)MathF.Round(row[i] / scale, MidpointRounding.ToEven), -127, 127);
                q[i] = (sbyte)value;
                sum += value;
            }

            scales[r] = scale;
            offsets[r] = -128 * sum;
            long panel = (long)(r / MR) * groups * MR * 4;
            int lane = r % MR;
            for (int i = 0; i < k; i++) packed[panel + (i / 4) * MR * 4 + lane * 4 + (i % 4)] = q[i];
        }

        return new Weights(packed, scales, offsets, rows, k);
    }

    /// <summary>The bytes the packed activations of a k x cols tile take, with their column scales after them.</summary>
    public static long PackedBytes(int k, int cols)
    {
        long panels = (long)GemmKernel.RoundUp(cols, NR) * ((k + 3) / 4) * 4;
        return GemmKernel.RoundUp((int)Math.Min(panels, int.MaxValue - 64), 64) + (long)GemmKernel.RoundUp(cols, NR) * sizeof(float);
    }

    /// <summary>Where the column scales of a packed tile start.</summary>
    public static float* ColumnScales(byte* packed, int k, int cols) =>
        (float*)(packed + GemmKernel.RoundUp(GemmKernel.RoundUp(cols, NR) * ((k + 3) / 4) * 4, 64));

    /// <summary>
    /// Quantises columns [0, cols) of a k x ld row-major block of activations (a pointwise convolution's input planes)
    /// into NR-column panels at dst: each column by its own largest magnitude to [-127, 127], stored plus 128, with the
    /// scale of each column at <see cref="ColumnScales"/>. Values of k past the end and columns past cols hold 128 (zero).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeColumns(float* src, nint ld, int k, int cols, byte* dst)
    {
        int groups = (k + 3) / 4;
        float* scales = ColumnScales(dst, k, cols);
        for (int q = 0; q < cols; q += NR)
        {
            int width = Math.Min(NR, cols - q);
            byte* panel = dst + (long)q * groups * 4;
            if (width == NR && Avx512F.IsSupported) QuantizePanelWide(src + q, ld, k, groups, panel, scales + q);
            else if (width == NR && Avx2.IsSupported) QuantizePanel(src + q, ld, k, groups, panel, scales + q);
            else QuantizePanelScalar(src + q, ld, k, width, groups, panel, scales + q);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizePanel(float* src, nint ld, int k, int groups, byte* panel, float* scales)
    {
        var signMask = Vector256.Create(0x7FFFFFFF).AsSingle();
        Vector256<float> m0 = Vector256<float>.Zero, m1 = Vector256<float>.Zero;
        float* s = src;
        for (int kk = 0; kk < k; kk++, s += ld)
        {
            m0 = Avx.Max(m0, Avx.And(Avx.LoadVector256(s), signMask));
            m1 = Avx.Max(m1, Avx.And(Avx.LoadVector256(s + 8), signMask));
        }

        var limit = Vector256.Create(127f);
        var zero = Vector256<float>.Zero;
        Avx.Store(scales, Avx.Divide(m0, limit));
        Avx.Store(scales + 8, Avx.Divide(m1, limit));
        // A column of zeros gets the scale 0 and the inverse 0, so its values quantise to 0 rather than to NaN.
        var inv0 = Avx.AndNot(Avx.CompareEqual(m0, zero), Avx.Divide(limit, m0));
        var inv1 = Avx.AndNot(Avx.CompareEqual(m1, zero), Avx.Divide(limit, m1));
        int* d = (int*)panel;
        for (int g = 0; g < groups; g++, d += NR)
        {
            float* row = src + 4 * g * ld;
            int kk = 4 * g;
            for (int half = 0; half < 2; half++)
            {
                var inv = half == 0 ? inv0 : inv1;
                float* r = row + 8 * half;
                var u0 = QuantizeRow(r, inv, kk < k);
                var u1 = QuantizeRow(r + ld, inv, kk + 1 < k);
                var u2 = QuantizeRow(r + 2 * ld, inv, kk + 2 < k);
                var u3 = QuantizeRow(r + 3 * ld, inv, kk + 3 < k);
                var packed = Avx2.Or(Avx2.Or(u0, Avx2.ShiftLeftLogical(u1, 8)), Avx2.Or(Avx2.ShiftLeftLogical(u2, 16), Avx2.ShiftLeftLogical(u3, 24)));
                Avx.Store(d + 8 * half, packed);
            }
        }
    }

    /// <summary>
    /// <see cref="QuantizePanel"/> on 512-bit vectors: a panel's sixteen columns are one vector, so a group of four
    /// values of k is four loads and one store. The operations are the same, so are the bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizePanelWide(float* src, nint ld, int k, int groups, byte* panel, float* scales)
    {
        var signMask = Vector512.Create(0x7FFFFFFF).AsSingle();
        var m = Vector512<float>.Zero;
        float* s = src;
        for (int kk = 0; kk < k; kk++, s += ld) m = Avx512F.Max(m, Vector512.Load(s) & signMask);
        var limit = Vector512.Create(127f);
        Vector512.Store(Avx512F.Divide(m, limit), scales);
        var inv = Vector512.AndNot(Avx512F.Divide(limit, m), Vector512.Equals(m, Vector512<float>.Zero));
        int* d = (int*)panel;
        for (int g = 0; g < groups; g++, d += NR)
        {
            float* row = src + 4 * g * ld;
            int kk = 4 * g;
            var u0 = QuantizeRowWide(row, inv, kk < k);
            var u1 = QuantizeRowWide(row + ld, inv, kk + 1 < k);
            var u2 = QuantizeRowWide(row + 2 * ld, inv, kk + 2 < k);
            var u3 = QuantizeRowWide(row + 3 * ld, inv, kk + 3 < k);
            Vector512.Store(u0 | Avx512F.ShiftLeftLogical(u1, 8) | Avx512F.ShiftLeftLogical(u2, 16) | Avx512F.ShiftLeftLogical(u3, 24), d);
        }
    }

    /// <summary>Sixteen activations quantised and offset to [1, 255], or 128 (zero) for a value of k past the end.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<int> QuantizeRowWide(float* row, Vector512<float> inv, bool valid)
    {
        if (!valid) return Vector512.Create(128);
        var q = Avx512F.ConvertToVector512Int32(Avx512F.Multiply(Vector512.Load(row), inv));
        return Avx512F.Add(Avx512F.Min(Avx512F.Max(q, Vector512.Create(-127)), Vector512.Create(127)), Vector512.Create(128));
    }

    /// <summary>Eight activations quantised and offset to [1, 255], or 128 (zero) for a value of k past the end.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> QuantizeRow(float* row, Vector256<float> inv, bool valid)
    {
        if (!valid) return Vector256.Create(128);
        var q = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(row), inv));
        return Avx2.Add(Avx2.Min(Avx2.Max(q, Vector256.Create(-127)), Vector256.Create(127)), Vector256.Create(128));
    }

    /// <summary>The same quantisation one value at a time, for a panel's last columns and for processors without AVX2.</summary>
    private static void QuantizePanelScalar(float* src, nint ld, int k, int width, int groups, byte* panel, float* scales)
    {
        for (int j = 0; j < NR; j++)
        {
            float max = 0;
            if (j < width)
            {
                for (int kk = 0; kk < k; kk++) max = Math.Max(max, Math.Abs(src[kk * ld + j]));
            }

            float inv = max == 0 ? 0 : 127f / max;
            scales[j] = max / 127f;
            for (int kk = 0; kk < groups * 4; kk++)
            {
                int value = j < width && kk < k ? Quantize(src[kk * ld + j], inv) : 0;
                panel[(kk / 4) * NR * 4 + j * 4 + (kk % 4)] = (byte)(value + 128);
            }
        }
    }

    /// <summary>One activation quantised as the vector code does: the product rounded to even, then clamped.</summary>
    internal static int Quantize(float value, float inverseScale) =>
        Math.Clamp((int)MathF.Round(value * inverseScale, MidpointRounding.ToEven), -127, 127);

    /// <summary>
    /// C[rows x cols] = dequantised (A x B) + bias for rows [m0, m0 + rows) of the weights and a packed tile of cols
    /// columns; m0 is a multiple of MR. The epilogue runs on each panel of MR rows as soon as it is written, while it
    /// is still in the cache, as in the float kernel: over whole rows the activation's constants and dispatch are paid
    /// once per row rather than once per block, which made the activation in the block cost twice as much for GELU.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Multiply(Weights weights, int m0, int rows, byte* packed, int cols, float* c, nint ldc, float* bias, in Epilogue epilogue = default)
    {
        bool hasEpilogue = !epilogue.IsEmpty;
        int groups = weights.Groups;
        int k = weights.K;
        float* colScales = ColumnScales(packed, k, cols);
        float* edge = stackalloc float[MR * NR];
        fixed (sbyte* a = weights.Packed)
        fixed (float* rowScales = weights.Scales)
        fixed (int* offsets = weights.Offsets)
        {
            for (int i = 0; i < rows; i += MR)
            {
                int row = m0 + i;
                int h = Math.Min(MR, rows - i);
                sbyte* ap = a + (long)row * groups * 4;
                for (int j = 0; j < cols; j += NR)
                {
                    int w = Math.Min(NR, cols - j);
                    byte* bp = packed + (long)j * groups * 4;
                    if (h == MR && w == NR)
                    {
                        Kernel6x16(groups, ap, bp, offsets + row, c + i * ldc + j, ldc, rowScales + row, colScales + j, bias + row);
                    }
                    else
                    {
                        Kernel6x16(groups, ap, bp, offsets + row, edge, NR, rowScales + row, colScales + j, bias + row);
                        for (int r = 0; r < h; r++)
                            Unsafe.CopyBlockUnaligned(c + (i + r) * ldc + j, edge + r * NR, (uint)(w * sizeof(float)));
                    }
                }

                if (hasEpilogue) epilogue.Apply(c + i * ldc, ldc, i * ldc, h, cols);
            }
        }
    }

    /// <summary>
    /// The scalar version of <see cref="QuantizeColumns"/> and <see cref="Multiply"/> without the epilogue, value by value
    /// with the same arithmetic, for tests: C = bias + float(sum) x (row scale x column scale), fused.
    /// </summary>
    internal static void Reference(Weights weights, ReadOnlySpan<float> b, int k, int cols, Span<float> c, ReadOnlySpan<float> bias)
    {
        for (int j = 0; j < cols; j++)
        {
            float max = 0;
            for (int kk = 0; kk < k; kk++) max = Math.Max(max, Math.Abs(b[kk * cols + j]));
            float inv = max == 0 ? 0 : 127f / max;
            float columnScale = max / 127f;
            for (int r = 0; r < weights.Rows; r++)
            {
                int sum = weights.Offsets[r];
                for (int kk = 0; kk < k; kk++)
                {
                    long at = (long)(r / MR) * weights.Groups * MR * 4 + (kk / 4) * MR * 4 + (r % MR) * 4 + (kk % 4);
                    sum += weights.Packed[at] * (Quantize(b[kk * cols + j], inv) + 128);
                }

                c[r * cols + j] = MathF.FusedMultiplyAdd(sum, weights.Scales[r] * columnScale, bias[r]);
            }
        }
    }

    /// <summary>
    /// A 6 x 16 block: twelve 256-bit integer accumulators over all groups of k, each then stored as bias + float(sum) x
    /// (row scale x column scale). The epilogue is not here: with an activation in the same method the JIT, which can
    /// name only the sixteen registers of VEX for AVX-VNNI, kept four of the twelve accumulators on the stack.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x16(int groups, sbyte* a, byte* b, int* offsets, float* c, nint ldc, float* rowScales, float* colScales, float* bias)
    {
#if NET9_0_OR_GREATER
        var c0_0 = Vector256.Create(offsets[0]);
        var c0_1 = c0_0;
        var c1_0 = Vector256.Create(offsets[1]);
        var c1_1 = c1_0;
        var c2_0 = Vector256.Create(offsets[2]);
        var c2_1 = c2_0;
        var c3_0 = Vector256.Create(offsets[3]);
        var c3_1 = c3_0;
        var c4_0 = Vector256.Create(offsets[4]);
        var c4_1 = c4_0;
        var c5_0 = Vector256.Create(offsets[5]);
        var c5_1 = c5_0;
        for (; groups > 0; groups--)
        {
            var b0 = Avx.LoadVector256(b);
            var b1 = Avx.LoadVector256(b + 32);
            var a0 = Vector256.Create(*(int*)a).AsSByte();
            c0_0 = AvxVnni.MultiplyWideningAndAdd(c0_0, b0, a0);
            c0_1 = AvxVnni.MultiplyWideningAndAdd(c0_1, b1, a0);
            var a1 = Vector256.Create(*(int*)(a + 4)).AsSByte();
            c1_0 = AvxVnni.MultiplyWideningAndAdd(c1_0, b0, a1);
            c1_1 = AvxVnni.MultiplyWideningAndAdd(c1_1, b1, a1);
            var a2 = Vector256.Create(*(int*)(a + 8)).AsSByte();
            c2_0 = AvxVnni.MultiplyWideningAndAdd(c2_0, b0, a2);
            c2_1 = AvxVnni.MultiplyWideningAndAdd(c2_1, b1, a2);
            var a3 = Vector256.Create(*(int*)(a + 12)).AsSByte();
            c3_0 = AvxVnni.MultiplyWideningAndAdd(c3_0, b0, a3);
            c3_1 = AvxVnni.MultiplyWideningAndAdd(c3_1, b1, a3);
            var a4 = Vector256.Create(*(int*)(a + 16)).AsSByte();
            c4_0 = AvxVnni.MultiplyWideningAndAdd(c4_0, b0, a4);
            c4_1 = AvxVnni.MultiplyWideningAndAdd(c4_1, b1, a4);
            var a5 = Vector256.Create(*(int*)(a + 20)).AsSByte();
            c5_0 = AvxVnni.MultiplyWideningAndAdd(c5_0, b0, a5);
            c5_1 = AvxVnni.MultiplyWideningAndAdd(c5_1, b1, a5);
            a += 24;
            b += 64;
        }

        var s0 = Avx.LoadVector256(colScales);
        var s1 = Avx.LoadVector256(colScales + 8);
        Scale(c0_0, c0_1, s0, s1, rowScales[0], bias[0], c);
        Scale(c1_0, c1_1, s0, s1, rowScales[1], bias[1], c + ldc);
        Scale(c2_0, c2_1, s0, s1, rowScales[2], bias[2], c + 2 * ldc);
        Scale(c3_0, c3_1, s0, s1, rowScales[3], bias[3], c + 3 * ldc);
        Scale(c4_0, c4_1, s0, s1, rowScales[4], bias[4], c + 4 * ldc);
        Scale(c5_0, c5_1, s0, s1, rowScales[5], bias[5], c + 5 * ldc);
#else
        // Never reached: IsSupported is false on .NET 8, so no model gets the 8-bit layer there.
        throw new PlatformNotSupportedException();
#endif
    }

    /// <summary>Stores two vectors of one row: bias + float(sum) x (row scale x column scale), fused.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Scale(Vector256<int> sum0, Vector256<int> sum1, Vector256<float> colScale0, Vector256<float> colScale1, float rowScale, float bias, float* c)
    {
        var scale = Vector256.Create(rowScale);
        var b = Vector256.Create(bias);
        Avx.Store(c, Fma.MultiplyAdd(Avx.ConvertToVector256Single(sum0), Avx.Multiply(scale, colScale0), b));
        Avx.Store(c + 8, Fma.MultiplyAdd(Avx.ConvertToVector256Single(sum1), Avx.Multiply(scale, colScale1), b));
    }
}
