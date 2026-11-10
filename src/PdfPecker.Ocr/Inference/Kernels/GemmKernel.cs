using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Kernels;

/// <summary>
/// A register-blocked matrix multiplication microkernel and the panel layout it reads. The kernel computes an MR x NR
/// block of C from an A panel (MR values per step of k) and a B panel (NR values per step of k), so the weights of a
/// layer are packed into panels once when the model loads and activations are packed per tile at run time.
/// </summary>
/// <remarks>
/// Every value of C is a sum over the whole of k in one kernel call, in the same order, whichever tile it lands in;
/// edges are padded with zeros rather than handled by a different code path. That is what makes results independent of
/// the tiling and therefore of the number of threads. A later tuning phase can add k blocking or fused epilogues here
/// without changing callers.
/// </remarks>
internal sealed unsafe class GemmKernel
{
    private readonly delegate*<int, float*, float*, float*, nint, float*, float*, void> _kernel;
    private readonly delegate*<int, float*, float*, int*, float*, nint, float*, void> _direct;

    private GemmKernel(string name, int mr, int nr, delegate*<int, float*, float*, float*, nint, float*, float*, void> kernel, delegate*<int, float*, float*, int*, float*, nint, float*, void> direct, bool isSupported)
    {
        Name = name;
        MR = mr;
        NR = nr;
        _kernel = kernel;
        _direct = direct;
        IsSupported = isSupported;
    }

    /// <summary>12 x 32 with AVX-512.</summary>
    public static GemmKernel Avx512 { get; } = new("avx512", 12, 32, &Kernel12x32Avx512, &Direct12x32Avx512, Avx512F.IsSupported);

    /// <summary>6 x 16 with AVX2 and FMA.</summary>
    public static GemmKernel Avx2 { get; } = new("avx2", 6, 16, &Kernel6x16Avx2, &Direct6x16Avx2, Avx.IsSupported && Fma.IsSupported);

    /// <summary>6 x 8 with Vector128, which every platform runs (in software where it has no SIMD).</summary>
    public static GemmKernel Portable { get; } = new("portable", 6, 8, &Kernel6x8Vector128, &Direct6x8Vector128, true);

    /// <summary>The fastest kernel the processor supports.</summary>
    public static GemmKernel Best { get; } = Avx512.IsSupported ? Avx512 : Avx2.IsSupported ? Avx2 : Portable;

    /// <summary>All kernels this processor can run, for tests.</summary>
    public static IEnumerable<GemmKernel> Supported => new[] { Avx512, Avx2, Portable }.Where(k => k.IsSupported);

    public string Name { get; }

    /// <summary>Rows of C per kernel call.</summary>
    public int MR { get; }

    /// <summary>Columns of C per kernel call.</summary>
    public int NR { get; }

    public bool IsSupported { get; }

    /// <summary>The number of floats an MR-panel packing of a rows x k matrix takes.</summary>
    public long PackedASize(int rows, int k) => (long)RoundUp(rows, MR) * k;

    /// <summary>The number of floats an NR-panel packing of a k x cols matrix takes.</summary>
    public long PackedBSize(int k, int cols) => (long)RoundUp(cols, NR) * k;

    /// <summary>Packs a row-major rows x k matrix, such as convolution weights, into zero-padded MR-row panels.</summary>
    public float[] PackA(ReadOnlySpan<float> a, int rows, int k)
    {
        var packed = Allocate(PackedASize(rows, k));
        for (int p = 0; p < rows; p += MR)
        {
            int dst = p * k;
            for (int kk = 0; kk < k; kk++)
                for (int r = 0; r < MR; r++)
                    packed[dst++] = p + r < rows ? a[(p + r) * k + kk] : 0f;
        }
        return packed;
    }

    /// <summary>Packs a row-major k x cols matrix, such as linear weights, into zero-padded NR-column panels.</summary>
    public float[] PackB(ReadOnlySpan<float> b, int k, int cols)
    {
        var packed = Allocate(PackedBSize(k, cols));
        for (int q = 0; q < cols; q += NR)
        {
            int dst = q * k;
            for (int kk = 0; kk < k; kk++)
                for (int j = 0; j < NR; j++)
                    packed[dst++] = q + j < cols ? b[kk * cols + q + j] : 0f;
        }
        return packed;
    }

    /// <summary>A copy of a bias vector padded with zeros, so kernels can read whole panels of it.</summary>
    public float[] PadBias(ReadOnlySpan<float> bias, int length)
    {
        var padded = Allocate(RoundUp(length, Math.Max(MR, NR)) + 64);
        bias[..length].CopyTo(padded);
        return padded;
    }

    /// <summary>
    /// Packs columns [0, cols) of a k x ld row-major block into the NR-column panels at dst as columns [first, first +
    /// cols), so that pieces of several blocks fill one set of panels. Lanes after the last piece are left as they are;
    /// <see cref="ClearPanelTail"/> zeroes them.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void PackBRowsAt(float* src, nint ld, int k, int cols, float* dst, int first)
    {
        int nr = NR;
        for (int pos = first, end = first + cols; pos < end;)
        {
            int lane = pos % nr;
            int w = Math.Min(nr - lane, end - pos);
            float* d = dst + ((nint)(pos - lane) * k) + lane;
            float* s = src + (pos - first);
            for (int kk = 0; kk < k; kk++, s += ld, d += nr) Unsafe.CopyBlockUnaligned(d, s, (uint)(w * sizeof(float)));
            pos += w;
        }
    }

    /// <summary>Zeroes the lanes after column cols of the last NR-column panel of a k-row packing.</summary>
    public void ClearPanelTail(float* dst, int k, int cols)
    {
        int used = cols % NR;
        if (used == 0) return;
        float* panel = dst + ((nint)(cols - used) * k);
        for (int r = 0; r < k; r++) new Span<float>(panel + ((nint)r * NR) + used, NR - used).Clear();
    }

    /// <summary>Packs columns [0, cols) of a k x ld row-major block (activations) into NR-column panels at dst.</summary>
    /// <remarks>
    /// Row by row rather than panel by panel: each source row is then read once from start to end, one stream the
    /// hardware prefetcher follows, where a panel at a time read a short piece of all k rows, more streams than it
    /// tracks, and waited for memory on most of them.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void PackBRows(float* src, nint ld, int k, int cols, float* dst)
    {
        int nr = NR;
        int whole = cols / nr * nr;
        nint panel = (nint)nr * k;
        for (int kk = 0; kk < k; kk++)
        {
            float* s = src + kk * ld;
            float* d = dst + (nint)kk * nr;
            int q = 0;
            if (nr == 32 && Vector512.IsHardwareAccelerated)
            {
                for (; q < whole; q += 32, d += panel)
                {
                    Vector512.Store(Vector512.Load(s + q), d);
                    Vector512.Store(Vector512.Load(s + q + 16), d + 16);
                }
            }
            else if (nr == 16 && Vector256.IsHardwareAccelerated)
            {
                for (; q < whole; q += 16, d += panel)
                {
                    Vector256.Store(Vector256.Load(s + q), d);
                    Vector256.Store(Vector256.Load(s + q + 8), d + 8);
                }
            }
            else
            {
                for (; q < whole; q += nr, d += panel) Unsafe.CopyBlockUnaligned(d, s + q, (uint)(nr * sizeof(float)));
            }

            if (q < cols)
            {
                int w = cols - q;
                Unsafe.CopyBlockUnaligned(d, s + q, (uint)(w * sizeof(float)));
                new Span<float>(d + w, nr - w).Clear();
            }
        }
    }

    /// <summary>
    /// Packs the rows of a row-major block (activations, row stride ld) that <paramref name="rowIndex"/> lists, in that
    /// order, into MR-row panels, as <see cref="PackARows(float*, nint, int, int, float*)"/> packs consecutive ones.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void PackARows(float* src, nint ld, ReadOnlySpan<int> rowIndex, int k, float* dst)
    {
        int mr = MR, rows = rowIndex.Length;
        for (int p = 0; p < rows; p += mr)
        {
            int h = Math.Min(mr, rows - p);
            float* d = dst + (nint)p * k;
            for (int r = 0; r < mr; r++, d++)
            {
                if (r >= h)
                {
                    for (int kk = 0; kk < k; kk++) d[(nint)kk * mr] = 0f;
                    continue;
                }

                float* s = src + rowIndex[p + r] * ld;
                for (int kk = 0; kk < k; kk++) d[(nint)kk * mr] = s[kk];
            }
        }
    }

    /// <summary>Packs rows [0, rows) of a rows x k row-major block (activations, row stride ld) into MR-row panels.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void PackARows(float* src, nint ld, int rows, int k, float* dst)
    {
        int mr = MR;
        for (int p = 0; p < rows; p += mr)
        {
            int h = Math.Min(mr, rows - p);
            float* d = dst + (nint)p * k;
            for (int kk = 0; kk < k; kk++, d += mr)
            {
                float* s = src + p * ld + kk;
                int r = 0;
                for (; r < h; r++, s += ld) d[r] = *s;
                for (; r < mr; r++) d[r] = 0f;
            }
        }
    }

    /// <summary>
    /// C[rows x cols] = A x B (+ bias), where A is MR-row panels starting at a row that is a multiple of MR and B is
    /// NR-column panels starting at a column that is a multiple of NR. rowBias (one value per row of C) or colBias (one
    /// per column) seed the sums; both must be padded to whole panels. The epilogue runs on each panel of MR rows as
    /// soon as it is written, while it is still in the cache.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Multiply(int k, float* aPanels, int rows, float* bPanels, int cols, float* c, nint ldc, float* rowBias, float* colBias, in Epilogue epilogue = default)
    {
        bool hasEpilogue = !epilogue.IsEmpty;
        int mr = MR, nr = NR;
        var kernel = _kernel;
        float* edge = stackalloc float[12 * 32];
        for (int i = 0; i < rows; i += mr)
        {
            int h = Math.Min(mr, rows - i);
            float* ap = aPanels + (nint)i * k;
            float* rb = rowBias == null ? null : rowBias + i;
            for (int j = 0; j < cols; j += nr)
            {
                int w = Math.Min(nr, cols - j);
                float* bp = bPanels + (nint)j * k;
                float* cb = colBias == null ? null : colBias + j;
                if (h == mr && w == nr)
                {
                    kernel(k, ap, bp, c + i * ldc + j, ldc, rb, cb);
                }
                else
                {
                    kernel(k, ap, bp, edge, nr, rb, cb);
                    for (int r = 0; r < h; r++)
                        Unsafe.CopyBlockUnaligned(c + (i + r) * ldc + j, edge + r * nr, (uint)(w * sizeof(float)));
                }
            }
            if (hasEpilogue) epilogue.Apply(c + i * ldc, ldc, i, h, cols);
        }
    }

    /// <summary>
    /// C[rows x cols] = A x B + rowBias for one panel of B, cols at most NR, that is read where it lies instead of
    /// packed: step kk of k reads the NR values at b + offsets[kk], such as one tap's run of an input row, all of which
    /// must be readable. The sums are the ones <see cref="Multiply"/> computes from the same values packed, in the same
    /// order. There is no epilogue: callers apply it to a whole tile. A partial block goes through edge, MR x NR floats
    /// the caller provides, so that a call per panel does not clear a buffer of its own on the stack.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void MultiplyDirect(int k, float* aPanels, int rows, float* b, int* offsets, float* c, nint ldc, float* rowBias, float* edge, int cols)
    {
        int mr = MR, nr = NR;
        var kernel = _direct;
        for (int i = 0; i < rows; i += mr)
        {
            int h = Math.Min(mr, rows - i);
            float* ap = aPanels + (nint)i * k;
            float* rb = rowBias == null ? null : rowBias + i;
            if (h == mr && cols == nr)
            {
                kernel(k, ap, b, offsets, c + i * ldc, ldc, rb);
            }
            else
            {
                kernel(k, ap, b, offsets, edge, nr, rb);
                for (int r = 0; r < h; r++)
                    Unsafe.CopyBlockUnaligned(c + (i + r) * ldc, edge + r * nr, (uint)(cols * sizeof(float)));
            }
        }
    }

    public static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

    private static float[] Allocate(long length) => GC.AllocateArray<float>(checked((int)Math.Max(length, 1)), pinned: true);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        if (Fma.IsSupported) return Fma.MultiplyAdd(a, b, c);
        if (AdvSimd.IsSupported) return AdvSimd.FusedMultiplyAdd(c, a, b);
        return a * b + c;
    }

    /// <summary>AVX-512 kernel: 12 rows by 32 columns in 24 zmm accumulators.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel12x32Avx512(int k, float* a, float* b, float* c, nint ldc, float* rowBias, float* colBias)
    {
        Vector512<float> c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1, c6_0, c6_1, c7_0, c7_1, c8_0, c8_1, c9_0, c9_1, c10_0, c10_1, c11_0, c11_1;
        if (rowBias != null)
        {
            c0_0 = Vector512.Create(rowBias[0]);
            c0_1 = c0_0;
            c1_0 = Vector512.Create(rowBias[1]);
            c1_1 = c1_0;
            c2_0 = Vector512.Create(rowBias[2]);
            c2_1 = c2_0;
            c3_0 = Vector512.Create(rowBias[3]);
            c3_1 = c3_0;
            c4_0 = Vector512.Create(rowBias[4]);
            c4_1 = c4_0;
            c5_0 = Vector512.Create(rowBias[5]);
            c5_1 = c5_0;
            c6_0 = Vector512.Create(rowBias[6]);
            c6_1 = c6_0;
            c7_0 = Vector512.Create(rowBias[7]);
            c7_1 = c7_0;
            c8_0 = Vector512.Create(rowBias[8]);
            c8_1 = c8_0;
            c9_0 = Vector512.Create(rowBias[9]);
            c9_1 = c9_0;
            c10_0 = Vector512.Create(rowBias[10]);
            c10_1 = c10_0;
            c11_0 = Vector512.Create(rowBias[11]);
            c11_1 = c11_0;
        }
        else if (colBias != null)
        {
            c0_0 = Vector512.Load(colBias + 0);
            c0_1 = Vector512.Load(colBias + 16);
            c1_0 = c0_0;
            c1_1 = c0_1;
            c2_0 = c0_0;
            c2_1 = c0_1;
            c3_0 = c0_0;
            c3_1 = c0_1;
            c4_0 = c0_0;
            c4_1 = c0_1;
            c5_0 = c0_0;
            c5_1 = c0_1;
            c6_0 = c0_0;
            c6_1 = c0_1;
            c7_0 = c0_0;
            c7_1 = c0_1;
            c8_0 = c0_0;
            c8_1 = c0_1;
            c9_0 = c0_0;
            c9_1 = c0_1;
            c10_0 = c0_0;
            c10_1 = c0_1;
            c11_0 = c0_0;
            c11_1 = c0_1;
        }
        else
        {
            c0_0 = Vector512<float>.Zero;
            c0_1 = Vector512<float>.Zero;
            c1_0 = Vector512<float>.Zero;
            c1_1 = Vector512<float>.Zero;
            c2_0 = Vector512<float>.Zero;
            c2_1 = Vector512<float>.Zero;
            c3_0 = Vector512<float>.Zero;
            c3_1 = Vector512<float>.Zero;
            c4_0 = Vector512<float>.Zero;
            c4_1 = Vector512<float>.Zero;
            c5_0 = Vector512<float>.Zero;
            c5_1 = Vector512<float>.Zero;
            c6_0 = Vector512<float>.Zero;
            c6_1 = Vector512<float>.Zero;
            c7_0 = Vector512<float>.Zero;
            c7_1 = Vector512<float>.Zero;
            c8_0 = Vector512<float>.Zero;
            c8_1 = Vector512<float>.Zero;
            c9_0 = Vector512<float>.Zero;
            c9_1 = Vector512<float>.Zero;
            c10_0 = Vector512<float>.Zero;
            c10_1 = Vector512<float>.Zero;
            c11_0 = Vector512<float>.Zero;
            c11_1 = Vector512<float>.Zero;
        }
        for (; k > 0; k--)
        {
            var b0 = Vector512.Load(b + 0);
            var b1 = Vector512.Load(b + 16);
            var a0 = Vector512.Create(a[0]);
            c0_0 = Avx512F.FusedMultiplyAdd(a0, b0, c0_0);
            c0_1 = Avx512F.FusedMultiplyAdd(a0, b1, c0_1);
            var a1 = Vector512.Create(a[1]);
            c1_0 = Avx512F.FusedMultiplyAdd(a1, b0, c1_0);
            c1_1 = Avx512F.FusedMultiplyAdd(a1, b1, c1_1);
            var a2 = Vector512.Create(a[2]);
            c2_0 = Avx512F.FusedMultiplyAdd(a2, b0, c2_0);
            c2_1 = Avx512F.FusedMultiplyAdd(a2, b1, c2_1);
            var a3 = Vector512.Create(a[3]);
            c3_0 = Avx512F.FusedMultiplyAdd(a3, b0, c3_0);
            c3_1 = Avx512F.FusedMultiplyAdd(a3, b1, c3_1);
            var a4 = Vector512.Create(a[4]);
            c4_0 = Avx512F.FusedMultiplyAdd(a4, b0, c4_0);
            c4_1 = Avx512F.FusedMultiplyAdd(a4, b1, c4_1);
            var a5 = Vector512.Create(a[5]);
            c5_0 = Avx512F.FusedMultiplyAdd(a5, b0, c5_0);
            c5_1 = Avx512F.FusedMultiplyAdd(a5, b1, c5_1);
            var a6 = Vector512.Create(a[6]);
            c6_0 = Avx512F.FusedMultiplyAdd(a6, b0, c6_0);
            c6_1 = Avx512F.FusedMultiplyAdd(a6, b1, c6_1);
            var a7 = Vector512.Create(a[7]);
            c7_0 = Avx512F.FusedMultiplyAdd(a7, b0, c7_0);
            c7_1 = Avx512F.FusedMultiplyAdd(a7, b1, c7_1);
            var a8 = Vector512.Create(a[8]);
            c8_0 = Avx512F.FusedMultiplyAdd(a8, b0, c8_0);
            c8_1 = Avx512F.FusedMultiplyAdd(a8, b1, c8_1);
            var a9 = Vector512.Create(a[9]);
            c9_0 = Avx512F.FusedMultiplyAdd(a9, b0, c9_0);
            c9_1 = Avx512F.FusedMultiplyAdd(a9, b1, c9_1);
            var a10 = Vector512.Create(a[10]);
            c10_0 = Avx512F.FusedMultiplyAdd(a10, b0, c10_0);
            c10_1 = Avx512F.FusedMultiplyAdd(a10, b1, c10_1);
            var a11 = Vector512.Create(a[11]);
            c11_0 = Avx512F.FusedMultiplyAdd(a11, b0, c11_0);
            c11_1 = Avx512F.FusedMultiplyAdd(a11, b1, c11_1);
            a += 12;
            b += 32;
        }
        Vector512.Store(c0_0, c);
        Vector512.Store(c0_1, c + 16);
        Vector512.Store(c1_0, c + ldc);
        Vector512.Store(c1_1, c + ldc + 16);
        Vector512.Store(c2_0, c + 2 * ldc);
        Vector512.Store(c2_1, c + 2 * ldc + 16);
        Vector512.Store(c3_0, c + 3 * ldc);
        Vector512.Store(c3_1, c + 3 * ldc + 16);
        Vector512.Store(c4_0, c + 4 * ldc);
        Vector512.Store(c4_1, c + 4 * ldc + 16);
        Vector512.Store(c5_0, c + 5 * ldc);
        Vector512.Store(c5_1, c + 5 * ldc + 16);
        Vector512.Store(c6_0, c + 6 * ldc);
        Vector512.Store(c6_1, c + 6 * ldc + 16);
        Vector512.Store(c7_0, c + 7 * ldc);
        Vector512.Store(c7_1, c + 7 * ldc + 16);
        Vector512.Store(c8_0, c + 8 * ldc);
        Vector512.Store(c8_1, c + 8 * ldc + 16);
        Vector512.Store(c9_0, c + 9 * ldc);
        Vector512.Store(c9_1, c + 9 * ldc + 16);
        Vector512.Store(c10_0, c + 10 * ldc);
        Vector512.Store(c10_1, c + 10 * ldc + 16);
        Vector512.Store(c11_0, c + 11 * ldc);
        Vector512.Store(c11_1, c + 11 * ldc + 16);
    }

    /// <summary>AVX2 and FMA kernel: 6 rows by 16 columns in 12 ymm accumulators.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x16Avx2(int k, float* a, float* b, float* c, nint ldc, float* rowBias, float* colBias)
    {
        Vector256<float> c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
        if (rowBias != null)
        {
            c0_0 = Vector256.Create(rowBias[0]);
            c0_1 = c0_0;
            c1_0 = Vector256.Create(rowBias[1]);
            c1_1 = c1_0;
            c2_0 = Vector256.Create(rowBias[2]);
            c2_1 = c2_0;
            c3_0 = Vector256.Create(rowBias[3]);
            c3_1 = c3_0;
            c4_0 = Vector256.Create(rowBias[4]);
            c4_1 = c4_0;
            c5_0 = Vector256.Create(rowBias[5]);
            c5_1 = c5_0;
        }
        else if (colBias != null)
        {
            c0_0 = Vector256.Load(colBias + 0);
            c0_1 = Vector256.Load(colBias + 8);
            c1_0 = c0_0;
            c1_1 = c0_1;
            c2_0 = c0_0;
            c2_1 = c0_1;
            c3_0 = c0_0;
            c3_1 = c0_1;
            c4_0 = c0_0;
            c4_1 = c0_1;
            c5_0 = c0_0;
            c5_1 = c0_1;
        }
        else
        {
            c0_0 = Vector256<float>.Zero;
            c0_1 = Vector256<float>.Zero;
            c1_0 = Vector256<float>.Zero;
            c1_1 = Vector256<float>.Zero;
            c2_0 = Vector256<float>.Zero;
            c2_1 = Vector256<float>.Zero;
            c3_0 = Vector256<float>.Zero;
            c3_1 = Vector256<float>.Zero;
            c4_0 = Vector256<float>.Zero;
            c4_1 = Vector256<float>.Zero;
            c5_0 = Vector256<float>.Zero;
            c5_1 = Vector256<float>.Zero;
        }
        for (; k > 0; k--)
        {
            var b0 = Vector256.Load(b + 0);
            var b1 = Vector256.Load(b + 8);
            var a0 = Vector256.Create(a[0]);
            c0_0 = Fma.MultiplyAdd(a0, b0, c0_0);
            c0_1 = Fma.MultiplyAdd(a0, b1, c0_1);
            var a1 = Vector256.Create(a[1]);
            c1_0 = Fma.MultiplyAdd(a1, b0, c1_0);
            c1_1 = Fma.MultiplyAdd(a1, b1, c1_1);
            var a2 = Vector256.Create(a[2]);
            c2_0 = Fma.MultiplyAdd(a2, b0, c2_0);
            c2_1 = Fma.MultiplyAdd(a2, b1, c2_1);
            var a3 = Vector256.Create(a[3]);
            c3_0 = Fma.MultiplyAdd(a3, b0, c3_0);
            c3_1 = Fma.MultiplyAdd(a3, b1, c3_1);
            var a4 = Vector256.Create(a[4]);
            c4_0 = Fma.MultiplyAdd(a4, b0, c4_0);
            c4_1 = Fma.MultiplyAdd(a4, b1, c4_1);
            var a5 = Vector256.Create(a[5]);
            c5_0 = Fma.MultiplyAdd(a5, b0, c5_0);
            c5_1 = Fma.MultiplyAdd(a5, b1, c5_1);
            a += 6;
            b += 16;
        }
        Vector256.Store(c0_0, c);
        Vector256.Store(c0_1, c + 8);
        Vector256.Store(c1_0, c + ldc);
        Vector256.Store(c1_1, c + ldc + 8);
        Vector256.Store(c2_0, c + 2 * ldc);
        Vector256.Store(c2_1, c + 2 * ldc + 8);
        Vector256.Store(c3_0, c + 3 * ldc);
        Vector256.Store(c3_1, c + 3 * ldc + 8);
        Vector256.Store(c4_0, c + 4 * ldc);
        Vector256.Store(c4_1, c + 4 * ldc + 8);
        Vector256.Store(c5_0, c + 5 * ldc);
        Vector256.Store(c5_1, c + 5 * ldc + 8);
    }

    /// <summary>Portable kernel: 6 rows by 8 columns in 12 Vector128 accumulators, for Arm64 (AdvSimd) and x64 without AVX2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x8Vector128(int k, float* a, float* b, float* c, nint ldc, float* rowBias, float* colBias)
    {
        Vector128<float> c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
        if (rowBias != null)
        {
            c0_0 = Vector128.Create(rowBias[0]);
            c0_1 = c0_0;
            c1_0 = Vector128.Create(rowBias[1]);
            c1_1 = c1_0;
            c2_0 = Vector128.Create(rowBias[2]);
            c2_1 = c2_0;
            c3_0 = Vector128.Create(rowBias[3]);
            c3_1 = c3_0;
            c4_0 = Vector128.Create(rowBias[4]);
            c4_1 = c4_0;
            c5_0 = Vector128.Create(rowBias[5]);
            c5_1 = c5_0;
        }
        else if (colBias != null)
        {
            c0_0 = Vector128.Load(colBias + 0);
            c0_1 = Vector128.Load(colBias + 4);
            c1_0 = c0_0;
            c1_1 = c0_1;
            c2_0 = c0_0;
            c2_1 = c0_1;
            c3_0 = c0_0;
            c3_1 = c0_1;
            c4_0 = c0_0;
            c4_1 = c0_1;
            c5_0 = c0_0;
            c5_1 = c0_1;
        }
        else
        {
            c0_0 = Vector128<float>.Zero;
            c0_1 = Vector128<float>.Zero;
            c1_0 = Vector128<float>.Zero;
            c1_1 = Vector128<float>.Zero;
            c2_0 = Vector128<float>.Zero;
            c2_1 = Vector128<float>.Zero;
            c3_0 = Vector128<float>.Zero;
            c3_1 = Vector128<float>.Zero;
            c4_0 = Vector128<float>.Zero;
            c4_1 = Vector128<float>.Zero;
            c5_0 = Vector128<float>.Zero;
            c5_1 = Vector128<float>.Zero;
        }
        for (; k > 0; k--)
        {
            var b0 = Vector128.Load(b + 0);
            var b1 = Vector128.Load(b + 4);
            var a0 = Vector128.Create(a[0]);
            c0_0 = MultiplyAdd(a0, b0, c0_0);
            c0_1 = MultiplyAdd(a0, b1, c0_1);
            var a1 = Vector128.Create(a[1]);
            c1_0 = MultiplyAdd(a1, b0, c1_0);
            c1_1 = MultiplyAdd(a1, b1, c1_1);
            var a2 = Vector128.Create(a[2]);
            c2_0 = MultiplyAdd(a2, b0, c2_0);
            c2_1 = MultiplyAdd(a2, b1, c2_1);
            var a3 = Vector128.Create(a[3]);
            c3_0 = MultiplyAdd(a3, b0, c3_0);
            c3_1 = MultiplyAdd(a3, b1, c3_1);
            var a4 = Vector128.Create(a[4]);
            c4_0 = MultiplyAdd(a4, b0, c4_0);
            c4_1 = MultiplyAdd(a4, b1, c4_1);
            var a5 = Vector128.Create(a[5]);
            c5_0 = MultiplyAdd(a5, b0, c5_0);
            c5_1 = MultiplyAdd(a5, b1, c5_1);
            a += 6;
            b += 8;
        }
        Vector128.Store(c0_0, c);
        Vector128.Store(c0_1, c + 4);
        Vector128.Store(c1_0, c + ldc);
        Vector128.Store(c1_1, c + ldc + 4);
        Vector128.Store(c2_0, c + 2 * ldc);
        Vector128.Store(c2_1, c + 2 * ldc + 4);
        Vector128.Store(c3_0, c + 3 * ldc);
        Vector128.Store(c3_1, c + 3 * ldc + 4);
        Vector128.Store(c4_0, c + 4 * ldc);
        Vector128.Store(c4_1, c + 4 * ldc + 4);
        Vector128.Store(c5_0, c + 5 * ldc);
        Vector128.Store(c5_1, c + 5 * ldc + 4);
    }

    /// <summary>Kernel12x32Avx512 with B read in place: step kk of k reads the 32 values at b + offsets[kk].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Direct12x32Avx512(int k, float* a, float* b, int* offsets, float* c, nint ldc, float* rowBias)
    {
        Vector512<float> c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1, c6_0, c6_1, c7_0, c7_1, c8_0, c8_1, c9_0, c9_1, c10_0, c10_1, c11_0, c11_1;
        if (rowBias != null)
        {
            c0_0 = Vector512.Create(rowBias[0]);
            c0_1 = c0_0;
            c1_0 = Vector512.Create(rowBias[1]);
            c1_1 = c1_0;
            c2_0 = Vector512.Create(rowBias[2]);
            c2_1 = c2_0;
            c3_0 = Vector512.Create(rowBias[3]);
            c3_1 = c3_0;
            c4_0 = Vector512.Create(rowBias[4]);
            c4_1 = c4_0;
            c5_0 = Vector512.Create(rowBias[5]);
            c5_1 = c5_0;
            c6_0 = Vector512.Create(rowBias[6]);
            c6_1 = c6_0;
            c7_0 = Vector512.Create(rowBias[7]);
            c7_1 = c7_0;
            c8_0 = Vector512.Create(rowBias[8]);
            c8_1 = c8_0;
            c9_0 = Vector512.Create(rowBias[9]);
            c9_1 = c9_0;
            c10_0 = Vector512.Create(rowBias[10]);
            c10_1 = c10_0;
            c11_0 = Vector512.Create(rowBias[11]);
            c11_1 = c11_0;
        }
        else
        {
            c0_0 = Vector512<float>.Zero;
            c0_1 = Vector512<float>.Zero;
            c1_0 = Vector512<float>.Zero;
            c1_1 = Vector512<float>.Zero;
            c2_0 = Vector512<float>.Zero;
            c2_1 = Vector512<float>.Zero;
            c3_0 = Vector512<float>.Zero;
            c3_1 = Vector512<float>.Zero;
            c4_0 = Vector512<float>.Zero;
            c4_1 = Vector512<float>.Zero;
            c5_0 = Vector512<float>.Zero;
            c5_1 = Vector512<float>.Zero;
            c6_0 = Vector512<float>.Zero;
            c6_1 = Vector512<float>.Zero;
            c7_0 = Vector512<float>.Zero;
            c7_1 = Vector512<float>.Zero;
            c8_0 = Vector512<float>.Zero;
            c8_1 = Vector512<float>.Zero;
            c9_0 = Vector512<float>.Zero;
            c9_1 = Vector512<float>.Zero;
            c10_0 = Vector512<float>.Zero;
            c10_1 = Vector512<float>.Zero;
            c11_0 = Vector512<float>.Zero;
            c11_1 = Vector512<float>.Zero;
        }
        for (int kk = 0; kk < k; kk++)
        {
            float* bk = b + offsets[kk];
            var b0 = Vector512.Load(bk);
            var b1 = Vector512.Load(bk + 16);
            var a0 = Vector512.Create(a[0]);
            c0_0 = Avx512F.FusedMultiplyAdd(a0, b0, c0_0);
            c0_1 = Avx512F.FusedMultiplyAdd(a0, b1, c0_1);
            var a1 = Vector512.Create(a[1]);
            c1_0 = Avx512F.FusedMultiplyAdd(a1, b0, c1_0);
            c1_1 = Avx512F.FusedMultiplyAdd(a1, b1, c1_1);
            var a2 = Vector512.Create(a[2]);
            c2_0 = Avx512F.FusedMultiplyAdd(a2, b0, c2_0);
            c2_1 = Avx512F.FusedMultiplyAdd(a2, b1, c2_1);
            var a3 = Vector512.Create(a[3]);
            c3_0 = Avx512F.FusedMultiplyAdd(a3, b0, c3_0);
            c3_1 = Avx512F.FusedMultiplyAdd(a3, b1, c3_1);
            var a4 = Vector512.Create(a[4]);
            c4_0 = Avx512F.FusedMultiplyAdd(a4, b0, c4_0);
            c4_1 = Avx512F.FusedMultiplyAdd(a4, b1, c4_1);
            var a5 = Vector512.Create(a[5]);
            c5_0 = Avx512F.FusedMultiplyAdd(a5, b0, c5_0);
            c5_1 = Avx512F.FusedMultiplyAdd(a5, b1, c5_1);
            var a6 = Vector512.Create(a[6]);
            c6_0 = Avx512F.FusedMultiplyAdd(a6, b0, c6_0);
            c6_1 = Avx512F.FusedMultiplyAdd(a6, b1, c6_1);
            var a7 = Vector512.Create(a[7]);
            c7_0 = Avx512F.FusedMultiplyAdd(a7, b0, c7_0);
            c7_1 = Avx512F.FusedMultiplyAdd(a7, b1, c7_1);
            var a8 = Vector512.Create(a[8]);
            c8_0 = Avx512F.FusedMultiplyAdd(a8, b0, c8_0);
            c8_1 = Avx512F.FusedMultiplyAdd(a8, b1, c8_1);
            var a9 = Vector512.Create(a[9]);
            c9_0 = Avx512F.FusedMultiplyAdd(a9, b0, c9_0);
            c9_1 = Avx512F.FusedMultiplyAdd(a9, b1, c9_1);
            var a10 = Vector512.Create(a[10]);
            c10_0 = Avx512F.FusedMultiplyAdd(a10, b0, c10_0);
            c10_1 = Avx512F.FusedMultiplyAdd(a10, b1, c10_1);
            var a11 = Vector512.Create(a[11]);
            c11_0 = Avx512F.FusedMultiplyAdd(a11, b0, c11_0);
            c11_1 = Avx512F.FusedMultiplyAdd(a11, b1, c11_1);
            a += 12;
        }
        Vector512.Store(c0_0, c);
        Vector512.Store(c0_1, c + 16);
        Vector512.Store(c1_0, c + ldc);
        Vector512.Store(c1_1, c + ldc + 16);
        Vector512.Store(c2_0, c + 2 * ldc);
        Vector512.Store(c2_1, c + 2 * ldc + 16);
        Vector512.Store(c3_0, c + 3 * ldc);
        Vector512.Store(c3_1, c + 3 * ldc + 16);
        Vector512.Store(c4_0, c + 4 * ldc);
        Vector512.Store(c4_1, c + 4 * ldc + 16);
        Vector512.Store(c5_0, c + 5 * ldc);
        Vector512.Store(c5_1, c + 5 * ldc + 16);
        Vector512.Store(c6_0, c + 6 * ldc);
        Vector512.Store(c6_1, c + 6 * ldc + 16);
        Vector512.Store(c7_0, c + 7 * ldc);
        Vector512.Store(c7_1, c + 7 * ldc + 16);
        Vector512.Store(c8_0, c + 8 * ldc);
        Vector512.Store(c8_1, c + 8 * ldc + 16);
        Vector512.Store(c9_0, c + 9 * ldc);
        Vector512.Store(c9_1, c + 9 * ldc + 16);
        Vector512.Store(c10_0, c + 10 * ldc);
        Vector512.Store(c10_1, c + 10 * ldc + 16);
        Vector512.Store(c11_0, c + 11 * ldc);
        Vector512.Store(c11_1, c + 11 * ldc + 16);
    }

    /// <summary>Kernel6x16Avx2 with B read in place: step kk of k reads the 16 values at b + offsets[kk].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Direct6x16Avx2(int k, float* a, float* b, int* offsets, float* c, nint ldc, float* rowBias)
    {
        Vector256<float> c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
        if (rowBias != null)
        {
            c0_0 = Vector256.Create(rowBias[0]);
            c0_1 = c0_0;
            c1_0 = Vector256.Create(rowBias[1]);
            c1_1 = c1_0;
            c2_0 = Vector256.Create(rowBias[2]);
            c2_1 = c2_0;
            c3_0 = Vector256.Create(rowBias[3]);
            c3_1 = c3_0;
            c4_0 = Vector256.Create(rowBias[4]);
            c4_1 = c4_0;
            c5_0 = Vector256.Create(rowBias[5]);
            c5_1 = c5_0;
        }
        else
        {
            c0_0 = Vector256<float>.Zero;
            c0_1 = Vector256<float>.Zero;
            c1_0 = Vector256<float>.Zero;
            c1_1 = Vector256<float>.Zero;
            c2_0 = Vector256<float>.Zero;
            c2_1 = Vector256<float>.Zero;
            c3_0 = Vector256<float>.Zero;
            c3_1 = Vector256<float>.Zero;
            c4_0 = Vector256<float>.Zero;
            c4_1 = Vector256<float>.Zero;
            c5_0 = Vector256<float>.Zero;
            c5_1 = Vector256<float>.Zero;
        }
        for (int kk = 0; kk < k; kk++)
        {
            float* bk = b + offsets[kk];
            var b0 = Vector256.Load(bk);
            var b1 = Vector256.Load(bk + 8);
            var a0 = Vector256.Create(a[0]);
            c0_0 = Fma.MultiplyAdd(a0, b0, c0_0);
            c0_1 = Fma.MultiplyAdd(a0, b1, c0_1);
            var a1 = Vector256.Create(a[1]);
            c1_0 = Fma.MultiplyAdd(a1, b0, c1_0);
            c1_1 = Fma.MultiplyAdd(a1, b1, c1_1);
            var a2 = Vector256.Create(a[2]);
            c2_0 = Fma.MultiplyAdd(a2, b0, c2_0);
            c2_1 = Fma.MultiplyAdd(a2, b1, c2_1);
            var a3 = Vector256.Create(a[3]);
            c3_0 = Fma.MultiplyAdd(a3, b0, c3_0);
            c3_1 = Fma.MultiplyAdd(a3, b1, c3_1);
            var a4 = Vector256.Create(a[4]);
            c4_0 = Fma.MultiplyAdd(a4, b0, c4_0);
            c4_1 = Fma.MultiplyAdd(a4, b1, c4_1);
            var a5 = Vector256.Create(a[5]);
            c5_0 = Fma.MultiplyAdd(a5, b0, c5_0);
            c5_1 = Fma.MultiplyAdd(a5, b1, c5_1);
            a += 6;
        }
        Vector256.Store(c0_0, c);
        Vector256.Store(c0_1, c + 8);
        Vector256.Store(c1_0, c + ldc);
        Vector256.Store(c1_1, c + ldc + 8);
        Vector256.Store(c2_0, c + 2 * ldc);
        Vector256.Store(c2_1, c + 2 * ldc + 8);
        Vector256.Store(c3_0, c + 3 * ldc);
        Vector256.Store(c3_1, c + 3 * ldc + 8);
        Vector256.Store(c4_0, c + 4 * ldc);
        Vector256.Store(c4_1, c + 4 * ldc + 8);
        Vector256.Store(c5_0, c + 5 * ldc);
        Vector256.Store(c5_1, c + 5 * ldc + 8);
    }

    /// <summary>Kernel6x8Vector128 with B read in place: step kk of k reads the 8 values at b + offsets[kk].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Direct6x8Vector128(int k, float* a, float* b, int* offsets, float* c, nint ldc, float* rowBias)
    {
        Vector128<float> c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;
        if (rowBias != null)
        {
            c0_0 = Vector128.Create(rowBias[0]);
            c0_1 = c0_0;
            c1_0 = Vector128.Create(rowBias[1]);
            c1_1 = c1_0;
            c2_0 = Vector128.Create(rowBias[2]);
            c2_1 = c2_0;
            c3_0 = Vector128.Create(rowBias[3]);
            c3_1 = c3_0;
            c4_0 = Vector128.Create(rowBias[4]);
            c4_1 = c4_0;
            c5_0 = Vector128.Create(rowBias[5]);
            c5_1 = c5_0;
        }
        else
        {
            c0_0 = Vector128<float>.Zero;
            c0_1 = Vector128<float>.Zero;
            c1_0 = Vector128<float>.Zero;
            c1_1 = Vector128<float>.Zero;
            c2_0 = Vector128<float>.Zero;
            c2_1 = Vector128<float>.Zero;
            c3_0 = Vector128<float>.Zero;
            c3_1 = Vector128<float>.Zero;
            c4_0 = Vector128<float>.Zero;
            c4_1 = Vector128<float>.Zero;
            c5_0 = Vector128<float>.Zero;
            c5_1 = Vector128<float>.Zero;
        }
        for (int kk = 0; kk < k; kk++)
        {
            float* bk = b + offsets[kk];
            var b0 = Vector128.Load(bk);
            var b1 = Vector128.Load(bk + 4);
            var a0 = Vector128.Create(a[0]);
            c0_0 = MultiplyAdd(a0, b0, c0_0);
            c0_1 = MultiplyAdd(a0, b1, c0_1);
            var a1 = Vector128.Create(a[1]);
            c1_0 = MultiplyAdd(a1, b0, c1_0);
            c1_1 = MultiplyAdd(a1, b1, c1_1);
            var a2 = Vector128.Create(a[2]);
            c2_0 = MultiplyAdd(a2, b0, c2_0);
            c2_1 = MultiplyAdd(a2, b1, c2_1);
            var a3 = Vector128.Create(a[3]);
            c3_0 = MultiplyAdd(a3, b0, c3_0);
            c3_1 = MultiplyAdd(a3, b1, c3_1);
            var a4 = Vector128.Create(a[4]);
            c4_0 = MultiplyAdd(a4, b0, c4_0);
            c4_1 = MultiplyAdd(a4, b1, c4_1);
            var a5 = Vector128.Create(a[5]);
            c5_0 = MultiplyAdd(a5, b0, c5_0);
            c5_1 = MultiplyAdd(a5, b1, c5_1);
            a += 6;
        }
        Vector128.Store(c0_0, c);
        Vector128.Store(c0_1, c + 4);
        Vector128.Store(c1_0, c + ldc);
        Vector128.Store(c1_1, c + ldc + 4);
        Vector128.Store(c2_0, c + 2 * ldc);
        Vector128.Store(c2_1, c + 2 * ldc + 4);
        Vector128.Store(c3_0, c + 3 * ldc);
        Vector128.Store(c3_1, c + 3 * ldc + 4);
        Vector128.Store(c4_0, c + 4 * ldc);
        Vector128.Store(c4_1, c + 4 * ldc + 4);
        Vector128.Store(c5_0, c + 5 * ldc);
        Vector128.Store(c5_1, c + 5 * ldc + 4);
    }
}

/// <summary>
/// What a matrix product does to its result: an activation, then a residual added. The residual is laid out like C
/// (same leading dimension) and is read at the same offsets. Each value is treated alone, so the result does not
/// depend on the blocks it is applied to.
/// </summary>
internal readonly unsafe struct Epilogue(ActivationKind activation, float alpha, float beta, float* residual, nint residualStride = 0)
{
    public ActivationKind Activation { get; } = activation;

    public float Alpha { get; } = alpha;

    public float Beta { get; } = beta;

    /// <summary>The residual at the position of C's first value, or null.</summary>
    public float* Residual { get; } = residual;

    /// <summary>The floats between the residual's rows, or 0 when its rows lie as far apart as C's.</summary>
    public nint ResidualStride { get; } = residualStride;

    public bool IsEmpty => Activation == ActivationKind.None && Residual == null;

    /// <summary>Applies the epilogue to a rows x cols block at c, whose first row is row <paramref name="row"/> of C.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Apply(float* c, nint ldc, int row, int rows, int cols)
    {
        nint stride = ResidualStride != 0 ? ResidualStride : ldc;
        for (int r = 0; r < rows; r++)
        {
            float* line = c + r * ldc;
            if (Activation != ActivationKind.None) VectorMath.Activate(line, cols, Activation, Alpha, Beta);
            if (Residual != null) VectorMath.AddInPlace(line, Residual + (row + r) * stride, cols);
        }
    }
}
