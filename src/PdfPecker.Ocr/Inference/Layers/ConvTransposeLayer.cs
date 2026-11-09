using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// A transposed convolution whose kernel equals its stride, as in the detector's upsampling head. Output pixels do not
/// overlap, so it is one matrix product, weights [(Cout, dy, dx), Cin] times the input [Cin, H * W], whose rows are
/// scattered to the output positions (y * kh + dy, x * kw + dx).
/// </summary>
internal sealed unsafe class ConvTransposeLayer : Layer
{
    private readonly GemmKernel _gemm;
    private readonly float[] _weights;
    private readonly float[] _bias;
    private readonly int _kh, _kw, _cin, _cout, _rows;
    private readonly ActivationKind _activation;
    private readonly float _alpha, _beta;

    public ConvTransposeLayer(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
        : base(node)
    {
        _gemm = gemm;
        _kh = Int(node, 0);
        _kw = Int(node, 1);
        _activation = (ActivationKind)Int(node, 2);
        _alpha = Float(node, 0);
        _beta = Float(node, 1);
        var w = Weight(node, model, 0);
        if (w.Shape.Length != 4 || w.Shape[2] != _kh || w.Shape[3] != _kw || _kh < 1 || _kw < 1)
            throw new InvalidDataException("Transposed convolution weights do not match the kernel size.");
        _cin = w.Shape[0];
        _cout = w.Shape[1];
        _rows = _cout * _kh * _kw;
        var matrix = new float[_rows * _cin];
        for (int ci = 0; ci < _cin; ci++)
            for (int r = 0; r < _rows; r++)
                matrix[r * _cin + ci] = w.Data[ci * _rows + r];
        _weights = gemm.PackA(matrix, _rows, _cin);
        var b = Weight(node, model, 1).Data;
        if (b.Length != _cout) throw new InvalidDataException("Transposed convolution bias does not match.");
        var rowBias = new float[_rows];
        for (int r = 0; r < _rows; r++) rowBias[r] = b[r / (_kh * _kw)];
        _bias = gemm.PadBias(rowBias, _rows);
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        RequireRank(x, 4, "Transposed convolution");
        if (x[1] != _cin) throw Unsupported($"Transposed convolution expects {_cin} channels, got {x[1]}.");
        return [x[0], _cout, checked(x[2] * _kh), checked(x[3] * _kw)];
    }

    private int TileColumns(int columns)
    {
        int nr = _gemm.NR;
        return Tiling.Balanced(columns, Math.Clamp(16384 / Math.Max(_cin, _rows) / nr * nr, nr, 1024), nr);
    }

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int nc = TileColumns(step.InputShapes[0][2] * step.InputShapes[0][3]);
        return _gemm.PackedBSize(_cin, nc) + (long)GemmKernel.RoundUp(_rows, _gemm.MR) * nc;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int columns = step.InputShapes[0][2] * step.InputShapes[0][3];
        step.Tiling ??= Tiling.For(step.InputShapes[0][0], columns, TileColumns(columns), _rows, _gemm.MR);
        context.For(step.Tiling.Count, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (ConvTransposeLayer)step.Layer;
        var gemm = layer._gemm;
        step.Tiling!.Decode(tile, out int n, out int n0, out int columns, out int m0, out int rows);
        int h = step.InputShapes[0][2], w = step.InputShapes[0][3];
        int plane = h * w;
        float* packed = step.Context.Scratch(worker);
        float* result = packed + gemm.PackedBSize(layer._cin, step.Tiling.TileColumns);
        gemm.PackBRows(step.Input(0) + (long)n * layer._cin * plane + n0, plane, layer._cin, columns, packed);
        fixed (float* weights = layer._weights, bias = layer._bias)
            gemm.Multiply(layer._cin, weights + (long)m0 * layer._cin, rows, packed, columns, result, columns, bias + m0, null);
        int kh = layer._kh, kw = layer._kw, taps = kh * kw;
        int ow = w * kw;
        float* output = step.Output + (long)n * layer._cout * plane * taps;
        for (int r = 0; r < rows; r++) VectorMath.Activate(result + (long)r * columns, columns, layer._activation, layer._alpha, layer._beta);
        for (int r = 0; r < rows; r++)
        {
            float* src = result + (long)r * columns;
            int row = m0 + r;
            int co = row / taps, dy = row % taps / kw, dx = row % kw;
            float* outPlane = output + (long)co * plane * taps + dx;

            // With a kernel two wide, the rows of dx = 0 and 1 follow each other (tiles start at a multiple of the
            // panel height, which is even) and are interleaved into the output rows together.
            bool pair = kw == 2;
            int y = n0 / w, x = n0 - y * w;
            for (int j = 0; j < columns; y++, x = 0)
            {
                int run = Math.Min(w - x, columns - j);
                float* dst = outPlane + (long)(y * kh + dy) * ow + (long)x * kw;
                if (pair) Interleave(src + j, src + columns + j, dst, run);
                else for (int t = 0; t < run; t++) dst[t * kw] = src[j + t];
                j += run;
            }

            if (pair) r++;
        }
    }

    /// <summary>dst[2 t] = a[t] and dst[2 t + 1] = b[t] for t in [0, count).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Interleave(float* a, float* b, float* dst, int count)
    {
        int t = 0;
        if (Avx512F.IsSupported)
        {
            var low = Vector512.Create(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            var high = Vector512.Create(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            for (; t + 16 <= count; t += 16)
            {
                var va = Avx512F.LoadVector512(a + t);
                var vb = Avx512F.LoadVector512(b + t);
                Avx512F.Store(dst + (2 * t), Avx512F.PermuteVar16x32x2(va, low, vb));
                Avx512F.Store(dst + (2 * t) + 16, Avx512F.PermuteVar16x32x2(va, high, vb));
            }
        }
        else if (Avx.IsSupported)
        {
            for (; t + 8 <= count; t += 8)
            {
                // Within each 128-bit half: a0 b0 a1 b1 | a4 b4 a5 b5 and a2 b2 a3 b3 | a6 b6 a7 b7; the halves
                // then go in order.
                var va = Avx.LoadVector256(a + t);
                var vb = Avx.LoadVector256(b + t);
                var low = Avx.UnpackLow(va, vb);
                var high = Avx.UnpackHigh(va, vb);
                Avx.Store(dst + (2 * t), Avx.Permute2x128(low, high, 0x20));
                Avx.Store(dst + (2 * t) + 8, Avx.Permute2x128(low, high, 0x31));
            }
        }
        else if (AdvSimd.Arm64.IsSupported)
        {
            for (; t + 4 <= count; t += 4)
            {
                var va = AdvSimd.LoadVector128(a + t);
                var vb = AdvSimd.LoadVector128(b + t);
                AdvSimd.Store(dst + (2 * t), AdvSimd.Arm64.ZipLow(va, vb));
                AdvSimd.Store(dst + (2 * t) + 4, AdvSimd.Arm64.ZipHigh(va, vb));
            }
        }

        for (; t < count; t++)
        {
            dst[2 * t] = a[t];
            dst[(2 * t) + 1] = b[t];
        }
    }
}
