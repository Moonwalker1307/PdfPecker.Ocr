using System.Numerics;
using System.Runtime.CompilerServices;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// The convolution attributes shared by the matrix-multiplication and the depthwise implementations: NCHW input,
/// weights [Cout, Cin / group, kh, kw], bias, a fused activation and an optional residual added after it.
/// </summary>
internal abstract class ConvLayer : Layer
{
    protected ConvLayer(NodeDefinition node, ModelDefinition model)
        : base(node)
    {
        Kh = Int(node, 0);
        Kw = Int(node, 1);
        Sh = Int(node, 2);
        Sw = Int(node, 3);
        Dh = Int(node, 4);
        Dw = Int(node, 5);
        Pt = Int(node, 6);
        Pl = Int(node, 7);
        Pb = Int(node, 8);
        Pr = Int(node, 9);
        Group = Int(node, 10);
        Activation = (ActivationKind)Int(node, 11);
        HasResidual = Int(node, 12) != 0;
        Alpha = Float(node, 0);
        Beta = Float(node, 1);
        var w = Weight(node, model, 0);
        if (w.Shape.Length != 4 || w.Shape[2] != Kh || w.Shape[3] != Kw) throw new InvalidDataException("Convolution weights do not match the kernel size.");
        Cout = w.Shape[0];
        CinPerGroup = w.Shape[1];
        if (Kh < 1 || Kw < 1 || Sh < 1 || Sw < 1 || Dh < 1 || Dw < 1 || Group < 1 || Pt < 0 || Pl < 0 || Pb < 0 || Pr < 0 || Cout % Group != 0)
            throw new InvalidDataException("Bad convolution attributes.");
        if (Weight(node, model, 1).Data.Length != Cout) throw new InvalidDataException("Convolution bias does not match.");
        if (node.Inputs.Length != (HasResidual ? 2 : 1)) throw new InvalidDataException("Convolution has the wrong number of inputs.");
    }

    protected int Kh { get; }
    protected int Kw { get; }
    protected int Sh { get; }
    protected int Sw { get; }
    protected int Dh { get; }
    protected int Dw { get; }
    protected int Pt { get; }
    protected int Pl { get; }
    protected int Pb { get; }
    protected int Pr { get; }
    protected int Group { get; }
    protected int Cout { get; }
    protected int CinPerGroup { get; }
    protected ActivationKind Activation { get; }
    protected float Alpha { get; }
    protected float Beta { get; }
    protected bool HasResidual { get; }

    public static Layer CreateConv(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
    {
        int group = Int(node, 10);
        var w = Weight(node, model, 0);
        if (group == 1) return new GemmConvLayer(node, model, gemm);
        if (w.Shape.Length == 4 && w.Shape[1] == 1 && w.Shape[0] == group) return new DepthwiseConvLayer(node, model);
        throw new InvalidDataException("Grouped convolutions other than depthwise are not supported.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        RequireRank(x, 4, "Convolution");
        if (x[1] != CinPerGroup * Group) throw Unsupported($"Convolution expects {CinPerGroup * Group} channels, got {x[1]}.");
        int oh = (x[2] + Pt + Pb - Dh * (Kh - 1) - 1) / Sh + 1;
        int ow = (x[3] + Pl + Pr - Dw * (Kw - 1) - 1) / Sw + 1;
        if (oh < 1 || ow < 1) throw Unsupported("The input is smaller than the convolution kernel.");
        int[] output = [x[0], Cout, oh, ow];
        if (HasResidual && !inputs[1].AsSpan().SequenceEqual(output)) throw Unsupported("The residual does not match the convolution output.");
        return output;
    }
}

/// <summary>
/// A convolution without groups as a matrix product: weights [Cout, Cin * kh * kw] times the input arranged as
/// [Cin * kh * kw, OH * OW]. A 1x1 convolution packs input rows straight from the NCHW planes; others gather the
/// patches (im2col) into the packed panels of one tile at a time, so the full im2col matrix never exists.
/// </summary>
internal sealed unsafe class GemmConvLayer : ConvLayer
{
    private readonly GemmKernel _gemm;
    private readonly float[] _weights;
    private readonly float[] _bias;
    private readonly bool _pointwise;
    private readonly int _k;

    public GemmConvLayer(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
        : base(node, model)
    {
        _gemm = gemm;
        _k = CinPerGroup * Kh * Kw;
        _weights = gemm.PackA(Weight(node, model, 0).Data, Cout, _k);
        _bias = gemm.PadBias(Weight(node, model, 1).Data, Cout);
        _pointwise = Kh == 1 && Kw == 1 && Sh == 1 && Sw == 1 && Pt == 0 && Pl == 0 && Pb == 0 && Pr == 0;
    }

    /// <summary>The columns of one tile: the packed tile stays around a quarter of a megabyte.</summary>
    private int TileColumns(int columns)
    {
        int nr = _gemm.NR;
        return Tiling.Balanced(columns, Math.Clamp(65536 / _k / nr * nr, nr, 2048), nr);
    }

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int columns = step.OutputShape[2] * step.OutputShape[3];
        return _gemm.PackedBSize(_k, TileColumns(columns));
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int columns = step.OutputShape[2] * step.OutputShape[3];
        step.Tiling ??= Tiling.For(step.OutputShape[0], columns, TileColumns(columns), Cout, _gemm.MR);
        context.For(step.Tiling.Count, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (GemmConvLayer)step.Layer;
        var tiling = step.Tiling!;
        var gemm = layer._gemm;
        tiling.Decode(tile, out int n, out int n0, out int columns, out int m0, out int rows);
        int[] x = step.InputShapes[0];
        int oh = step.OutputShape[2], ow = step.OutputShape[3];
        int planeOut = oh * ow;
        float* input = step.Input(0) + (long)n * x[1] * x[2] * x[3];
        float* packed = step.Context.Scratch(worker);
        if (layer._pointwise) gemm.PackBRows(input + n0, planeOut, layer._k, columns, packed);
        else layer.PackPatches(input, x[1], x[2], x[3], ow, n0, columns, packed);
        float* output = step.Output;
        long baseOffset = (long)n * layer.Cout * planeOut + (long)m0 * planeOut + n0;
        var epilogue = new Epilogue(layer.Activation, layer.Alpha, layer.Beta, layer.HasResidual ? step.Input(1) + baseOffset : null);
        fixed (float* weights = layer._weights, bias = layer._bias)
            gemm.Multiply(layer._k, weights + (long)m0 * layer._k, rows, packed, columns, output + baseOffset, planeOut, bias + m0, null, epilogue);
    }

    /// <summary>
    /// Gathers the patches of output pixels [n0, n0 + columns) into NR-column panels: row (ci, ky, kx) of a panel holds
    /// the input value each output pixel's kernel tap reads, or zero in the padding. Panels are filled one after the
    /// other, so the writes are sequential; within a panel, the pixels of one output row read one run of an input row
    /// per tap, contiguous for stride 1 and every Sw-th value otherwise.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void PackPatches(float* input, int channels, int h, int w, int ow, int n0, int columns, float* packed)
    {
        int nr = _gemm.NR;
        int k = _k;
        // A panel's pixels split into at most nr runs that each lie in one output row.
        int* runStart = stackalloc int[nr];
        int* runLength = stackalloc int[nr];
        int* runY = stackalloc int[nr];
        int* runX = stackalloc int[nr];
        for (int j0 = 0; j0 < columns; j0 += nr)
        {
            int width = Math.Min(nr, columns - j0);
            int runs = 0;
            for (int t = 0; t < width;)
            {
                int n = n0 + j0 + t;
                int oy = n / ow, ox = n - oy * ow;
                int length = Math.Min(ow - ox, width - t);
                runStart[runs] = t;
                runLength[runs] = length;
                runY[runs] = oy * Sh - Pt;
                runX[runs] = ox * Sw - Pl;
                runs++;
                t += length;
            }
            float* dst = packed + (long)j0 * k;
            for (int ci = 0; ci < channels; ci++)
            {
                float* plane = input + (long)ci * h * w;
                for (int ky = 0; ky < Kh; ky++)
                {
                    for (int kx = 0; kx < Kw; kx++, dst += nr)
                    {
                        for (int r = 0; r < runs; r++)
                        {
                            float* d = dst + runStart[r];
                            int length = runLength[r];
                            int iy = runY[r] + ky * Dh;
                            if ((uint)iy >= (uint)h)
                            {
                                new Span<float>(d, length).Clear();
                                continue;
                            }
                            float* row = plane + (long)iy * w;
                            int ix = runX[r] + kx * Dw;
                            if (Sw == 1 && ix >= 0 && ix + length <= w)
                            {
                                Unsafe.CopyBlockUnaligned(d, row + ix, (uint)(length * sizeof(float)));
                            }
                            else if (ix >= 0 && ix + (length - 1) * Sw < w)
                            {
                                float* s = row + ix;
                                for (int t = 0; t < length; t++, s += Sw) d[t] = *s;
                            }
                            else
                            {
                                for (int t = 0; t < length; t++, ix += Sw) d[t] = (uint)ix < (uint)w ? row[ix] : 0f;
                            }
                        }
                        if (width < nr) new Span<float>(dst + width, nr - width).Clear();
                    }
                }
            }
        }
    }
}

/// <summary>
/// A depthwise convolution (one filter per channel), computed directly: each tile pads a band of input rows into
/// scratch, de-interleaving them for a horizontal stride of 2, and accumulates the kernel taps over vectors of output
/// pixels.
/// </summary>
internal sealed unsafe class DepthwiseConvLayer : ConvLayer
{
    private static readonly int Width = Vector<float>.Count;
    private readonly float[] _weights;
    private readonly float[] _bias;

    public DepthwiseConvLayer(NodeDefinition node, ModelDefinition model)
        : base(node, model)
    {
        if (Sw > 2 || Dh != 1 || Dw != 1) throw new InvalidDataException("Depthwise convolutions need dilation 1 and a horizontal stride of 1 or 2.");
        _weights = Weight(node, model, 0).Data;
        _bias = Weight(node, model, 1).Data;
    }

    private static int BandRows(int[] output)
    {
        // Bands only split planes when there are too few planes to keep the threads busy.
        int planes = output[0] * output[1];
        return planes >= 64 ? output[2] : Math.Max(1, Math.Min(output[2], 16384 / Math.Max(1, output[3])));
    }

    /// <summary>The length of one padded input row in scratch, including vector slack.</summary>
    private int RowLength(int ow) => Sw == 1 ? ow + Kw - 1 + Width : 2 * (ow + (Kw + 1) / 2 + Width);

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int band = BandRows(step.OutputShape);
        int rows = (band - 1) * Sh + Kh;
        return (long)rows * RowLength(step.OutputShape[3]) + Width;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int[] o = step.OutputShape;
        int band = BandRows(o);
        int bands = (o[2] + band - 1) / band;
        context.For(o[0] * o[1] * bands, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (DepthwiseConvLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int[] o = step.OutputShape;
        int band = BandRows(o);
        int bands = (o[2] + band - 1) / band;
        int plane = tile / bands;
        int y0 = tile % bands * band;
        int y1 = Math.Min(o[2], y0 + band);
        int c = plane % o[1];
        layer.Compute(step, (long)plane * x[2] * x[3], (long)plane * o[2] * o[3], c, x[2], x[3], o[3], y0, y1, step.Context.Scratch(worker));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Compute(Step step, long inOffset, long outOffset, int c, int h, int w, int ow, int y0, int y1, float* scratch)
    {
        float* input = step.Input(0) + inOffset;
        float* output = step.Output + outOffset;
        int rowLength = RowLength(ow);
        int half = rowLength / 2;
        int iyStart = y0 * Sh - Pt;
        int iyEnd = (y1 - 1) * Sh - Pt + Kh;
        for (int iy = iyStart; iy < iyEnd; iy++)
        {
            float* dst = scratch + (long)(iy - iyStart) * rowLength;
            new Span<float>(dst, rowLength).Clear();
            if ((uint)iy >= (uint)h) continue;
            float* src = input + (long)iy * w;
            if (Sw == 1)
            {
                int count = Math.Min(w, rowLength - Pl);
                new Span<float>(src, count).CopyTo(new Span<float>(dst + Pl, count));
            }
            else
            {
                // Padded position p = ix + Pl goes to the even half at p / 2 or the odd half at half + p / 2.
                for (int ix = 0; ix < w; ix++)
                {
                    int p = ix + Pl;
                    if ((p >> 1) < half) dst[(p & 1) == 0 ? p >> 1 : half + (p >> 1)] = src[ix];
                }
            }
        }

        float* taps = stackalloc float[Kh * Kw];
        fixed (float* weights = _weights) new Span<float>(weights + (long)c * Kh * Kw, Kh * Kw).CopyTo(new Span<float>(taps, Kh * Kw));
        var bias = new Vector<float>(_bias[c]);
        float* tail = stackalloc float[Width];
        for (int y = y0; y < y1; y++)
        {
            float* outRow = output + (long)y * ow;
            float* rows = scratch + (long)(y * Sh - Pt - iyStart) * rowLength;
            int ox = 0;
            // Four vectors of pixels at a time: four independent sums hide the latency of the additions, and each tap is
            // broadcast once for all four. Every pixel still sums its taps in the same order as in the loop below.
            for (; ox + 4 * Width <= ow; ox += 4 * Width)
            {
                Vector<float> a0 = bias, a1 = bias, a2 = bias, a3 = bias;
                for (int ky = 0; ky < Kh; ky++)
                {
                    float* row = rows + (long)ky * rowLength;
                    float* tap = taps + ky * Kw;
                    for (int kx = 0; kx < Kw; kx++)
                    {
                        float* src = Sw == 1 ? row + ox + kx : row + ((kx & 1) == 0 ? 0 : half) + ox + (kx >> 1);
                        var t = new Vector<float>(tap[kx]);
                        a0 += t * VectorMath.Load(src);
                        a1 += t * VectorMath.Load(src + Width);
                        a2 += t * VectorMath.Load(src + 2 * Width);
                        a3 += t * VectorMath.Load(src + 3 * Width);
                    }
                }
                VectorMath.Store(VectorMath.Activate(a0, Activation, Alpha, Beta), outRow + ox);
                VectorMath.Store(VectorMath.Activate(a1, Activation, Alpha, Beta), outRow + ox + Width);
                VectorMath.Store(VectorMath.Activate(a2, Activation, Alpha, Beta), outRow + ox + 2 * Width);
                VectorMath.Store(VectorMath.Activate(a3, Activation, Alpha, Beta), outRow + ox + 3 * Width);
            }
            for (; ox < ow; ox += Width)
            {
                var acc = bias;
                for (int ky = 0; ky < Kh; ky++)
                {
                    float* row = rows + (long)ky * rowLength;
                    float* tap = taps + ky * Kw;
                    if (Sw == 1)
                    {
                        for (int kx = 0; kx < Kw; kx++) acc += new Vector<float>(tap[kx]) * VectorMath.Load(row + ox + kx);
                    }
                    else
                    {
                        for (int kx = 0; kx < Kw; kx++)
                        {
                            float* src = (kx & 1) == 0 ? row + ox + (kx >> 1) : row + half + ox + (kx >> 1);
                            acc += new Vector<float>(tap[kx]) * VectorMath.Load(src);
                        }
                    }
                }
                acc = VectorMath.Activate(acc, Activation, Alpha, Beta);
                if (ox + Width <= ow)
                {
                    VectorMath.Store(acc, outRow + ox);
                }
                else
                {
                    VectorMath.Store(acc, tail);
                    new Span<float>(tail, ow - ox).CopyTo(new Span<float>(outRow + ox, ow - ox));
                }
            }
            if (HasResidual) VectorMath.AddInPlace(outRow, step.Input(1) + outOffset + (long)y * ow, ow);
        }
    }
}

/// <summary>
/// How a matrix-product layer splits into tiles: per batch item, columns in blocks of a fixed width, and rows in
/// blocks of whole panels when there are too few column blocks to keep the threads busy. It depends only on the
/// shapes, so the same values always land in the same kind of tile.
/// </summary>
internal sealed class Tiling
{
    private int _columnBlocks;
    private int _rowBlocks;
    private int _rowBlockPanels;

    public int Batch { get; private set; }
    public int Columns { get; private set; }
    public int TileColumns { get; private set; }
    public int Rows { get; private set; }
    public int Panel { get; private set; }
    public int Count => Batch * _columnBlocks * _rowBlocks;

    /// <summary>
    /// The width of the fewest equal column blocks, in whole panels, that are at most maxColumns wide (a multiple of the
    /// panel width), so that tiles do about the same work and the last one is not a sliver.
    /// </summary>
    public static int Balanced(int columns, int maxColumns, int panel)
    {
        int blocks = (columns + maxColumns - 1) / maxColumns;
        return GemmKernel.RoundUp((columns + blocks - 1) / blocks, panel);
    }

    public static Tiling For(int batch, int columns, int tileColumns, int rows, int panel)
    {
        var t = new Tiling { Batch = batch, Columns = columns, TileColumns = tileColumns, Rows = rows, Panel = panel };
        t._columnBlocks = (columns + tileColumns - 1) / tileColumns;
        int panels = (rows + panel - 1) / panel;
        int wanted = Math.Clamp((16 + batch * t._columnBlocks - 1) / (batch * t._columnBlocks), 1, panels);
        t._rowBlockPanels = (panels + wanted - 1) / wanted;
        t._rowBlocks = (panels + t._rowBlockPanels - 1) / t._rowBlockPanels;
        return t;
    }

    public void Decode(int tile, out int batch, out int column, out int columns, out int row, out int rows)
    {
        int rowBlock = tile % _rowBlocks;
        tile /= _rowBlocks;
        int columnBlock = tile % _columnBlocks;
        batch = tile / _columnBlocks;
        column = columnBlock * TileColumns;
        columns = Math.Min(TileColumns, Columns - column);
        row = rowBlock * _rowBlockPanels * Panel;
        rows = Math.Min(_rowBlockPanels * Panel, Rows - row);
    }
}
