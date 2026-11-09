using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
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

    /// <summary>A copy of another convolution's attributes, for a variant of a loaded layer.</summary>
    protected ConvLayer(ConvLayer source)
        : base(source.Node)
    {
        Kh = source.Kh;
        Kw = source.Kw;
        Sh = source.Sh;
        Sw = source.Sw;
        Dh = source.Dh;
        Dw = source.Dw;
        Pt = source.Pt;
        Pl = source.Pl;
        Pb = source.Pb;
        Pr = source.Pr;
        Group = source.Group;
        Activation = source.Activation;
        HasResidual = source.HasResidual;
        Alpha = source.Alpha;
        Beta = source.Beta;
        Cout = source.Cout;
        CinPerGroup = source.CinPerGroup;
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
    internal bool HasResidual { get; }

    public static Layer CreateConv(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
    {
        int group = Int(node, 10);
        var w = Weight(node, model, 0);
        if (group == 1) return new GemmConvLayer(node, model, gemm);
        if (w.Shape.Length == 4 && w.Shape[1] == 1 && w.Shape[0] == group) return new DepthwiseConvLayer(node, model);
        throw new InvalidDataException("Grouped convolutions other than depthwise are not supported.");
    }

    /// <summary>The input rows [first, end) that output rows [y0, y1) read inside a map of the given height.</summary>
    internal (int First, int End) InputRows(int y0, int y1, int height) =>
        (Math.Max(0, y0 * Sh - Pt), Math.Min(height, (y1 - 1) * Sh - Pt + Dh * (Kh - 1) + 1));

    /// <summary>The input columns [first, end) that output columns [x0, x1) read inside a map of the given width.</summary>
    internal (int First, int End) InputColumns(int x0, int x1, int width) =>
        (Math.Max(0, x0 * Sw - Pl), Math.Min(width, (x1 - 1) * Sw - Pl + Dw * (Kw - 1) + 1));

    /// <summary>The multiply-adds of one output pixel, all channels, for weighing work.</summary>
    internal long MultiplyAddsPerPixel => (long)Cout * CinPerGroup * Kh * Kw;

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
    private readonly bool _broadcast;

    public GemmConvLayer(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
        : base(node, model)
    {
        _gemm = gemm;
        _k = CinPerGroup * Kh * Kw;
        _weights = gemm.PackA(Weight(node, model, 0).Data, Cout, _k);
        _bias = gemm.PadBias(Weight(node, model, 1).Data, Cout);
        _pointwise = Kh == 1 && Kw == 1 && Sh == 1 && Sw == 1 && Pt == 0 && Pl == 0 && Pb == 0 && Pr == 0;
    }

    private GemmConvLayer(GemmConvLayer source)
        : base(source)
    {
        _gemm = source._gemm;
        _k = source._k;
        _weights = source._weights;
        _bias = source._bias;
        _pointwise = source._pointwise;
        _broadcast = true;
    }

    /// <summary>True when <see cref="WithBroadcastInput"/> can make a variant of this layer.</summary>
    public bool CanBroadcastInput => !_pointwise && !_broadcast && !HasResidual;

    /// <summary>
    /// The same convolution on a one-channel input that stands for its every input channel: a gray page's three equal
    /// colour planes. Each tap reads the one plane where it would read its channel's, so the products and their order
    /// are the same and so is every bit of the output, without the two copies of the plane. The weights are shared.
    /// </summary>
    public GemmConvLayer WithBroadcastInput() =>
        CanBroadcastInput ? new GemmConvLayer(this) : throw new InvalidOperationException("This convolution cannot read a broadcast input.");

    public override int[] OutputShape(int[][] inputs)
    {
        if (!_broadcast) return base.OutputShape(inputs);
        var x = inputs[0];
        RequireRank(x, 4, "Convolution");
        if (x[1] != 1) throw Unsupported($"Convolution expects one channel, got {x[1]}.");
        return base.OutputShape([[x[0], CinPerGroup, x[2], x[3]]]);
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
        else layer.PackPatches(input, layer._broadcast ? 0 : (long)x[2] * x[3], x[3], x[2], x[3], 0, 0, ow, 0, 0, n0, columns, packed);
        float* output = step.Output;
        long baseOffset = (long)n * layer.Cout * planeOut + (long)m0 * planeOut + n0;
        var epilogue = new Epilogue(layer.Activation, layer.Alpha, layer.Beta, layer.HasResidual ? step.Input(1) + baseOffset : null);
        fixed (float* weights = layer._weights, bias = layer._bias)
            gemm.Multiply(layer._k, weights + (long)m0 * layer._k, rows, packed, columns, output + baseOffset, planeOut, bias + m0, null, epilogue);
    }

    /// <summary>
    /// Computes every output channel of a block of output pixels, <paramref name="rows"/> rows of
    /// <paramref name="width"/> pixels from output pixel (<paramref name="oy0"/>, <paramref name="ox0"/>), into
    /// <paramref name="output"/>, which holds the block's pixels row after row and its channels
    /// <paramref name="outputPlane"/> floats apart. The input is a window of an h x w map that holds its rows from
    /// <paramref name="firstRow"/> and its columns from <paramref name="firstColumn"/>, <paramref name="rowStride"/>
    /// floats per row and <paramref name="inputPlane"/> per channel; it must hold every pixel inside the map that the
    /// block's taps read, and pixels outside the map are the padding as usual. Each output value is the sum the whole
    /// layer computes for it, in the same order, so a map computed block by block has the bits of one computed at
    /// once. This is how <see cref="StemLayer"/> runs a convolution on bands it keeps in scratch. The packed panels
    /// need <see cref="BlockScratch"/> floats.
    /// </summary>
    internal void ComputeBlock(float* input, long inputPlane, long rowStride, int h, int w, int firstRow, int firstColumn,
        int oy0, int ox0, int rows, int width, float* output, long outputPlane, float* packed)
    {
        if (HasResidual) throw new InvalidOperationException("A convolution with a residual cannot run on a block.");
        int columns = rows * width;
        int tile = TileColumns(columns);
        var epilogue = new Epilogue(Activation, Alpha, Beta, null);
        fixed (float* weights = _weights, bias = _bias)
        {
            for (int n0 = 0; n0 < columns; n0 += tile)
            {
                int count = Math.Min(tile, columns - n0);
                PackPatches(input, _broadcast ? 0 : inputPlane, rowStride, h, w, firstRow, firstColumn, width, oy0, ox0, n0, count, packed);
                _gemm.Multiply(_k, weights, Cout, packed, count, output + n0, (nint)outputPlane, bias, null, epilogue);
            }
        }
    }

    /// <summary>The floats of packed panels <see cref="ComputeBlock"/> needs for a block of the given number of pixels.</summary>
    internal long BlockScratch(int pixels) => _gemm.PackedBSize(_k, TileColumns(pixels));

    /// <summary>
    /// Gathers the patches of output pixels [n0, n0 + columns) of a block into NR-column panels: row (ci, ky, kx) of a
    /// panel holds the input value each output pixel's kernel tap reads, or zero in the padding. The block's pixels go
    /// row by row, <paramref name="bandWidth"/> to a row, from output pixel (<paramref name="oy0"/>, <paramref name="ox0"/>);
    /// for a whole layer the block is the output map. The input holds the rows of an h x w map from
    /// <paramref name="firstRow"/> and its columns from <paramref name="firstColumn"/>, <paramref name="rowStride"/>
    /// floats apart, channels <paramref name="planeSize"/> apart (0 for a broadcast plane). Panels are filled one after
    /// the other. A panel's pixels split into runs that each lie in one output row (usually one run for the whole
    /// panel), and each run fills its part of every row of the panel from one run of an input row per tap, contiguous
    /// for stride 1 and every Sw-th value otherwise. Looping over the taps inside a run keeps the run's position in
    /// registers: the bookkeeping per tap, not the copying, was most of the packing time.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void PackPatches(float* input, long planeSize, long rowStride, int h, int w, int firstRow, int firstColumn, int bandWidth,
        int oy0, int ox0, int n0, int columns, float* packed)
    {
        int nr = _gemm.NR;
        int k = _k;
        int channels = CinPerGroup;
        int kh = Kh, kw = Kw, sw = Sw, dh = Dh, dw = Dw;
        // The map's first channel as if the window held it whole; only rows and columns the window holds are read.
        float* origin = input - (long)firstRow * rowStride - firstColumn;
        for (int j0 = 0; j0 < columns; j0 += nr)
        {
            int width = Math.Min(nr, columns - j0);
            float* panel = packed + (long)j0 * k;
            for (int t = 0; t < width;)
            {
                int n = n0 + j0 + t;
                int ly = n / bandWidth, lx = n - ly * bandWidth;
                int length = Math.Min(bandWidth - lx, width - t);
                int y0 = (oy0 + ly) * Sh - Pt, x0 = (ox0 + lx) * sw - Pl;
                float* dst = panel + t;
                float* plane = origin;
                for (int ci = 0; ci < channels; ci++, plane += planeSize)
                {
                    for (int ky = 0, iy = y0; ky < kh; ky++, iy += dh)
                    {
                        if ((uint)iy >= (uint)h)
                        {
                            for (int kx = 0; kx < kw; kx++, dst += nr) new Span<float>(dst, length).Clear();
                            continue;
                        }

                        float* row = plane + (long)iy * rowStride;
                        for (int kx = 0, ix = x0; kx < kw; kx++, ix += dw, dst += nr) CopyTap(row, w, ix, sw, dst, length);
                    }
                }

                t += length;
            }

            if (width < nr)
            {
                for (int r = 0; r < k; r++) new Span<float>(panel + (long)r * nr + width, nr - width).Clear();
            }
        }
    }

    /// <summary>
    /// Writes the values of one tap for a run of <paramref name="length"/> output pixels: row[ix], row[ix + sw], ...,
    /// with zeros where the run reaches into the padding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CopyTap(float* row, int w, int ix, int sw, float* dst, int length)
    {
        if (sw == 1)
        {
            if (ix >= 0 && ix + length <= w)
            {
                CopyRun(row + ix, dst, length);
                return;
            }

            // Zeros before the row's start and after its end, the row's values in between.
            int lead = Math.Clamp(-ix, 0, length);
            int end = Math.Clamp(w - ix, lead, length);
            for (int t = 0; t < lead; t++) dst[t] = 0f;
            if (end > lead) CopyRun(row + ix + lead, dst + lead, end - lead);
            for (int t = end; t < length; t++) dst[t] = 0f;
            return;
        }

        if (ix >= 0 && ix + (length - 1) * sw < w)
        {
            if (sw == 2)
            {
                VectorMath.CopyEven(row + ix, dst, length);
            }
            else
            {
                float* s = row + ix;
                for (int t = 0; t < length; t++, s += sw) dst[t] = *s;
            }

            return;
        }

        for (int t = 0; t < length; t++, ix += sw) dst[t] = (uint)ix < (uint)w ? row[ix] : 0f;
    }

    /// <summary>
    /// Copies a run of at most a panel's width. A whole panel row, the usual run, is two vectors of the widest size the
    /// panel width allows; other runs take whole vectors, the last one overlapping the one before when the length is
    /// not a multiple. Both take far fewer instructions than a call to the block copy at these lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CopyRun(float* src, float* dst, int length)
    {
        if (length == 32 && Vector512.IsHardwareAccelerated)
        {
            Vector512.Store(Vector512.Load(src), dst);
            Vector512.Store(Vector512.Load(src + 16), dst + 16);
        }
        else if (length == 16 && Vector256.IsHardwareAccelerated)
        {
            Vector256.Store(Vector256.Load(src), dst);
            Vector256.Store(Vector256.Load(src + 8), dst + 8);
        }
        else if (length == 8 && Vector128.IsHardwareAccelerated)
        {
            Vector128.Store(Vector128.Load(src), dst);
            Vector128.Store(Vector128.Load(src + 4), dst + 4);
        }
        else if (Vector256.IsHardwareAccelerated && length >= 8)
        {
            int t = 0;
            for (; t <= length - 8; t += 8) Vector256.Store(Vector256.Load(src + t), dst + t);
            if (t < length) Vector256.Store(Vector256.Load(src + length - 8), dst + length - 8);
        }
        else if (Vector128.IsHardwareAccelerated && length >= 4)
        {
            int t = 0;
            for (; t <= length - 4; t += 4) Vector128.Store(Vector128.Load(src + t), dst + t);
            if (t < length) Vector128.Store(Vector128.Load(src + length - 4), dst + length - 4);
        }
        else
        {
            for (int t = 0; t < length; t++) dst[t] = src[t];
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

    // The slack after each padded row, enough for a 512-bit read that starts at the last pixel.
    private const int Slack = 16;
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
    private int RowLength(int ow) => Sw == 1 ? ow + Kw - 1 + Slack : 2 * (ow + (Kw + 1) / 2 + Slack);

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int band = BandRows(step.OutputShape);
        int rows = (band - 1) * Sh + Kh;
        return (long)rows * RowLength(step.OutputShape[3]) + Slack;
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
                // Padded position p = ix + Pl goes to the even half at p / 2 or the odd half at half + p / 2; positions
                // from 2 * half on are never read.
                int limit = Math.Min(w, 2 * half - Pl);
                for (int parity = 0; parity < 2; parity++)
                {
                    int first = (parity + Pl) & 1;
                    if (first < limit) VectorMath.CopyEven(src + first, dst + parity * half + ((first + Pl) >> 1), (limit - first + 1) / 2);
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
            int ox = VectorMath.Wide ? ComputeWide(rows, rowLength, half, taps, _bias[c], outRow, ow) : 0;
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

    /// <summary>
    /// The leading whole 512-bit vectors of one output row, for processors with AVX-512, where Vector&lt;T&gt; stays 256
    /// bits wide; returns how many pixels it did. Each pixel sums its taps in the same order, with the same separate
    /// products and sums, as the loops of <see cref="Compute"/>, and the wide activation matches the narrow one lane
    /// for lane, so the results do not depend on which loop a pixel lands in.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int ComputeWide(float* rows, int rowLength, int half, float* taps, float biasValue, float* outRow, int ow)
    {
        var bias = Vector512.Create(biasValue);
        int ox = 0;
        for (; ox + 64 <= ow; ox += 64)
        {
            Vector512<float> a0 = bias, a1 = bias, a2 = bias, a3 = bias;
            for (int ky = 0; ky < Kh; ky++)
            {
                float* row = rows + (long)ky * rowLength;
                float* tap = taps + ky * Kw;
                for (int kx = 0; kx < Kw; kx++)
                {
                    float* src = Sw == 1 ? row + ox + kx : row + ((kx & 1) == 0 ? 0 : half) + ox + (kx >> 1);
                    var t = Vector512.Create(tap[kx]);
                    a0 += t * Vector512.Load(src);
                    a1 += t * Vector512.Load(src + 16);
                    a2 += t * Vector512.Load(src + 32);
                    a3 += t * Vector512.Load(src + 48);
                }
            }
            Vector512.Store(VectorMath.ActivateWide(a0, Activation, Alpha, Beta), outRow + ox);
            Vector512.Store(VectorMath.ActivateWide(a1, Activation, Alpha, Beta), outRow + ox + 16);
            Vector512.Store(VectorMath.ActivateWide(a2, Activation, Alpha, Beta), outRow + ox + 32);
            Vector512.Store(VectorMath.ActivateWide(a3, Activation, Alpha, Beta), outRow + ox + 48);
        }
        for (; ox + 16 <= ow; ox += 16)
        {
            var acc = bias;
            for (int ky = 0; ky < Kh; ky++)
            {
                float* row = rows + (long)ky * rowLength;
                float* tap = taps + ky * Kw;
                for (int kx = 0; kx < Kw; kx++)
                {
                    float* src = Sw == 1 ? row + ox + kx : row + ((kx & 1) == 0 ? 0 : half) + ox + (kx >> 1);
                    acc += Vector512.Create(tap[kx]) * Vector512.Load(src);
                }
            }
            Vector512.Store(VectorMath.ActivateWide(acc, Activation, Alpha, Beta), outRow + ox);
        }
        return ox;
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
