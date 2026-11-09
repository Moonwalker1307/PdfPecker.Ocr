using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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

    // An output column is in a run when every column its taps read is; the rows of a window do not matter, since every
    // row's columns are equal along the run.
    public override void FindRuns(Step step)
    {
        WindowRuns(step.InputRuns[0], step.OutputRuns, step.OutputShape[3]);
        if (HasResidual) step.OutputRuns.Intersect(step.InputRuns[1]);
    }

    /// <summary>The runs of an output of the given width from those of the input, for this layer or one inside a fused layer.</summary>
    internal void WindowRuns(ColumnRuns input, ColumnRuns output, int outputWidth) =>
        output.FromWindow(input, 3, Kw, Sw, Dw, Pl, outputWidth);
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
    private readonly bool _inPlace;

    // The longest k that a stride-2 convolution reads from split rows (TileSplit). Each step of k reads its own split
    // row, and with many more streams than this the panels read in place were slower than packed ones: the
    // recogniser's 3x3 convolution on 96 channels (k = 864) lost 4% at one thread.
    private const int MaxSplitDepth = 512;

    public GemmConvLayer(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
        : base(node, model)
    {
        _gemm = gemm;
        _k = CinPerGroup * Kh * Kw;
        _weights = gemm.PackA(Weight(node, model, 0).Data, Cout, _k);
        _bias = gemm.PadBias(Weight(node, model, 1).Data, Cout);
        _pointwise = Kh == 1 && Kw == 1 && Sh == 1 && Sw == 1 && Pt == 0 && Pl == 0 && Pb == 0 && Pr == 0;
        _inPlace = !_pointwise && (Sw == 1 || (Sw == 2 && _k <= MaxSplitDepth));
    }

    private GemmConvLayer(GemmConvLayer source)
        : base(source)
    {
        _gemm = source._gemm;
        _k = source._k;
        _weights = source._weights;
        _bias = source._bias;
        _pointwise = source._pointwise;
        _inPlace = source._inPlace;
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
    private int TileColumns(int columns) => Tiling.Balanced(columns, TileBudget, _gemm.NR);

    /// <summary>The most columns of a packed tile: its panels of B then fill about 256 KB.</summary>
    private int TileBudget => Math.Clamp(65536 / _k / _gemm.NR * _gemm.NR, _gemm.NR, 2048);

    // The products of a gathered tile follow its packed panels in the worker's scratch, a cache line further on.
    private const int ProductOffset = 16;

    /// <summary>
    /// Whether the layer leaves out the columns of runs (<see cref="GatheredTile"/>): only with at least 32 values per
    /// output, since the stem's convolution of the three colour planes (k = 27) is cheaper to compute in full.
    /// </summary>
    private bool Gathers => _k >= 32;

    /// <summary>
    /// The fewest columns of a run worth leaving out, for a layer that <see cref="Gathers"/>: a gap splits a panel in
    /// two, whose pieces are copied, which costs about as much per output channel as two thousand multiply-adds, while
    /// a skipped column saves k of them.
    /// </summary>
    private int MinimumGap => Math.Max(2, (2048 + _k - 1) / _k);

    // The output may take the residual's memory: each tile reads the residual only where it writes, and copies that
    // part to its scratch before writing, so a residual block, the largest value of the block, needs no memory of its own.
    public override int InPlaceInput(int[][] inputs, int[] output) => HasResidual ? 1 : -1;

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int[] x = step.InputShapes[0], o = step.OutputShape;
        long work = WorkFloats(x, o);
        // Room for the tile's part of a residual that the output writes over: at most every row by the widest tile's
        // columns.
        return HasResidual ? Align(work) + (long)GemmKernel.RoundUp(Cout, _gemm.MR) * WidestTile(x, o) : work;
    }

    /// <summary>The columns of the step's plain tiles: whole output rows for a stride-2 convolution read in place.</summary>
    private int StepTileColumns(int[] x, int[] output)
    {
        int ow = output[3];
        return ReadsInPlace(x, ow) && Sw == 2 ? SplitTileColumns(ow) : TileColumns(output[2] * ow);
    }

    /// <summary>The most columns of any tile of the step, plain or gathered (<see cref="GatheredTile"/>).</summary>
    private int WidestTile(int[] x, int[] output)
    {
        int plain = StepTileColumns(x, output);
        return !Gathers ? plain : Math.Max(plain, TileColumns(output[2] * output[3]));
    }

    /// <summary>
    /// The floats a worker's tile works in before the copy of a residual: the packed tile, or for panels read in place
    /// one packed panel and, for a stride of 2, the split rows; and for a convolution that may leave out runs, a
    /// gathered tile's packed panels and its block of products.
    /// </summary>
    private long WorkFloats(int[] x, int[] output)
    {
        int ow = output[3];
        int columns = output[2] * ow;
        long work;
        if (!ReadsInPlace(x, ow))
        {
            work = _gemm.PackedBSize(_k, TileColumns(columns));
        }
        else
        {
            long panel = _gemm.PackedBSize(_k, _gemm.NR);
            work = Sw == 1 ? panel : panel + (long)SplitRows * 2 * HalfStride(ow);
        }

        if (Gathers) work = Math.Max(work, _gemm.PackedBSize(_k, TileColumns(columns)) + ProductOffset + ((long)_gemm.MR * _gemm.NR));
        return work;
    }

    /// <summary>
    /// True when the panels of B can be read where they lie (<see cref="TileInPlace"/>): a horizontal stride of 1 or 2,
    /// and tap offsets that fit an int.
    /// </summary>
    private bool ReadsInPlace(int[] x, int ow)
    {
        if (!_inPlace) return false;
        long last = Sw == 1
            ? (CinPerGroup - 1) * (_broadcast ? 0 : (long)x[2] * x[3]) + ((long)(Kh - 1) * Dh * x[3]) + ((long)(Kw - 1) * Dw)
            : (long)SplitRows * 2 * HalfStride(ow);
        return last <= int.MaxValue;
    }

    private static long Align(long floats) => (floats + 15) & ~15L;

    /// <summary>The input rows one output row reads, which a stride-2 tile splits into even and odd halves.</summary>
    private int SplitRows => (_broadcast ? 1 : CinPerGroup) * Kh;

    /// <summary>
    /// The longest run of one output row in a tile of a stride-2 convolution read in place: the whole row when it is
    /// not too long, so that the rows a tile splits serve every panel of the row.
    /// </summary>
    private int SplitRunColumns(int ow) => ow <= 512 ? ow : Tiling.Balanced(ow, 512, _gemm.NR);

    /// <summary>
    /// The tile width of a stride-2 convolution read in place: whole output rows, as many as the packed tiles of the
    /// layer would hold. One row a tile made the recogniser's first convolution (k = 27, rows of 160 to 800 pixels)
    /// slower at four threads than packing, from the number of small tiles.
    /// </summary>
    private int SplitTileColumns(int ow)
    {
        int run = SplitRunColumns(ow);
        return run < ow ? run : ow * Math.Max(1, TileBudget / ow);
    }

    /// <summary>
    /// The floats of each half of a split row: the longest run of pixels, a panel's width that the last panel may read
    /// past it, the taps' reach and slack.
    /// </summary>
    private int HalfStride(int ow) => GemmKernel.RoundUp(SplitRunColumns(ow) + _gemm.NR + ((Kw * Dw) / 2) + 1, 16);

    public override void Execute(Step step, InferenceContext context)
    {
        int ow = step.OutputShape[3];
        int columns = step.OutputShape[2] * ow;
        if (Gathers && step.OutputRuns.Any(3, MinimumGap + 1))
        {
            // Only the columns outside the runs and the first of each run are computed; the fill copies that one over
            // the rest of its run in every row.
            var gathered = context.RunTiling;
            if (gathered.Build(step.OutputRuns, step.OutputShape[2], ow, TileColumns(columns), _gemm.NR, Cout, _gemm.MR, MinimumGap))
            {
                context.For(gathered.Count, step, &GatheredTile);
                step.OutputRuns.FillLastAxis(step.Output, step.OutputShape, context, MinimumGap);
                return;
            }
        }

        step.Tiling ??= Tiling.For(step.OutputShape[0], columns, StepTileColumns(step.InputShapes[0], step.OutputShape), Cout, _gemm.MR);
        if (step.State is null && ReadsInPlace(step.InputShapes[0], ow)) step.State = TapOffsets(step.InputShapes[0], ow);
        context.For(step.Tiling.Count, step, &Tile);
    }

    /// <summary>
    /// Where each row of B, tap (ci, ky, kx), reads from a panel's first value: in the first input row it uses for a
    /// stride of 1, and in the split rows of the tile (<see cref="TileSplit"/>) for a stride of 2.
    /// </summary>
    private int[] TapOffsets(int[] x, int ow)
    {
        int w = x[3];
        long plane = _broadcast ? 0 : (long)x[2] * w;
        long half = HalfStride(ow);
        var offsets = GC.AllocateArray<int>(_k, pinned: true);
        for (int ci = 0, i = 0; ci < CinPerGroup; ci++)
        {
            for (int ky = 0; ky < Kh; ky++)
            {
                for (int kx = 0; kx < Kw; kx++)
                {
                    int reach = kx * Dw;
                    long row = ((_broadcast ? 0 : ci) * Kh) + ky;
                    offsets[i++] = (int)(Sw == 1
                        ? (ci * plane) + ((long)ky * Dh * w) + reach
                        : (row * 2 * half) + ((reach & 1) * half) + (reach >> 1));
                }
            }
        }

        return offsets;
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
        if (step.State is int[] offsets)
        {
            if (layer.Sw == 1) layer.TileInPlace(step, input, x[2], x[3], ow, n, n0, columns, m0, rows, offsets, packed);
            else layer.TileSplit(step, input, x[2], x[3], ow, n, n0, columns, m0, rows, offsets, packed);
            return;
        }

        if (layer._pointwise) gemm.PackBRows(input + n0, planeOut, layer._k, columns, packed);
        else
        {
            layer.PackPatches(input, layer._broadcast ? 0 : (long)x[2] * x[3], x[3], x[2], x[3], 0, 0, ow, 0, 0, n0, columns, packed);
            gemm.ClearPanelTail(packed, layer._k, columns);
        }

        float* output = step.Output;
        long baseOffset = (long)n * layer.Cout * planeOut + (long)m0 * planeOut + n0;
        var epilogue = layer.TileEpilogue(step, baseOffset, rows, columns, planeOut, packed);
        fixed (float* weights = layer._weights, bias = layer._bias)
            gemm.Multiply(layer._k, weights + (long)m0 * layer._k, rows, packed, columns, output + baseOffset, planeOut, bias + m0, null, epilogue);
    }

    /// <summary>
    /// The epilogue of the tile at <paramref name="baseOffset"/> in the output, <paramref name="rows"/> channels by
    /// <paramref name="columns"/> pixels. When the output writes over the residual, the tile's part of the residual is
    /// copied here, before anything of the tile is written, to the worker's scratch after what the tile works in
    /// (<see cref="ResidualCopy"/>), and the epilogue reads the copy, whose rows lie as far apart as the tile's columns.
    /// The values and the order of the additions are unchanged.
    /// </summary>
    private Epilogue TileEpilogue(Step step, long baseOffset, int rows, int columns, int planeOut, float* scratch)
    {
        if (!HasResidual) return new Epilogue(Activation, Alpha, Beta, null);
        float* residual = step.Input(1) + baseOffset;
        if (step.Input(1) != step.Output) return new Epilogue(Activation, Alpha, Beta, residual);
        float* copy = ResidualCopy(step, scratch);
        for (int r = 0; r < rows; r++) new ReadOnlySpan<float>(residual + (long)r * planeOut, columns).CopyTo(new Span<float>(copy + (long)r * columns, columns));
        return new Epilogue(Activation, Alpha, Beta, copy, columns);
    }

    /// <summary>Where a tile keeps its copy of a residual that the output writes over, in the worker's scratch.</summary>
    private float* ResidualCopy(Step step, float* scratch) => scratch + Align(WorkFloats(step.InputShapes[0], step.OutputShape));

    /// <summary>
    /// A tile of the columns a run leaves to compute, gathered from the pieces of the output plane they lie in: the
    /// pieces are packed side by side into one set of panels. Panels that lie within one piece are multiplied straight
    /// into the output, as the plain tile does; a panel that spans two pieces goes to scratch and is copied piece by
    /// piece, finished with the operations of the plain epilogue in the same order, so every value is the same. When
    /// the output writes over the residual, the tile first copies the residual of its gathered columns to scratch, side
    /// by side as they are packed, and reads it there.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GatheredTile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (GemmConvLayer)step.Layer;
        var gemm = layer._gemm;
        var tiling = step.Context.RunTiling;
        tiling.Decode(tile, out int n, out int v0, out int columns, out int m0, out int rows);
        int[] x = step.InputShapes[0];
        int ow = step.OutputShape[3];
        int planeOut = step.OutputShape[2] * ow;
        float* input = step.Input(0) + (long)n * x[1] * x[2] * x[3];
        long planeIn = layer._broadcast ? 0 : (long)x[2] * x[3];
        float* packed = step.Context.Scratch(worker);
        float* product = packed + gemm.PackedBSize(layer._k, columns) + ProductOffset;
        int first = tiling.FirstPiece(n, v0, out int firstSkip);
        for (int p = first, at = 0, skip = firstSkip; at < columns; p++, skip = 0)
        {
            int length = Math.Min(tiling.PieceLength(p) - skip, columns - at);
            int start = tiling.PieceStart(p) + skip;
            if (layer._pointwise) gemm.PackBRowsAt(input + start, planeOut, layer._k, length, packed, at);
            else layer.PackPatches(input, planeIn, x[3], x[2], x[3], 0, 0, ow, 0, 0, start, length, packed, at);
            at += length;
        }

        gemm.ClearPanelTail(packed, layer._k, columns);
        float* output = step.Output + (long)n * layer.Cout * planeOut + (long)m0 * planeOut;
        float* residual = layer.HasResidual ? step.Input(1) + (long)n * layer.Cout * planeOut + (long)m0 * planeOut : null;
        float* copy = null;
        if (residual != null && step.Input(1) == step.Output)
        {
            // The output writes over the residual: the tile's part of it goes to scratch before the tile writes.
            copy = layer.ResidualCopy(step, packed);
            for (int p = first, at = 0, skip = firstSkip; at < columns; p++, skip = 0)
            {
                int length = Math.Min(tiling.PieceLength(p) - skip, columns - at);
                long start = tiling.PieceStart(p) + skip;
                for (int r = 0; r < rows; r++)
                    new ReadOnlySpan<float>(residual + ((long)r * planeOut) + start, length).CopyTo(new Span<float>(copy + ((long)r * columns) + at, length));
                at += length;
            }
        }

        int nr = gemm.NR, k = layer._k;
        var activation = new Epilogue(layer.Activation, layer.Alpha, layer.Beta, null);
        fixed (float* weights = layer._weights, bias = layer._bias)
        {
            // One panel of rows at a time across the whole tile, so that its weights stay in the first-level cache.
            for (int i = 0; i < rows; i += gemm.MR)
            {
                int h = Math.Min(gemm.MR, rows - i);
                float* a = weights + (long)(m0 + i) * k;
                float* outRows = output + (long)i * planeOut;
                float* resRows = residual == null ? null : residual + (long)i * planeOut;
                float* copyRows = copy == null ? null : copy + (long)i * columns;
                for (int v = 0; v < columns;)
                {
                    int piece = tiling.FirstPiece(n, v0 + v, out int into);
                    int remaining = tiling.PieceLength(piece) - into, left = columns - v;
                    long start = tiling.PieceStart(piece) + into;
                    if (remaining >= left || remaining >= nr)
                    {
                        // Whole panels within one piece go straight to their place, finished there as in the plain tile.
                        int span = remaining >= left ? left : remaining / nr * nr;
                        var epilogue = copyRows != null ? new Epilogue(layer.Activation, layer.Alpha, layer.Beta, copyRows + v, columns)
                            : new Epilogue(layer.Activation, layer.Alpha, layer.Beta, resRows == null ? null : resRows + start);
                        gemm.Multiply(k, a, h, packed + (long)v * k, span, outRows + start, planeOut, bias + m0 + i, null, epilogue);
                        v += span;
                        continue;
                    }

                    // A panel that spans pieces goes to scratch with its activation, and each piece of it to its place,
                    // with the residual added after the activation as the plain epilogue adds it.
                    int w = Math.Min(nr, left);
                    gemm.Multiply(k, a, h, packed + (long)v * k, w, product, nr, bias + m0 + i, null, activation);
                    for (int r = 0; r < h; r++)
                    {
                        float* src = product + (long)r * nr;
                        long row = (long)r * planeOut;
                        for (int q = piece, at = 0, skip = into; at < w; q++, skip = 0)
                        {
                            int length = Math.Min(tiling.PieceLength(q) - skip, w - at);
                            long offset = row + tiling.PieceStart(q) + skip;
                            if (copyRows != null) VectorMath.Add(src + at, copyRows + ((long)r * columns) + v + at, outRows + offset, length);
                            else if (resRows != null) VectorMath.Add(src + at, resRows + offset, outRows + offset, length);
                            else CopyRun(src + at, outRows + offset, length);
                            at += length;
                        }
                    }

                    v += w;
                }
            }
        }
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
                _gemm.ClearPanelTail(packed, _k, count);
                _gemm.Multiply(_k, weights, Cout, packed, count, output + n0, (nint)outputPlane, bias, null, epilogue);
            }
        }
    }

    /// <summary>The floats of packed panels <see cref="ComputeBlock"/> needs for a block of the given number of pixels.</summary>
    internal long BlockScratch(int pixels) => _gemm.PackedBSize(_k, TileColumns(pixels));

    /// <summary>
    /// One tile of a convolution with a horizontal stride of 1, panel by panel. A panel whose pixels lie in one output
    /// row and whose taps all read inside the input is not packed: each tap's NR values are a run of one input row, so
    /// the kernel reads them where they are, at the tap's offset from the panel's first value. Packing such a panel
    /// cost about as much as multiplying it for the detector's and recogniser's first convolutions, whose few output
    /// channels use each packed value only 12 to 48 times. Panels at the edges and across the end of an output row are
    /// packed as before. Both multiply the same values in the same order.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void TileInPlace(Step step, float* input, int h, int w, int ow, int batch, int n0, int columns, int m0, int rows, int[] offsets, float* packed)
    {
        int nr = _gemm.NR;
        int planeOut = step.OutputShape[2] * ow;
        long baseOffset = (long)batch * Cout * planeOut + (long)m0 * planeOut + n0;
        float* output = step.Output + baseOffset;

        // A residual that the output writes over is copied before the first panel writes.
        var epilogue = TileEpilogue(step, baseOffset, rows, columns, planeOut, packed);

        // The first input row and column a panel's taps read must lie within these bounds for every tap to read inside.
        int rowEnd = h - ((Kh - 1) * Dh);
        int columnLast = w - ((Kw - 1) * Dw) - nr;
        float* edge = stackalloc float[12 * 32];
        fixed (float* weights = _weights, bias = _bias)
        fixed (int* tapOffsets = offsets)
        {
            float* a = weights + (long)m0 * _k;
            for (int j0 = 0; j0 < columns; j0 += nr)
            {
                int width = Math.Min(nr, columns - j0);
                int n = n0 + j0;
                int oy = n / ow, ox = n - oy * ow;
                int iy = oy * Sh - Pt, ix = ox - Pl;
                if (width == nr && ox + nr <= ow && iy >= 0 && iy < rowEnd && ix >= 0 && ix <= columnLast)
                {
                    _gemm.MultiplyDirect(_k, a, rows, input + (long)iy * w + ix, tapOffsets, output + j0, planeOut, bias + m0, edge, nr);
                }
                else
                {
                    PackPatches(input, _broadcast ? 0 : (long)h * w, w, h, w, 0, 0, ow, 0, 0, n, width, packed);
                    _gemm.ClearPanelTail(packed, _k, width);
                    _gemm.Multiply(_k, a, rows, packed, width, output + j0, planeOut, bias + m0, null);
                }
            }
        }

        // The activation and the residual once for the tile, a block of rows at a time while it is still in the cache;
        // each value is treated alone, so the result is the same as per panel.
        if (epilogue.IsEmpty) return;
        int mr = _gemm.MR;
        for (int i = 0; i < rows; i += mr) epilogue.Apply(output + (nint)i * planeOut, planeOut, i, Math.Min(mr, rows - i), columns);
    }

    /// <summary>
    /// One tile of a convolution with a horizontal stride of 2. A panel's values for one tap are every other value of
    /// an input row, so the tile first splits the input rows each of its output rows reads into even and odd halves,
    /// with zeros in the padding, one run per output row the tile covers. Then a tap's values for a panel are a run of
    /// one half, and every panel inside one output row is read in place, at the tap's offset into the split rows, as
    /// <see cref="TileInPlace"/> does for a stride of 1; panels across the end of an output row are packed. The values,
    /// zeros included, and their order are those of the packed panels, so the output is the same.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void TileSplit(Step step, float* input, int h, int w, int ow, int batch, int n0, int columns, int m0, int rows, int[] offsets, float* packed)
    {
        int nr = _gemm.NR;
        int planeOut = step.OutputShape[2] * ow;
        long baseOffset = (long)batch * Cout * planeOut + (long)m0 * planeOut + n0;
        float* output = step.Output + baseOffset;

        // A residual that the output writes over is copied before the first panel writes.
        var epilogue = TileEpilogue(step, baseOffset, rows, columns, planeOut, packed);
        int half = HalfStride(ow);
        float* split = packed + _gemm.PackedBSize(_k, nr);
        long planeSize = _broadcast ? 0 : (long)h * w;
        int channels = _broadcast ? 1 : CinPerGroup;
        float* edge = stackalloc float[12 * 32];
        fixed (float* weights = _weights, bias = _bias)
        fixed (int* tapOffsets = offsets)
        {
            float* a = weights + (long)m0 * _k;
            for (int s0 = 0; s0 < columns;)
            {
                // One run of the tile's pixels in one output row, and the panels that lie wholly inside it (the tile's
                // last panel may be narrower than the others).
                int n = n0 + s0;
                int oy = n / ow, ox0 = n - oy * ow;
                int length = Math.Min(ow - ox0, columns - s0);
                int s1 = s0 + length;
                int first = GemmKernel.RoundUp(s0, nr);
                if (first < s1 && first + Math.Min(nr, columns - first) <= s1)
                {
                    // A narrower last panel still reads a whole panel's width, so the halves reach that far past the
                    // run.
                    int halfLength = length + nr + ((Kw * Dw) / 2) + 1;
                    int ix0 = ox0 * 2 - Pl;
                    float* dst = split;
                    float* plane = input;
                    for (int ci = 0; ci < channels; ci++, plane += planeSize)
                    {
                        for (int ky = 0, iy = oy * Sh - Pt; ky < Kh; ky++, iy += Dh, dst += 2 * half)
                        {
                            if ((uint)iy >= (uint)h)
                            {
                                new Span<float>(dst, half + halfLength).Clear();
                                continue;
                            }

                            SplitRow(plane + (long)iy * w, w, ix0, dst, half, halfLength);
                        }
                    }

                    for (int j0 = first; j0 < s1; j0 += nr)
                    {
                        int width = Math.Min(nr, columns - j0);
                        if (j0 + width > s1) break;
                        _gemm.MultiplyDirect(_k, a, rows, split + (j0 - s0), tapOffsets, output + j0, planeOut, bias + m0, edge, width);
                    }
                }

                s0 = s1;
            }

            // The panels that span two output rows, packed.
            for (int j0 = 0; j0 < columns; j0 += nr)
            {
                int width = Math.Min(nr, columns - j0);
                int n = n0 + j0;
                if (n / ow == (n + width - 1) / ow) continue;
                PackPatches(input, planeSize, w, h, w, 0, 0, ow, 0, 0, n, width, packed);
                _gemm.ClearPanelTail(packed, _k, width);
                _gemm.Multiply(_k, a, rows, packed, width, output + j0, planeOut, bias + m0, null);
            }
        }

        if (epilogue.IsEmpty) return;
        int mr = _gemm.MR;
        for (int i = 0; i < rows; i += mr) epilogue.Apply(output + (nint)i * planeOut, planeOut, i, Math.Min(mr, rows - i), columns);
    }

    /// <summary>
    /// Splits input columns ix0, ix0 + 1, ... of one row into even ones at dst and odd ones at dst + half, count of
    /// each, with zeros for the columns outside the row.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SplitRow(float* row, int w, int ix0, float* dst, int half, int count)
    {
        for (int parity = 0; parity < 2; parity++, dst += half)
        {
            // Column ix0 + 2 i + parity lies in the row for i in [start, end).
            int c = ix0 + parity;
            int start = c >= 0 ? 0 : Math.Min(count, (1 - c) / 2);
            int end = Math.Clamp(c > w - 1 ? 0 : ((w - 1 - c) / 2) + 1, start, count);
            new Span<float>(dst, start).Clear();
            if (end > start) VectorMath.CopyEven(row + c + (2 * start), dst + start, end - start);
            new Span<float>(dst + end, count - end).Clear();
        }
    }

    /// <summary>
    /// Gathers the patches of output pixels [n0, n0 + columns) of a block into NR-column panels, as columns [first,
    /// first + columns) of them (lanes after the last are left for <see cref="GemmKernel.ClearPanelTail"/>): row
    /// (ci, ky, kx) of a panel holds the input value each output pixel's kernel tap reads, or zero in the padding. The block's pixels go
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
        int oy0, int ox0, int n0, int columns, float* packed, int first = 0)
    {
        int nr = _gemm.NR;
        int k = _k;
        int channels = CinPerGroup;
        int kh = Kh, kw = Kw, sw = Sw, dh = Dh, dw = Dw;
        // The map's first channel as if the window held it whole; only rows and columns the window holds are read.
        float* origin = input - (long)firstRow * rowStride - firstColumn;
        for (int pos = first, end = first + columns; pos < end;)
        {
            int lane = pos % nr;
            int width = Math.Min(nr - lane, end - pos);
            float* panel = packed + ((long)(pos - lane) * k) + lane;
            for (int t = 0; t < width;)
            {
                int n = n0 + (pos - first) + t;
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
                        if (Sse.IsSupported)
                        {
                            // The next panel reads on from here in each of these rows: more rows than the hardware
                            // prefetcher follows, so they are fetched while this panel is packed.
                            float* next = row + x0 + nr * sw;
                            Sse.Prefetch0(next);
                            Sse.Prefetch0(next + 16);
                            if (sw > 1)
                            {
                                Sse.Prefetch0(next + 32);
                                Sse.Prefetch0(next + 48);
                            }
                        }

                        for (int kx = 0, ix = x0; kx < kw; kx++, ix += dw, dst += nr) CopyTap(row, w, ix, sw, dst, length);
                    }
                }

                t += length;
            }

            pos += width;
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

    // When each tile takes a whole plane, it copies all of the plane's input rows to its scratch before writing any output
    // row, and reads no other plane, so the output may take the input's memory.
    public override int InPlaceInput(int[][] inputs, int[] output) => BandRows(output) == output[2] ? 0 : -1;

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

        int tapCount = Kh * Kw;
        float* taps = stackalloc float[tapCount];
        fixed (float* weights = _weights) new Span<float>(weights + (long)c * tapCount, tapCount).CopyTo(new Span<float>(taps, tapCount));

        // Where each tap reads, from an output pixel's position in its first padded row: for a stride of 2, the even
        // taps read the even half and the odd taps the odd half.
        int* offsets = stackalloc int[tapCount];
        for (int ky = 0; ky < Kh; ky++)
        {
            for (int kx = 0; kx < Kw; kx++)
                offsets[(ky * Kw) + kx] = (ky * rowLength) + (Sw == 1 ? kx : ((kx & 1) == 0 ? 0 : half) + (kx >> 1));
        }

        // Each row goes in blocks of four vectors, and the whole vectors left at the ends of the rows of the band go
        // four at a time, whichever rows they are in: four independent sums hide the latency of the additions, which
        // a row's last one to three vectors alone did not, and a plane of the orientation classifier, 96 pixels wide,
        // has only one block of four per row. Every pixel still sums its taps in the same order, with the same
        // separate products and sums.
        bool wide = VectorMath.Wide;
        int vector = wide ? 16 : Width;
        int whole = ow / vector * vector;
        float** sources = stackalloc float*[4];
        float** targets = stackalloc float*[4];
        float* spare = stackalloc float[16];
        int pending = 0;
        for (int y = y0; y < y1; y++)
        {
            float* rows = scratch + (long)(y * Sh - Pt - iyStart) * rowLength;
            float* outRow = output + (long)y * ow;
            int ox = wide ? RowBlocksWide(rows, rowLength, half, taps, _bias[c], outRow, ow) : RowBlocks(rows, rowLength, half, taps, _bias[c], outRow, ow);
            for (; ox < whole; ox += vector)
            {
                sources[pending] = rows + ox;
                targets[pending] = outRow + ox;
                if (++pending == 4)
                {
                    if (wide) BlockWide(sources, targets, taps, offsets, tapCount, _bias[c]);
                    else Block(sources, targets, taps, offsets, tapCount, _bias[c]);
                    pending = 0;
                }
            }
        }

        if (pending > 0)
        {
            // The last group is filled up with copies of its last vector, written to a spare buffer.
            for (int i = pending; i < 4; i++)
            {
                sources[i] = sources[pending - 1];
                targets[i] = spare;
            }

            if (wide) BlockWide(sources, targets, taps, offsets, tapCount, _bias[c]);
            else Block(sources, targets, taps, offsets, tapCount, _bias[c]);
        }

        // The pixels after a row's last whole vector, through a zero-padded vector.
        var bias = new Vector<float>(_bias[c]);
        float* tail = stackalloc float[Width];
        for (int y = y0; y < y1; y++)
        {
            float* outRow = output + (long)y * ow;
            float* rows = scratch + (long)(y * Sh - Pt - iyStart) * rowLength;
            for (int ox = whole; ox < ow; ox += Width)
            {
                var acc = bias;
                for (int i = 0; i < tapCount; i++) acc += new Vector<float>(taps[i]) * VectorMath.Load(rows + ox + offsets[i]);
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

    /// <summary>The blocks of four whole vectors at the start of one output row; returns how many pixels it did.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int RowBlocks(float* rows, int rowLength, int half, float* taps, float biasValue, float* outRow, int ow)
    {
        var bias = new Vector<float>(biasValue);
        int ox = 0;
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
        return ox;
    }

    /// <summary><see cref="RowBlocks"/> on 512-bit vectors, for processors with AVX-512.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int RowBlocksWide(float* rows, int rowLength, int half, float* taps, float biasValue, float* outRow, int ow)
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
        return ox;
    }

    /// <summary>Four vectors of output pixels, each summing the taps from its own position.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Block(float** src, float** dst, float* taps, int* offsets, int count, float biasValue)
    {
        float* p0 = src[0], p1 = src[1], p2 = src[2], p3 = src[3];
        var bias = new Vector<float>(biasValue);
        Vector<float> a0 = bias, a1 = bias, a2 = bias, a3 = bias;
        for (int i = 0; i < count; i++)
        {
            int o = offsets[i];
            var t = new Vector<float>(taps[i]);
            a0 += t * VectorMath.Load(p0 + o);
            a1 += t * VectorMath.Load(p1 + o);
            a2 += t * VectorMath.Load(p2 + o);
            a3 += t * VectorMath.Load(p3 + o);
        }

        VectorMath.Store(VectorMath.Activate(a0, Activation, Alpha, Beta), dst[0]);
        VectorMath.Store(VectorMath.Activate(a1, Activation, Alpha, Beta), dst[1]);
        VectorMath.Store(VectorMath.Activate(a2, Activation, Alpha, Beta), dst[2]);
        VectorMath.Store(VectorMath.Activate(a3, Activation, Alpha, Beta), dst[3]);
    }

    /// <summary>
    /// <see cref="Block"/> on 512-bit vectors, for processors with AVX-512, where Vector&lt;T&gt; stays 256 bits wide. The
    /// products, sums and the wide activation match the narrow ones lane for lane, so the results do not depend on
    /// which path a pixel takes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void BlockWide(float** src, float** dst, float* taps, int* offsets, int count, float biasValue)
    {
        float* p0 = src[0], p1 = src[1], p2 = src[2], p3 = src[3];
        var bias = Vector512.Create(biasValue);
        Vector512<float> a0 = bias, a1 = bias, a2 = bias, a3 = bias;
        for (int i = 0; i < count; i++)
        {
            int o = offsets[i];
            var t = Vector512.Create(taps[i]);
            a0 += t * Vector512.Load(p0 + o);
            a1 += t * Vector512.Load(p1 + o);
            a2 += t * Vector512.Load(p2 + o);
            a3 += t * Vector512.Load(p3 + o);
        }

        Vector512.Store(VectorMath.ActivateWide(a0, Activation, Alpha, Beta), dst[0]);
        Vector512.Store(VectorMath.ActivateWide(a1, Activation, Alpha, Beta), dst[1]);
        Vector512.Store(VectorMath.ActivateWide(a2, Activation, Alpha, Beta), dst[2]);
        Vector512.Store(VectorMath.ActivateWide(a3, Activation, Alpha, Beta), dst[3]);
    }
}

/// <summary>
/// The tiles of a convolution that computes only the columns outside the output's runs and the first column of each
/// run (see <see cref="ColumnRuns"/>). Per batch item, the output pixels in row-major order split into pieces that end
/// where a row's run starts after its first column and resume where it ends; the pieces of an item are numbered on
/// from one to the next as if they were one row of columns, and those columns are tiled as <see cref="Tiling"/> tiles
/// a plane: in blocks of whole panels and rows in blocks. Kept by the context and rebuilt for each run of a
/// layer, since the runs change with the lines of a batch; only one convolution runs at a time.
/// </summary>
internal sealed class RunTiling
{
    private int[] _pieceStart = new int[64], _pieceLength = new int[64], _pieceColumn = new int[64];
    private int[] _itemPiece = new int[8];
    private int[] _tileItem = new int[64], _tileColumn = new int[64], _tileColumns = new int[64];
    private int _tiles, _rowBlocks, _rowBlockPanels, _panel, _rows;

    public int Count => _tiles * _rowBlocks;

    /// <summary>
    /// Builds the pieces and tiles for output planes of <paramref name="height"/> x <paramref name="width"/>, leaving
    /// out runs of at least <paramref name="minimumGap"/> columns after their first; false when no item has one.
    /// </summary>
    public bool Build(ColumnRuns runs, int height, int width, int tileColumns, int panelWidth, int rows, int panel, int minimumGap)
    {
        int pieces = 0;
        _tiles = 0;
        bool skips = false;
        int plane = height * width;
        if (_itemPiece.Length < runs.Batch + 1) _itemPiece = new int[runs.Batch + 1];
        for (int n = 0; n < runs.Batch; n++)
        {
            _itemPiece[n] = pieces;
            int p = 0, column = 0;
            if (runs.HasRun(n) && runs.End[n] - runs.Start[n] - 1 >= minimumGap)
            {
                int s = runs.Start[n], e = runs.End[n];
                for (int y = 0; y < height; y++)
                {
                    int stop = (y * width) + s + 1;
                    AddPiece(ref pieces, p, stop - p, ref column);
                    p = (y * width) + e;
                }

                skips = true;
            }

            if (p < plane) AddPiece(ref pieces, p, plane - p, ref column);

            // The item's columns in tiles of whole panels, as the plain tiling splits a plane.
            int size = Tiling.Balanced(column, tileColumns, panelWidth);
            for (int v = 0; v < column; v += size) AddTile(n, v, Math.Min(size, column - v));
        }

        _itemPiece[runs.Batch] = pieces;
        if (!skips) return false;
        _panel = panel;
        _rows = rows;
        int panels = (rows + panel - 1) / panel;
        int wanted = Math.Clamp((16 + _tiles - 1) / _tiles, 1, panels);
        _rowBlockPanels = (panels + wanted - 1) / wanted;
        _rowBlocks = (panels + _rowBlockPanels - 1) / _rowBlockPanels;
        return true;
    }

    private void AddPiece(ref int pieces, int start, int length, ref int column)
    {
        if (length <= 0) return;
        if (pieces == _pieceStart.Length)
        {
            Array.Resize(ref _pieceStart, pieces * 2);
            Array.Resize(ref _pieceLength, pieces * 2);
            Array.Resize(ref _pieceColumn, pieces * 2);
        }

        _pieceStart[pieces] = start;
        _pieceLength[pieces] = length;
        _pieceColumn[pieces] = column;
        pieces++;
        column += length;
    }

    private void AddTile(int n, int column, int columns)
    {
        if (_tiles == _tileItem.Length)
        {
            Array.Resize(ref _tileItem, _tiles * 2);
            Array.Resize(ref _tileColumn, _tiles * 2);
            Array.Resize(ref _tileColumns, _tiles * 2);
        }

        _tileItem[_tiles] = n;
        _tileColumn[_tiles] = column;
        _tileColumns[_tiles] = columns;
        _tiles++;
    }

    /// <summary>A tile: its batch item, its first gathered column and their number, and its block of output rows.</summary>
    public void Decode(int tile, out int batch, out int column, out int columns, out int row, out int rows)
    {
        int rowBlock = tile % _rowBlocks;
        int t = tile / _rowBlocks;
        batch = _tileItem[t];
        column = _tileColumn[t];
        columns = _tileColumns[t];
        row = rowBlock * _rowBlockPanels * _panel;
        rows = Math.Min(_rowBlockPanels * _panel, _rows - row);
    }

    /// <summary>The piece of item n that holds its gathered column, and how far into the piece that column is.</summary>
    public int FirstPiece(int n, int column, out int skip)
    {
        int p = _itemPiece[n];
        while (p + 1 < _itemPiece[n + 1] && _pieceColumn[p + 1] <= column) p++;
        skip = column - _pieceColumn[p];
        return p;
    }

    /// <summary>Where a piece starts in its item's output plane.</summary>
    public int PieceStart(int piece) => _pieceStart[piece];

    public int PieceLength(int piece) => _pieceLength[piece];
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
