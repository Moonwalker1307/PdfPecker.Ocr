using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// The stem that the detector and the recogniser share (PP-HGNetV2's): a 3x3 convolution with stride 2 to map A, a 2x2
/// convolution of A to B and another of B to C, a 2x2 maximum of A to D, and a 3x3 convolution with stride 2 of D and
/// C concatenated, which gives the stem's output F at a quarter of the input's size. Run layer by layer, A, B, C and the
/// concatenation are full maps at half the input's size, and they set both networks' working memory: A, B and C alive
/// together took the detector's whole 96 MB arena on an A4 page, and for a batch of long lines, whose concatenation
/// cannot share memory with its inputs because the batch items lie between their channels, the concatenation and its
/// inputs took twice what the rest of the recogniser needs.
/// </summary>
/// <remarks>
/// This layer computes F block by block instead. For each block of F it computes, into the worker's scratch, the
/// window of A, B, C and D that the block reads, with C and D side by side as the concatenation, and then the block.
/// Windows of neighbouring blocks overlap by the kernels' reach, so a little of A, B and C is computed twice; the
/// blocks are shaped to keep that small within a fixed amount of scratch. Every value is the sum the separate layers
/// compute for it, in the same order (see <see cref="GemmConvLayer.ComputeBlock"/>), so the output has the same bits.
/// </remarks>
internal sealed unsafe class StemLayer : Layer
{
    // The floats one worker's windows may take. Four megabytes keep the windows mostly in the caches while the blocks
    // are large enough that little is computed twice and the matrix products run on long rows; smaller budgets made the
    // detector's stem up to 40% slower than the separate layers, and on four threads the windows still fit in the arena
    // beside F.
    private const long WindowBudget = 1024 * 1024;

    private readonly GemmConvLayer _first;
    private readonly GemmConvLayer _reduce;
    private readonly GemmConvLayer _expand;
    private readonly PoolLayer _pool;
    private readonly GemmConvLayer _last;
    private readonly bool _poolFirst;

    private StemLayer(NodeDefinition node, GemmConvLayer first, GemmConvLayer reduce, GemmConvLayer expand, PoolLayer pool, GemmConvLayer last, bool poolFirst)
        : base(node)
    {
        _first = first;
        _reduce = reduce;
        _expand = expand;
        _pool = pool;
        _last = last;
        _poolFirst = poolFirst;
    }

    /// <summary>
    /// The stem that the first six layers form, or null when they are not such a stem or a later layer reads one of its
    /// inner values. <paramref name="inputs"/> lists the values each layer reads, as <see cref="OcrModel.Inputs"/> does.
    /// </summary>
    public static StemLayer? TryCreate(IReadOnlyList<Layer> layers, int[][] inputs)
    {
        if (layers.Count < 7) return null;
        if (layers[0] is not GemmConvLayer first || layers[1] is not GemmConvLayer reduce || layers[2] is not GemmConvLayer expand
            || layers[3] is not PoolLayer { IsStemMaximum: true } pool || layers[4] is not ConcatLayer concat || layers[5] is not GemmConvLayer last)
            return null;
        if (first.HasResidual || reduce.HasResidual || expand.HasResidual || last.HasResidual) return null;
        // Values: 1 = A, 2 = B, 3 = C, 4 = D, 5 = the concatenation, 6 = F.
        if (inputs[1] is not [1] || inputs[2] is not [2] || inputs[3] is not [1] || inputs[5] is not [5]) return null;
        bool poolFirst;
        if (inputs[4] is [4, 3]) poolFirst = true;
        else if (inputs[4] is [3, 4]) poolFirst = false;
        else return null;
        if (concat.Factor(0) != (1, 1) || concat.Factor(1) != (1, 1)) return null;
        for (int i = 6; i < layers.Count; i++)
            foreach (int v in inputs[i])
                if (v is >= 1 and <= 5) return null;
        var node = new NodeDefinition(last.Node.Op, inputs[0], last.Node.Output, last.Node.Ints, last.Node.Floats, last.Node.Weights);
        return new StemLayer(node, first, reduce, expand, pool, last, poolFirst);
    }

    /// <summary>The shapes of A, B, C and F for an input shape, checking that the inner maps line up.</summary>
    private Shapes ShapesFor(int[] x)
    {
        int[] a = _first.OutputShape([x]);
        int[] b = _reduce.OutputShape([a]);
        int[] c = _expand.OutputShape([b]);
        int[] d = _pool.OutputShape([a]);
        if (c[2] != a[2] || c[3] != a[3] || !d.AsSpan().SequenceEqual(a)) throw Unsupported("The stem's maps do not line up.");
        int[] f = _last.OutputShape([[a[0], a[1] + c[1], a[2], a[3]]]);
        return new Shapes(x, a, b, c, f);
    }

    public override int[] OutputShape(int[][] inputs)
    {
        RequireRank(inputs[0], 4, "The stem");
        return ShapesFor(inputs[0]).F;
    }

    // The output's runs are those the separate layers find: each inner map's from the map it reads, the concatenation's
    // where both of its parts have one, and the output's from the concatenation's.
    public override void FindRuns(Step step)
    {
        var g = Plan(step);
        Shapes s = g.Shapes;
        _first.WindowRuns(step.InputRuns[0], g.RunsA, s.A[3]);
        _reduce.WindowRuns(g.RunsA, g.RunsB, s.B[3]);
        _expand.WindowRuns(g.RunsB, g.RunsC, s.C[3]);
        _pool.WindowRuns(g.RunsA, g.RunsD, s.A[3]);
        g.RunsD.Intersect(g.RunsC);
        _last.WindowRuns(g.RunsD, step.OutputRuns, s.F[3]);
    }

    // Each worker's windows are its part of the step's shared scratch, which the plan places in the arena: while the
    // stem runs, the arena holds little else.
    public override long SharedScratch(Step step) => Plan(step).Scratch * step.Context.MaxDegreeOfParallelism;

    public override void Execute(Step step, InferenceContext context)
    {
        var plan = Plan(step);
        context.For(plan.Count, step, &Tile);
    }

    private Geometry Plan(Step step)
    {
        if (step.State is Geometry g) return g;
        g = new Geometry(this, ShapesFor(step.InputShapes[0]), step.Context.MaxDegreeOfParallelism);
        step.State = g;
        return g;
    }

    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (StemLayer)step.Layer;
        var g = (Geometry)step.State!;
        g.Decode(tile, out int n, out int fy0, out int fy1, out int fx0, out int fx1);
        layer.ComputeTile(step, g, n, fy0, fy1, fx0, fx1, step.SharedScratch + worker * g.Scratch);
    }

    private void ComputeTile(Step step, Geometry g, int n, int fy0, int fy1, int fx0, int fx1, float* scratch)
    {
        Shapes s = g.Shapes;
        int ha = s.A[2], wa = s.A[3], hb = s.B[2], wb = s.B[3];
        var w = g.Windows(fy0, fy1, fx0, fx1);
        long aPlane = (long)w.ARows * w.ACols, bPlane = (long)w.BRows * w.BCols, ePlane = (long)w.ERows * w.ECols;
        float* a = scratch;
        float* b = a + Align(s.A[1] * aPlane);
        float* e = b + Align(s.B[1] * bPlane);
        float* f = e + Align((s.A[1] + s.C[1]) * ePlane);
        float* packed = f + Align(g.FBlockFloats);
        float* d = _poolFirst ? e : e + s.C[1] * ePlane;
        float* c = _poolFirst ? e + s.A[1] * ePlane : e;

        int[] x = s.X;
        float* input = step.Input(0) + (long)n * x[1] * x[2] * x[3];
        _first.ComputeBlock(input, (long)x[2] * x[3], x[3], x[2], x[3], 0, 0, w.AY, w.AX, w.ARows, w.ACols, a, aPlane, packed);
        _reduce.ComputeBlock(a, aPlane, w.ACols, ha, wa, w.AY, w.AX, w.BY, w.BX, w.BRows, w.BCols, b, bPlane, packed);
        _expand.ComputeBlock(b, bPlane, w.BCols, hb, wb, w.BY, w.BX, w.EY, w.EX, w.ERows, w.ECols, c, ePlane, packed);
        PoolLayer.MaxPool2x2Stride1Block(a, aPlane, w.ACols, ha, wa, w.AY, w.AX, w.EY, w.EX, w.ERows, w.ECols, d, ePlane, s.A[1]);

        int hf = s.F[2], wf = s.F[3];
        long fPlane = (long)hf * wf;
        float* output = step.Output + (long)n * s.F[1] * fPlane;
        int rows = fy1 - fy0, cols = fx1 - fx0;
        if (cols == wf)
        {
            // Whole rows of F are contiguous in the output, so the block goes straight there.
            _last.ComputeBlock(e, ePlane, w.ECols, ha, wa, w.EY, w.EX, fy0, fx0, rows, cols, output + (long)fy0 * wf, fPlane, packed);
            return;
        }

        // A narrower block is one run of pixels for the matrix product, which then moves row by row into the output.
        long block = (long)rows * cols;
        _last.ComputeBlock(e, ePlane, w.ECols, ha, wa, w.EY, w.EX, fy0, fx0, rows, cols, f, block, packed);
        for (int ch = 0; ch < s.F[1]; ch++)
        {
            for (int r = 0; r < rows; r++)
            {
                new ReadOnlySpan<float>(f + ch * block + (long)r * cols, cols).CopyTo(new Span<float>(output + ch * fPlane + (long)(fy0 + r) * wf + fx0, cols));
            }
        }
    }

    // Windows start at 64-byte boundaries in scratch.
    private static long Align(long floats) => (floats + 15) & ~15L;

    private readonly record struct Shapes(int[] X, int[] A, int[] B, int[] C, int[] F);

    /// <summary>The windows of A, B and the concatenation that a block of F reads: first row and column, and size.</summary>
    private readonly record struct Windows(int AY, int AX, int ARows, int ACols, int BY, int BX, int BRows, int BCols, int EY, int EX, int ERows, int ECols);

    /// <summary>How a step's output splits into blocks, and the scratch the largest block needs.</summary>
    private sealed class Geometry
    {
        private readonly StemLayer _layer;
        private readonly int _rowBlocks, _columnBlocks, _blockRows, _blockColumns;

        public Geometry(StemLayer layer, Shapes shapes, int threads)
        {
            _layer = layer;
            Shapes = shapes;
            int hf = shapes.F[2], wf = shapes.F[3];
            (_blockRows, _blockColumns) = Choose(hf, wf, threads);
            _rowBlocks = (hf + _blockRows - 1) / _blockRows;
            _columnBlocks = (wf + _blockColumns - 1) / _blockColumns;
            Count = shapes.F[0] * _rowBlocks * _columnBlocks;
            FBlockFloats = _blockColumns == wf ? 0 : (long)shapes.F[1] * _blockRows * _blockColumns;
            long windows = 0, packed = layer._last.BlockScratch(_blockRows * _blockColumns);
            for (int tile = 0; tile < _rowBlocks * _columnBlocks; tile++)
            {
                Decode(tile, out _, out int fy0, out int fy1, out int fx0, out int fx1);
                var w = Windows(fy0, fy1, fx0, fx1);
                windows = Math.Max(windows, WindowFloats(w));
                packed = Math.Max(packed, Math.Max(layer._first.BlockScratch(w.ARows * w.ACols),
                    Math.Max(layer._reduce.BlockScratch(w.BRows * w.BCols), layer._expand.BlockScratch(w.ERows * w.ECols))));
            }

            Scratch = Align(windows + Align(FBlockFloats) + packed);
            int batch = shapes.X[0];
            RunsA = new ColumnRuns(batch);
            RunsB = new ColumnRuns(batch);
            RunsC = new ColumnRuns(batch);
            RunsD = new ColumnRuns(batch);
        }

        /// <summary>The runs of A, B, C and D (then of the concatenation) while the step's runs are found.</summary>
        public ColumnRuns RunsA { get; }

        public ColumnRuns RunsB { get; }

        public ColumnRuns RunsC { get; }

        public ColumnRuns RunsD { get; }

        public Shapes Shapes { get; }

        public int Count { get; }

        public long FBlockFloats { get; }

        public long Scratch { get; }

        public void Decode(int tile, out int n, out int fy0, out int fy1, out int fx0, out int fx1)
        {
            int column = tile % _columnBlocks;
            tile /= _columnBlocks;
            int row = tile % _rowBlocks;
            n = tile / _rowBlocks;
            fy0 = row * _blockRows;
            fy1 = Math.Min(Shapes.F[2], fy0 + _blockRows);
            fx0 = column * _blockColumns;
            fx1 = Math.Min(Shapes.F[3], fx0 + _blockColumns);
        }

        public Windows Windows(int fy0, int fy1, int fx0, int fx1)
        {
            var l = _layer;
            int ha = Shapes.A[2], wa = Shapes.A[3];
            var (ey0, ey1) = l._last.InputRows(fy0, fy1, ha);
            var (ex0, ex1) = l._last.InputColumns(fx0, fx1, wa);
            var (by0, by1) = l._expand.InputRows(ey0, ey1, Shapes.B[2]);
            var (bx0, bx1) = l._expand.InputColumns(ex0, ex1, Shapes.B[3]);
            var (ry0, ry1) = l._reduce.InputRows(by0, by1, ha);
            var (rx0, rx1) = l._reduce.InputColumns(bx0, bx1, wa);
            // The maximum reads its row and column and the next ones.
            int ay0 = Math.Min(ry0, ey0), ay1 = Math.Max(ry1, Math.Min(ha, ey1 + 1));
            int ax0 = Math.Min(rx0, ex0), ax1 = Math.Max(rx1, Math.Min(wa, ex1 + 1));
            return new Windows(ay0, ax0, ay1 - ay0, ax1 - ax0, by0, bx0, by1 - by0, bx1 - bx0, ey0, ex0, ey1 - ey0, ex1 - ex0);
        }

        private long WindowFloats(Windows w) =>
            Align(Shapes.A[1] * (long)w.ARows * w.ACols) + Align(Shapes.B[1] * (long)w.BRows * w.BCols) + Align((Shapes.A[1] + Shapes.C[1]) * (long)w.ERows * w.ECols);

        /// <summary>
        /// The block size that takes the least time on the given number of threads, counting what neighbouring windows
        /// compute twice and the rounds of blocks the threads take, among those whose windows fit the budget. Blocks
        /// are as equal as the map allows, so the last one is not a sliver. The choice changes no output bit.
        /// </summary>
        private (int Rows, int Columns) Choose(int hf, int wf, int threads)
        {
            var l = _layer;
            long costA = l._first.MultiplyAddsPerPixel, costB = l._reduce.MultiplyAddsPerPixel, costC = l._expand.MultiplyAddsPerPixel;
            long costF = l._last.MultiplyAddsPerPixel;
            (int, int) best = (1, 1);
            double bestWork = double.MaxValue;
            int lastRows = -1;
            for (int rowBlocks = 1; rowBlocks <= hf; rowBlocks++)
            {
                int rows = (hf + rowBlocks - 1) / rowBlocks;
                if (rows == lastRows) continue;
                lastRows = rows;
                int lastColumns = -1;
                for (int columnBlocks = 1; columnBlocks <= wf; columnBlocks++)
                {
                    int columns = (wf + columnBlocks - 1) / columnBlocks;
                    if (columns == lastColumns) continue;
                    lastColumns = columns;
                    // Interior windows are the largest; the first block stands in for them when there is only one.
                    var w = Windows(Math.Min(rows, hf - rows), Math.Min(rows, hf - rows) + rows, Math.Min(columns, wf - columns), Math.Min(columns, wf - columns) + columns);
                    long f = columns == wf ? 0 : (long)Shapes.F[1] * rows * columns;
                    if (WindowFloats(w) + f > WindowBudget) continue;
                    int blocks = ((hf + rows - 1) / rows) * ((wf + columns - 1) / columns);
                    int rounds = (Shapes.F[0] * blocks + threads - 1) / threads;
                    double work = rounds * ((double)w.ARows * w.ACols * costA + (double)w.BRows * w.BCols * costB + (double)w.ERows * w.ECols * costC
                        + (double)rows * columns * costF);
                    if (work < bestWork)
                    {
                        bestWork = work;
                        best = (rows, columns);
                    }

                    // Narrower blocks of these rows only fit better and cost more, but for the rounds of threads.
                    if (Shapes.F[0] * blocks >= threads) break;
                }
            }

            return best;
        }
    }
}
