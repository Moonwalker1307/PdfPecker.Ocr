using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>Helpers for layers that work on rows of the last axis.</summary>
internal static class Rows
{
    /// <summary>Rows per tile, so a tile holds at least a few thousand values; it depends only on the shape.</summary>
    public static int PerTile(int rowLength) => Math.Max(1, 8192 / Math.Max(rowLength, 1));

    public static int Count(int[] shape) => shape.Length == 0 ? 1 : Tensor.ElementCount(shape) / Math.Max(shape[^1], 1);
}

/// <summary>
/// Layer normalisation over the last axis, computed as the original graph does: d = x - mean, var = mean(d * d),
/// y = d / sqrt(var + eps) * gamma + beta.
/// </summary>
internal sealed unsafe class LayerNormLayer : Layer
{
    private readonly float[] _gamma, _beta;
    private readonly float _epsilon;

    public LayerNormLayer(NodeDefinition node, ModelDefinition model)
        : base(node)
    {
        _epsilon = Float(node, 0);
        _gamma = Weight(node, model, 0).Data;
        _beta = Weight(node, model, 1).Data;
        if (_gamma.Length != _beta.Length) throw new InvalidDataException("Layer normalisation weights do not match.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        if (x.Length == 0 || x[^1] != _gamma.Length) throw Unsupported($"Layer normalisation expects {_gamma.Length} features.");
        return (int[])x.Clone();
    }

    public override int InPlaceInput(int[][] inputs, int[] output) => 0;

    public override void Execute(Step step, InferenceContext context)
    {
        int c = _gamma.Length;
        int perTile = Rows.PerTile(c);
        context.For((Rows.Count(step.OutputShape) + perTile - 1) / perTile, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (LayerNormLayer)step.Layer;
        int c = layer._gamma.Length;
        int perTile = Rows.PerTile(c);
        int rows = Rows.Count(step.OutputShape);
        int width = Vector<float>.Count;
        fixed (float* gamma = layer._gamma, beta = layer._beta)
        {
            for (int r = tile * perTile, end = Math.Min(rows, r + perTile); r < end; r++)
            {
                float* x = step.Input(0) + (long)r * c;
                float* y = step.Output + (long)r * c;
                float mean = VectorMath.Sum(x, c) / c;
                var vmean = new Vector<float>(mean);
                var acc = Vector<float>.Zero;
                int i = 0;
                for (; i <= c - width; i += width)
                {
                    var d = VectorMath.Load(x + i) - vmean;
                    acc += d * d;
                }
                float squares = Vector.Sum(acc);
                for (; i < c; i++)
                {
                    float d = x[i] - mean;
                    squares += d * d;
                }
                float deviation = MathF.Sqrt(squares / c + layer._epsilon);
                var vdev = new Vector<float>(deviation);
                for (i = 0; i <= c - width; i += width)
                {
                    var d = VectorMath.Load(x + i) - vmean;
                    VectorMath.Store(d / vdev * VectorMath.Load(gamma + i) + VectorMath.Load(beta + i), y + i);
                }
                for (; i < c; i++) y[i] = (x[i] - mean) / deviation * gamma[i] + beta[i];
            }
        }
    }
}

/// <summary>
/// x [.., K] times constant weights [K, N] plus bias, with an activation and an optional residual. The activation rows
/// are packed once into shared scratch; tiles then run over blocks of output columns and rows.
/// </summary>
internal sealed unsafe class LinearLayer : Layer
{
    private readonly GemmKernel _gemm;
    private readonly float[] _weights;
    private readonly float[] _bias;
    private readonly int _k, _n;
    private readonly ActivationKind _activation;
    private readonly float _alpha, _beta;
    private readonly bool _hasResidual;

    public LinearLayer(NodeDefinition node, ModelDefinition model, GemmKernel gemm)
        : base(node)
    {
        _gemm = gemm;
        _activation = (ActivationKind)Int(node, 0);
        _hasResidual = Int(node, 1) != 0;
        _alpha = Float(node, 0);
        _beta = Float(node, 1);
        var w = Weight(node, model, 0);
        if (w.Shape.Length != 2) throw new InvalidDataException("Linear weights must be a matrix.");
        _k = w.Shape[0];
        _n = w.Shape[1];
        _weights = gemm.PackB(w.Data, _k, _n);
        var b = Weight(node, model, 1).Data;
        if (b.Length != _n) throw new InvalidDataException("Linear bias does not match.");
        _bias = gemm.PadBias(b, _n);
        if (node.Inputs.Length != (_hasResidual ? 2 : 1)) throw new InvalidDataException("Linear has the wrong number of inputs.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        if (x.Length == 0 || x[^1] != _k) throw Unsupported($"Linear expects {_k} features.");
        var output = (int[])x.Clone();
        output[^1] = _n;
        if (_hasResidual && !inputs[1].AsSpan().SequenceEqual(output)) throw Unsupported("The residual does not match the linear output.");
        return output;
    }

    /// <summary>True when the layer is a plain product plus bias, with no activation or residual.</summary>
    public bool IsPlain => _activation == ActivationKind.None && !_hasResidual;

    /// <summary>The input features.</summary>
    public int InputFeatures => _k;

    /// <summary>The output features.</summary>
    public int OutputFeatures => _n;

    /// <summary>The kernel the weights are packed for.</summary>
    public GemmKernel Gemm => _gemm;

    /// <summary>The weights in the kernel's B panels, which <see cref="ArgMaxHeadLayer"/> shares.</summary>
    public float[] PackedWeights => _weights;

    /// <summary>The bias padded to whole panels.</summary>
    public float[] PaddedBias => _bias;

    private int TileColumns() => Tiling.Balanced(_n, Math.Clamp(32768 / _k / _gemm.NR * _gemm.NR, _gemm.NR, 1024), _gemm.NR);

    /// <summary>The tiles of this layer's product for the given number of rows.</summary>
    public Tiling ColumnTiling(int rows) => Tiling.For(1, _n, TileColumns(), rows, _gemm.MR);

    public override long SharedScratch(Step step) => _gemm.PackedASize(Rows.Count(step.OutputShape), _k);

    public override void Execute(Step step, InferenceContext context)
    {
        int rows = Rows.Count(step.OutputShape);
        var pack = step.State as RowPacking ?? (RowPacking)(step.State = new RowPacking(_gemm));
        pack.Run(context, step.Input(0), _k, rows, _k, context.SharedScratch);
        step.Tiling ??= Tiling.For(1, _n, TileColumns(), rows, _gemm.MR);
        context.For(step.Tiling.Count, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (LinearLayer)step.Layer;
        step.Tiling!.Decode(tile, out _, out int n0, out int columns, out int m0, out int rows);
        int n = layer._n, k = layer._k;
        long offset = (long)m0 * n + n0;
        var epilogue = new Epilogue(layer._activation, layer._alpha, layer._beta, layer._hasResidual ? step.Input(1) + offset : null);
        fixed (float* weights = layer._weights, bias = layer._bias)
            layer._gemm.Multiply(k, step.Context.SharedScratch + (long)m0 * k, rows, weights + (long)n0 * k, columns, step.Output + offset, n, null, bias + n0, epilogue);
    }
}

/// <summary>
/// Packs the rows of a matrix product's activations into the kernel's row panels on the context's threads, a few panels
/// per item. The recogniser's linear layers packed up to 600 rows on one thread while the others waited, which was most
/// of what kept them from scaling with threads. Each panel is packed alone, so the result does not depend on the split.
/// </summary>
internal sealed unsafe class RowPacking(GemmKernel gemm)
{
    private const int PanelsPerItem = 4;
    private readonly GemmKernel _gemm = gemm;
    private float* _source, _destination;
    private nint _ld;
    private int _rows, _k;

    public void Run(InferenceContext context, float* source, nint ld, int rows, int k, float* destination)
    {
        int block = PanelsPerItem * _gemm.MR;
        int items = (rows + block - 1) / block;
        if (items <= 1 || context.MaxDegreeOfParallelism == 1)
        {
            _gemm.PackARows(source, ld, rows, k, destination);
            return;
        }

        _source = source;
        _ld = ld;
        _rows = rows;
        _k = k;
        _destination = destination;
        try
        {
            context.For(items, this, &Item);
        }
        finally
        {
            _source = _destination = null;
        }
    }

    private static void Item(object state, int item, int worker)
    {
        var job = (RowPacking)state;
        int block = PanelsPerItem * job._gemm.MR;
        int first = item * block;
        job._gemm.PackARows(job._source + (first * job._ld), job._ld, Math.Min(block, job._rows - first), job._k, job._destination + ((nint)first * job._k));
    }
}

/// <summary>
/// Multi-head self-attention on packed query, key and value: input [N, T, 3C] holds q, k and v for H heads of C / H
/// features each; per head, scores = (q * scale) k^T, softmax over keys, then times v. Output [N, T, C].
/// </summary>
internal sealed unsafe class AttentionLayer : Layer
{
    private readonly int _heads;
    private readonly float _scale;

    public AttentionLayer(NodeDefinition node)
        : base(node)
    {
        _heads = Int(node, 0);
        _scale = Float(node, 0);
        if (_heads < 1) throw new InvalidDataException("Bad attention head count.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        RequireRank(x, 3, "Attention");
        if (x[2] % (3 * _heads) != 0) throw Unsupported("Attention features do not divide into heads.");
        return [x[0], x[1], x[2] / 3];
    }

    private static readonly int Width = Vector<float>.Count;

    // The lanes the head's loops take at a time: 512-bit vectors where the processor has AVX-512 (Vector<T> stays 256
    // bits wide there), Vector<T> otherwise. Each lane is one score or one output value with its own sum, so the width
    // does not change any result.
    private static readonly bool Wide = VectorMath.Wide && Width == 8;
    private static readonly int Lanes = Wide ? 16 : Width;

    /// <summary>The scratch of one head: q, k transposed, v with rows padded to whole vectors, and one row of scores.</summary>
    private static (int KeyStride, int ValueStride, long Floats) Layout(int t, int d)
    {
        int keyStride = GemmKernel.RoundUp(t, Lanes);
        int valueStride = GemmKernel.RoundUp(d, Lanes);
        return (keyStride, valueStride, d + ((long)d * keyStride) + ((long)t * valueStride) + keyStride);
    }

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int[] x = step.InputShapes[0];
        return Layout(x[1], x[2] / 3 / _heads).Floats + 64;
    }

    public override void Execute(Step step, InferenceContext context) =>
        context.For(step.InputShapes[0][0] * _heads, step, &Head);

    /// <summary>
    /// One head of one batch item. Each score and each output value is the same sum, in the same order and with the
    /// same roundings, as the plain loops sum(q[i] * k[i]) over features and sum(p * v) over keys; the vectors only
    /// compute several of them at once, scores across keys (from k transposed) and outputs across features.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Head(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (AttentionLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int t = x[1], c = x[2] / 3, heads = layer._heads, d = c / heads;
        int n = tile / heads, h = tile % heads;
        var (keyStride, valueStride, _) = Layout(t, d);
        float* qkv = step.Input(0) + (long)n * t * 3 * c;
        float* output = step.Output + (long)n * t * c;
        float* q = step.Context.Scratch(worker);
        float* keys = q + d;
        float* values = keys + ((long)d * keyStride);
        float* scores = values + ((long)t * valueStride);

        // Keys transposed, so that one vector holds one feature of consecutive keys, and values padded with zeros to
        // whole vectors; the padding is never stored to the output.
        for (int t2 = 0; t2 < t; t2++)
        {
            float* row = qkv + ((long)t2 * 3 * c) + c + (h * d);
            for (int i = 0; i < d; i++) keys[((long)i * keyStride) + t2] = row[i];
            float* value = values + ((long)t2 * valueStride);
            new Span<float>(row + c, d).CopyTo(new Span<float>(value, d));
            new Span<float>(value + d, valueStride - d).Clear();
        }

        for (int i = 0; i < d; i++) new Span<float>(keys + ((long)i * keyStride) + t, keyStride - t).Clear();

        if (Wide)
        {
            HeadWide(qkv, output, q, keys, values, scores, t, c, d, h, keyStride, valueStride, layer._scale);
            return;
        }

        for (int t1 = 0; t1 < t; t1++)
        {
            float* qRow = qkv + ((long)t1 * 3 * c) + (h * d);
            for (int i = 0; i < d; i++) q[i] = qRow[i] * layer._scale;
            for (int t2 = 0; t2 < keyStride; t2 += Width)
            {
                var acc = Vector<float>.Zero;
                for (int i = 0; i < d; i++) acc += new Vector<float>(q[i]) * VectorMath.Load(keys + ((long)i * keyStride) + t2);
                VectorMath.Store(acc, scores + t2);
            }

            VectorMath.Softmax(scores, t);
            float* outRow = output + ((long)t1 * c) + (h * d);
            for (int i = 0; i < d; i += Width)
            {
                var acc = Vector<float>.Zero;
                for (int t2 = 0; t2 < t; t2++) acc += new Vector<float>(scores[t2]) * VectorMath.Load(values + ((long)t2 * valueStride) + i);
                if (i + Width <= d)
                {
                    VectorMath.Store(acc, outRow + i);
                }
                else
                {
                    for (int j = 0; j < d - i; j++) outRow[i + j] = acc[j];
                }
            }
        }
    }

    /// <summary>
    /// The scores and outputs of <see cref="Head"/> on 512-bit vectors, the same products and sums in the same order
    /// for each lane.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void HeadWide(float* qkv, float* output, float* q, float* keys, float* values, float* scores, int t, int c, int d, int h, int keyStride, int valueStride, float scale)
    {
        float* tail = stackalloc float[16];
        for (int t1 = 0; t1 < t; t1++)
        {
            float* qRow = qkv + ((long)t1 * 3 * c) + (h * d);
            for (int i = 0; i < d; i++) q[i] = qRow[i] * scale;
            for (int t2 = 0; t2 < keyStride; t2 += 16)
            {
                var acc = Vector512<float>.Zero;
                for (int i = 0; i < d; i++) acc += Vector512.Create(q[i]) * Vector512.Load(keys + ((long)i * keyStride) + t2);
                Vector512.Store(acc, scores + t2);
            }

            VectorMath.Softmax(scores, t);
            float* outRow = output + ((long)t1 * c) + (h * d);
            for (int i = 0; i < d; i += 16)
            {
                var acc = Vector512<float>.Zero;
                for (int t2 = 0; t2 < t; t2++) acc += Vector512.Create(scores[t2]) * Vector512.Load(values + ((long)t2 * valueStride) + i);
                if (i + 16 <= d)
                {
                    Vector512.Store(acc, outRow + i);
                }
                else
                {
                    Vector512.Store(acc, tail);
                    new Span<float>(tail, d - i).CopyTo(new Span<float>(outRow + i, d - i));
                }
            }
        }
    }
}

/// <summary>Softmax over the last axis, row by row in place.</summary>
internal sealed unsafe class SoftmaxLayer(NodeDefinition node) : Layer(node)
{
    public override int[] OutputShape(int[][] inputs) =>
        inputs[0].Length == 0 ? throw Unsupported("Softmax needs at least one dimension.") : (int[])inputs[0].Clone();

    public override int InPlaceInput(int[][] inputs, int[] output) => 0;

    public override void Execute(Step step, InferenceContext context)
    {
        int perTile = Rows.PerTile(step.OutputShape[^1]);
        context.For((Rows.Count(step.OutputShape) + perTile - 1) / perTile, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        int c = step.OutputShape[^1];
        int perTile = Rows.PerTile(c);
        int rows = Rows.Count(step.OutputShape);
        for (int r = tile * perTile, end = Math.Min(rows, r + perTile); r < end; r++)
        {
            float* x = step.Input(0) + (long)r * c, y = step.Output + (long)r * c;
            if (x != y) new Span<float>(x, c).CopyTo(new Span<float>(y, c));
            VectorMath.Softmax(y, c);
        }
    }
}

/// <summary>
/// A <see cref="LinearLayer"/> followed by a softmax over its output, reduced to what greedy decoding reads: for each row,
/// the index of the most probable class and its probability, as [.., 2]. Both values are bit for bit what the two layers
/// and a search for the first largest probability give (see <see cref="VectorMath.SoftmaxArgMax"/>), but the
/// probabilities of all classes are never kept: the recogniser's 18,710 classes made an output of 75 KB per time step,
/// 45 to 180 MB for a batch of long lines, which the decoder then read once more.
/// </summary>
/// <remarks>
/// The rows go in chunks: the linear layer writes a chunk's logits into shared scratch, tiled over blocks of output
/// columns as <see cref="LinearLayer"/> is, so a block of weights stays in the cache for every row of the chunk; then
/// the rows of the chunk are reduced in parallel. A value of the product does not depend on the tile it lands in (see
/// <see cref="GemmKernel"/>), and each row is reduced on its own, so results do not depend on the chunks or threads.
/// </remarks>
internal sealed unsafe class ArgMaxHeadLayer : Layer
{
    // Rows per chunk: 8 panels of 12 rows hold 7 MB of logits for 18,710 classes, which stay in a large cache, and read
    // the weights once for 96 rows.
    private const int ChunkRows = 96;

    // Rows per reduction tile, enough to give every thread several tiles of a chunk.
    private const int RowsPerTile = 4;

    private readonly LinearLayer _linear;
    private readonly int _stride;

    public ArgMaxHeadLayer(LinearLayer linear, NodeDefinition node)
        : base(node)
    {
        _linear = linear;
        _stride = GemmKernel.RoundUp(linear.OutputFeatures, 16);
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        if (x.Length == 0 || x[^1] != _linear.InputFeatures) throw Unsupported($"Linear expects {_linear.InputFeatures} features.");
        var output = (int[])x.Clone();
        output[^1] = 2;
        return output;
    }

    // The packed rows come first in the shared scratch, and the chunk's logits after them on a 64-byte boundary.
    private long PackedSize(int rows) => (_linear.Gemm.PackedASize(rows, _linear.InputFeatures) + 15) / 16 * 16;

    public override long SharedScratch(Step step)
    {
        int rows = Rows.Count(step.OutputShape);
        return PackedSize(rows) + (long)Math.Min(rows, ChunkRows) * _stride;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int rows = Rows.Count(step.OutputShape);
        var chunk = step.State as Chunk ?? (Chunk)(step.State = new Chunk(step, _linear.Gemm));
        chunk.Packing.Run(context, step.Input(0), _linear.InputFeatures, rows, _linear.InputFeatures, context.SharedScratch);
        chunk.Logits = context.SharedScratch + PackedSize(rows);
        for (int m0 = 0; m0 < rows; m0 += ChunkRows)
        {
            chunk.First = m0;
            chunk.Count = Math.Min(ChunkRows, rows - m0);
            chunk.Tiling = _linear.ColumnTiling(chunk.Count);
            context.For(chunk.Tiling.Count, chunk, &Product);
            context.For((chunk.Count + RowsPerTile - 1) / RowsPerTile, chunk, &Reduce);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Product(object state, int tile, int worker)
    {
        var chunk = (Chunk)state;
        var layer = (ArgMaxHeadLayer)chunk.Step.Layer;
        var linear = layer._linear;
        int k = linear.InputFeatures;
        chunk.Tiling!.Decode(tile, out _, out int n0, out int columns, out int m0, out int rows);
        float* a = chunk.Step.Context.SharedScratch + (long)(chunk.First + m0) * k;
        fixed (float* weights = linear.PackedWeights, bias = linear.PaddedBias)
            linear.Gemm.Multiply(k, a, rows, weights + (long)n0 * k, columns, chunk.Logits + (long)m0 * layer._stride + n0, layer._stride, null, bias + n0);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Reduce(object state, int tile, int worker)
    {
        var chunk = (Chunk)state;
        var layer = (ArgMaxHeadLayer)chunk.Step.Layer;
        int n = layer._linear.OutputFeatures;
        float* output = chunk.Step.Output + (long)chunk.First * 2;
        for (int r = tile * RowsPerTile, end = Math.Min(chunk.Count, r + RowsPerTile); r < end; r++)
        {
            int index = VectorMath.SoftmaxArgMax(chunk.Logits + (long)r * layer._stride, n, out float probability);
            output[2 * r] = index;
            output[(2 * r) + 1] = probability;
        }
    }

    /// <summary>The rows of one chunk and where its logits go, handed to the tiles; kept with the step between runs.</summary>
    private sealed class Chunk(Step step, GemmKernel gemm)
    {
        public Step Step { get; } = step;

        public RowPacking Packing { get; } = new(gemm);

        public float* Logits { get; set; }

        public int First { get; set; }

        public int Count { get; set; }

        public Tiling? Tiling { get; set; }
    }
}
