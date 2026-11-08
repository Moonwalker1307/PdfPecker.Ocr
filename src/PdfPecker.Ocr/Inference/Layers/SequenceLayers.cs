using System.Numerics;
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

    private int TileColumns() => Tiling.Balanced(_n, Math.Clamp(32768 / _k / _gemm.NR * _gemm.NR, _gemm.NR, 1024), _gemm.NR);

    public override long SharedScratch(Step step) => _gemm.PackedASize(Rows.Count(step.OutputShape), _k);

    public override void Execute(Step step, InferenceContext context)
    {
        int rows = Rows.Count(step.OutputShape);
        _gemm.PackARows(step.Input(0), _k, rows, _k, context.SharedScratch);
        step.Tiling ??= Tiling.For(1, _n, TileColumns(), rows, _gemm.MR);
        context.For(step.Tiling.Count, step, &Tile);
    }

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

    public override long ScratchPerWorker(Step step, InferenceContext context) => step.InputShapes[0][1] + 64;

    public override void Execute(Step step, InferenceContext context) =>
        context.For(step.InputShapes[0][0] * _heads, step, &Head);

    private static void Head(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (AttentionLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int t = x[1], c = x[2] / 3, heads = layer._heads, d = c / heads;
        int n = tile / heads, h = tile % heads;
        float* qkv = step.Input(0) + (long)n * t * 3 * c;
        float* output = step.Output + (long)n * t * c;
        float* scores = step.Context.Scratch(worker);
        float* q = stackalloc float[d];
        for (int t1 = 0; t1 < t; t1++)
        {
            float* qRow = qkv + (long)t1 * 3 * c + h * d;
            for (int i = 0; i < d; i++) q[i] = qRow[i] * layer._scale;
            for (int t2 = 0; t2 < t; t2++)
            {
                float* kRow = qkv + (long)t2 * 3 * c + c + h * d;
                float acc = 0f;
                for (int i = 0; i < d; i++) acc += q[i] * kRow[i];
                scores[t2] = acc;
            }
            VectorMath.Softmax(scores, t);
            float* outRow = output + (long)t1 * c + h * d;
            for (int i = 0; i < d; i++) outRow[i] = 0f;
            for (int t2 = 0; t2 < t; t2++)
            {
                float p = scores[t2];
                float* vRow = qkv + (long)t2 * 3 * c + 2 * c + h * d;
                for (int i = 0; i < d; i++) outRow[i] += p * vRow[i];
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
