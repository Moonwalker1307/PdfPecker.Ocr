using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// Elementwise sum or product. With upsampling factors the second input is a smaller NCHW map that is enlarged by
/// nearest neighbour on the fly (the feature pyramid's Resize then Add); otherwise the inputs broadcast as in NumPy.
/// </summary>
internal sealed unsafe class BinaryLayer : Layer
{
    private const int ChunkSize = 1 << 16;
    private readonly bool _multiply;
    private readonly int _fh, _fw;

    public BinaryLayer(NodeDefinition node, bool multiply)
        : base(node)
    {
        _multiply = multiply;
        _fh = Int(node, 0);
        _fw = Int(node, 1);
        if (_fh < 1 || _fw < 1 || node.Inputs.Length != 2) throw new InvalidDataException("Bad elementwise attributes.");
    }

    private bool Upsamples => _fh > 1 || _fw > 1;

    public override int[] OutputShape(int[][] inputs)
    {
        int[] a = inputs[0], b = inputs[1];
        if (Upsamples)
        {
            RequireRank(a, 4, "Upsampling addition");
            RequireRank(b, 4, "Upsampling addition");
            if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2] * _fh || a[3] != b[3] * _fw)
                throw Unsupported("The upsampled input does not match the other input.");
            return (int[])a.Clone();
        }
        int rank = Math.Max(a.Length, b.Length);
        var output = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int da = i - (rank - a.Length) >= 0 ? a[i - (rank - a.Length)] : 1;
            int db = i - (rank - b.Length) >= 0 ? b[i - (rank - b.Length)] : 1;
            if (da != db && da != 1 && db != 1) throw Unsupported("The inputs of an elementwise operation do not broadcast.");
            output[i] = Math.Max(da, db);
        }
        return output;
    }

    public override int InPlaceInput(int[][] inputs, int[] output)
    {
        if (inputs[0].AsSpan().SequenceEqual(output)) return 0;
        if (!Upsamples && inputs[1].AsSpan().SequenceEqual(output)) return 1;
        return -1;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int[] a = step.InputShapes[0], b = step.InputShapes[1], o = step.OutputShape;
        if (Upsamples)
        {
            context.For(o[0] * o[1], step, &UpsampledPlane);
        }
        else if (a.AsSpan().SequenceEqual(b))
        {
            context.For((Tensor.ElementCount(o) + ChunkSize - 1) / ChunkSize, step, &SameShapeChunk);
        }
        else
        {
            Broadcast(step.Input(0), a, step.Input(1), b, step.Output, o);
        }
    }

    private static void SameShapeChunk(object state, int chunk, int worker)
    {
        var step = (Step)state;
        var layer = (BinaryLayer)step.Layer;
        long start = (long)chunk * ChunkSize;
        int length = (int)Math.Min(ChunkSize, Tensor.ElementCount(step.OutputShape) - start);
        if (layer._multiply) VectorMath.Multiply(step.Input(0) + start, step.Input(1) + start, step.Output + start, length);
        else VectorMath.Add(step.Input(0) + start, step.Input(1) + start, step.Output + start, length);
    }

    private static void UpsampledPlane(object state, int plane, int worker)
    {
        var step = (Step)state;
        var layer = (BinaryLayer)step.Layer;
        int[] o = step.OutputShape, b = step.InputShapes[1];
        int h = o[2], w = o[3], bw = b[3];
        float* a = step.Input(0) + (long)plane * h * w;
        float* small = step.Input(1) + (long)plane * b[2] * bw;
        float* output = step.Output + (long)plane * h * w;
        for (int y = 0; y < h; y++)
        {
            float* src = small + (long)(y / layer._fh) * bw;
            float* ar = a + (long)y * w, orow = output + (long)y * w;
            if (layer._multiply)
                for (int x = 0; x < w; x++) orow[x] = ar[x] * src[x / layer._fw];
            else
                for (int x = 0; x < w; x++) orow[x] = ar[x] + src[x / layer._fw];
        }
    }

    private void Broadcast(float* a, int[] sa, float* b, int[] sb, float* output, int[] so)
    {
        int rank = so.Length;
        if (rank > 8) throw Unsupported("Broadcasting supports up to 8 dimensions.");
        Span<long> stridesA = stackalloc long[rank];
        Span<long> stridesB = stackalloc long[rank];
        BroadcastStrides(sa, so, stridesA);
        BroadcastStrides(sb, so, stridesB);
        int inner = rank == 0 ? 1 : so[^1];
        long outer = rank == 0 ? 1 : Tensor.ElementCount(so) / Math.Max(inner, 1);
        long ia = rank == 0 ? 0 : stridesA[^1], ib = rank == 0 ? 0 : stridesB[^1];
        Span<int> index = stackalloc int[rank];
        for (long o = 0; o < outer; o++)
        {
            long offA = 0, offB = 0;
            for (int d = 0; d < rank - 1; d++)
            {
                offA += index[d] * stridesA[d];
                offB += index[d] * stridesB[d];
            }
            float* dst = output + o * inner;
            if (ia == 1 && ib == 1)
            {
                if (_multiply) VectorMath.Multiply(a + offA, b + offB, dst, inner);
                else VectorMath.Add(a + offA, b + offB, dst, inner);
            }
            else if (ia == 1 && ib == 0)
            {
                if (_multiply) VectorMath.MultiplyScalar(a + offA, b[offB], dst, inner);
                else VectorMath.AddScalar(a + offA, b[offB], dst, inner);
            }
            else if (ia == 0 && ib == 1)
            {
                if (_multiply) VectorMath.MultiplyScalar(b + offB, a[offA], dst, inner);
                else VectorMath.AddScalar(b + offB, a[offA], dst, inner);
            }
            else
            {
                for (int i = 0; i < inner; i++)
                {
                    float va = a[offA + i * ia], vb = b[offB + i * ib];
                    dst[i] = _multiply ? va * vb : va + vb;
                }
            }
            for (int d = rank - 2; d >= 0; d--)
            {
                if (++index[d] < so[d]) break;
                index[d] = 0;
            }
        }
    }

    /// <summary>Strides of an input aligned to the output's rank, zero along broadcast dimensions.</summary>
    private static void BroadcastStrides(int[] shape, int[] output, Span<long> strides)
    {
        int rank = output.Length;
        long stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            int j = i - (rank - shape.Length);
            int d = j >= 0 ? shape[j] : 1;
            strides[i] = d == 1 && output[i] != 1 ? 0 : stride;
            stride *= d;
        }
    }
}

/// <summary>A standalone activation, applied in chunks.</summary>
internal sealed unsafe class ActivationLayer : Layer
{
    private const int ChunkSize = 1 << 16;
    private readonly ActivationKind _kind;
    private readonly float _alpha, _beta;

    public ActivationLayer(NodeDefinition node)
        : base(node)
    {
        _kind = (ActivationKind)Int(node, 0);
        _alpha = Float(node, 0);
        _beta = Float(node, 1);
    }

    public override int[] OutputShape(int[][] inputs) => (int[])inputs[0].Clone();

    public override int InPlaceInput(int[][] inputs, int[] output) => 0;

    public override void Execute(Step step, InferenceContext context) =>
        context.For((Tensor.ElementCount(step.OutputShape) + ChunkSize - 1) / ChunkSize, step, &Chunk);

    private static void Chunk(object state, int chunk, int worker)
    {
        var step = (Step)state;
        var layer = (ActivationLayer)step.Layer;
        long start = (long)chunk * ChunkSize;
        int length = (int)Math.Min(ChunkSize, Tensor.ElementCount(step.OutputShape) - start);
        float* input = step.Input(0) + start, output = step.Output + start;
        if (input != output) new Span<float>(input, length).CopyTo(new Span<float>(output, length));
        VectorMath.Activate(output, length, layer._kind, layer._alpha, layer._beta);
    }
}

/// <summary>Nearest-neighbour upsampling of NCHW planes by integer factors.</summary>
internal sealed unsafe class UpsampleLayer : Layer
{
    private readonly int _fh, _fw;

    public UpsampleLayer(NodeDefinition node)
        : base(node)
    {
        _fh = Int(node, 0);
        _fw = Int(node, 1);
        if (_fh < 1 || _fw < 1) throw new InvalidDataException("Bad upsampling factors.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        RequireRank(x, 4, "Upsampling");
        return [x[0], x[1], checked(x[2] * _fh), checked(x[3] * _fw)];
    }

    public override void Execute(Step step, InferenceContext context) =>
        context.For(step.OutputShape[0] * step.OutputShape[1], step, &Plane);

    private static void Plane(object state, int plane, int worker)
    {
        var step = (Step)state;
        var layer = (UpsampleLayer)step.Layer;
        int[] x = step.InputShapes[0];
        Upsample(step.Input(0) + (long)plane * x[2] * x[3], x[2], x[3], layer._fh, layer._fw, step.Output + (long)plane * x[2] * layer._fh * x[3] * layer._fw);
    }

    /// <summary>Writes one plane enlarged by (fh, fw) to dst, whose rows are w * fw long.</summary>
    public static void Upsample(float* src, int h, int w, int fh, int fw, float* dst)
    {
        int ow = w * fw;
        for (int y = 0; y < h; y++)
        {
            float* row = dst + (long)y * fh * ow;
            float* s = src + (long)y * w;
            for (int x = 0; x < w; x++)
            {
                float v = s[x];
                for (int i = 0; i < fw; i++) row[x * fw + i] = v;
            }
            for (int i = 1; i < fh; i++) new Span<float>(row, ow).CopyTo(new Span<float>(row + (long)i * ow, ow));
        }
    }
}
