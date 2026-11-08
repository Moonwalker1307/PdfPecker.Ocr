using System.Runtime.CompilerServices;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// Concatenation along an axis, where each input may first be enlarged by nearest neighbour. When nothing precedes the
/// axis but dimensions of size 1, each input's slice of the output is contiguous, and the plan lets the layers that
/// produce the inputs write straight into it; those inputs need no copy here.
/// </summary>
internal sealed unsafe class ConcatLayer : Layer
{
    private readonly int _axis;
    private readonly (int H, int W)[] _factors;

    public ConcatLayer(NodeDefinition node)
        : base(node)
    {
        _axis = Int(node, 0);
        _factors = new (int, int)[node.Inputs.Length];
        for (int i = 0; i < _factors.Length; i++)
        {
            _factors[i] = (Int(node, 1 + 2 * i), Int(node, 2 + 2 * i));
            if (_factors[i].H < 1 || _factors[i].W < 1) throw new InvalidDataException("Bad concatenation factors.");
        }
        if (node.Inputs.Length == 0) throw new InvalidDataException("Concatenation without inputs.");
    }

    public (int H, int W) Factor(int input) => _factors[input];

    private int Axis(int rank) => _axis < 0 ? _axis + rank : _axis;

    private int[] Enlarged(int[] shape, int input)
    {
        var (fh, fw) = _factors[input];
        if (fh == 1 && fw == 1) return shape;
        RequireRank(shape, 4, "Upsampling concatenation");
        return [shape[0], shape[1], checked(shape[2] * fh), checked(shape[3] * fw)];
    }

    public override int[] OutputShape(int[][] inputs)
    {
        int[] first = Enlarged(inputs[0], 0);
        int rank = first.Length;
        int axis = Axis(rank);
        if ((uint)axis >= (uint)rank) throw Unsupported("Concatenation axis out of range.");
        var output = (int[])first.Clone();
        for (int i = 1; i < inputs.Length; i++)
        {
            int[] s = Enlarged(inputs[i], i);
            if (s.Length != rank) throw Unsupported("Concatenated inputs differ in rank.");
            for (int d = 0; d < rank; d++)
                if (d != axis && s[d] != first[d]) throw Unsupported("Concatenated inputs differ in shape.");
            output[axis] = checked(output[axis] + s[axis]);
        }
        return output;
    }

    /// <summary>True when each input's slice of the output is one contiguous block.</summary>
    public bool CanAliasInputs(Step step)
    {
        int axis = Axis(step.OutputShape.Length);
        for (int d = 0; d < axis; d++)
            if (step.OutputShape[d] != 1) return false;
        return true;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int[] o = step.OutputShape;
        if (o.Length == 4 && Axis(4) == 1) context.For(o[0] * o[1], step, &Plane);
        else Generic(step);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Plane(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (ConcatLayer)step.Layer;
        int[] o = step.OutputShape;
        int n = tile / o[1], c = tile % o[1];
        long plane = (long)o[2] * o[3];
        float* dst = step.Output + (long)tile * plane;
        int input = 0;
        while (c >= step.InputShapes[input][1])
        {
            c -= step.InputShapes[input][1];
            input++;
        }
        int[] s = step.InputShapes[input];
        float* src = step.Input(input) + ((long)n * s[1] + c) * s[2] * s[3];
        var (fh, fw) = layer._factors[input];
        if (fh == 1 && fw == 1)
        {
            if (src != dst) new Span<float>(src, (int)plane).CopyTo(new Span<float>(dst, (int)plane));
        }
        else
        {
            UpsampleLayer.Upsample(src, s[2], s[3], fh, fw, dst);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Generic(Step step)
    {
        int[] o = step.OutputShape;
        int axis = Axis(o.Length);
        long outer = 1;
        for (int d = 0; d < axis; d++) outer *= o[d];
        long outInner = Tensor.ElementCount(o) / Math.Max(outer, 1);
        float* output = step.Output;
        for (long i = 0; i < outer; i++)
        {
            long at = i * outInner;
            for (int j = 0; j < step.InputShapes.Length; j++)
            {
                if (_factors[j] != (1, 1)) throw Unsupported("Upsampling concatenation needs NCHW channels.");
                long chunk = Tensor.ElementCount(step.InputShapes[j]) / Math.Max(outer, 1);
                float* src = step.Input(j) + i * chunk;
                if (src != output + at) Buffer.MemoryCopy(src, output + at, chunk * sizeof(float), chunk * sizeof(float));
                at += chunk;
            }
        }
    }
}

/// <summary>Axis permutation of a small tensor.</summary>
internal sealed unsafe class TransposeLayer : Layer
{
    private readonly int[] _perm;

    public TransposeLayer(NodeDefinition node)
        : base(node)
    {
        _perm = node.Ints;
        var seen = new bool[_perm.Length];
        foreach (int p in _perm)
        {
            if ((uint)p >= (uint)_perm.Length || seen[p]) throw new InvalidDataException("Bad permutation.");
            seen[p] = true;
        }
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        if (x.Length != _perm.Length) throw Unsupported("Transpose rank does not match.");
        return _perm.Select(p => x[p]).ToArray();
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int[] x = step.InputShapes[0];
        int rank = x.Length;
        if (rank > 8) throw Unsupported("Transpose supports up to 8 dimensions.");
        Span<long> inStrides = stackalloc long[rank];
        long stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            inStrides[d] = stride;
            stride *= x[d];
        }
        // The input stride for each output axis; the output is written in order.
        int[] o = step.OutputShape;
        Span<long> strides = stackalloc long[rank];
        for (int d = 0; d < rank; d++) strides[d] = inStrides[_perm[d]];
        float* input = step.Input(0), output = step.Output;
        long total = Tensor.ElementCount(o);
        int inner = rank == 0 ? 1 : o[^1];
        long innerStride = rank == 0 ? 1 : strides[^1];
        Span<int> index = stackalloc int[rank];
        for (long done = 0; done < total; done += inner)
        {
            long offset = 0;
            for (int d = 0; d < rank - 1; d++) offset += index[d] * strides[d];
            float* dst = output + done;
            for (int i = 0; i < inner; i++) dst[i] = input[offset + i * innerStride];
            for (int d = rank - 2; d >= 0; d--)
            {
                if (++index[d] < o[d]) break;
                index[d] = 0;
            }
        }
    }
}

/// <summary>A reshape: the plan makes the output share the input's memory, so it costs nothing.</summary>
internal sealed unsafe class ReshapeLayer(NodeDefinition node) : Layer(node)
{
    public override bool IsView => true;

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        int[] spec = Node.Ints;
        var output = new int[spec.Length];
        int infer = -1;
        long known = 1;
        for (int i = 0; i < spec.Length; i++)
        {
            int d = spec[i];
            if (d == 0)
            {
                if (i >= x.Length) throw Unsupported("Reshape copies a dimension the input lacks.");
                d = x[i];
            }
            if (d == -1)
            {
                if (infer >= 0) throw new InvalidDataException("Reshape infers more than one dimension.");
                infer = i;
                continue;
            }
            if (d < 0) throw new InvalidDataException("Bad reshape dimension.");
            output[i] = d;
            known *= d;
        }
        long total = Tensor.ElementCount(x);
        if (infer >= 0)
        {
            if (known == 0 || total % known != 0) throw Unsupported("Reshape cannot infer a dimension.");
            output[infer] = (int)(total / known);
        }
        else if (known != total)
        {
            throw Unsupported("Reshape changes the number of values.");
        }
        return output;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        // The plan shares memory; copy only if it could not.
        float* input = step.Input(0), output = step.Output;
        if (input != output)
        {
            long bytes = Tensor.ElementCount(step.OutputShape) * (long)sizeof(float);
            Buffer.MemoryCopy(input, output, bytes, bytes);
        }
    }
}
