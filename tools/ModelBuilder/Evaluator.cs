namespace ModelBuilder;

/// <summary>A tensor during evaluation: always a shape, and values when they can be computed at build time.</summary>
internal sealed class Value
{
    public Value(long[] shape, float[]? floats = null, long[]? ints = null)
    {
        Shape = shape;
        Floats = floats;
        Ints = ints;
    }

    public long[] Shape { get; }
    public float[]? Floats { get; }
    public long[]? Ints { get; }
    public bool HasData => Floats is not null || Ints is not null;
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);
    public double At(long i) => Floats is not null ? Floats[i] : Ints![i];

    public static Value Of(Constant c) => new(c.Dims, c.Floats, c.Ints);

    public Constant ToConstant() => Floats is not null ? new Constant(Shape, Floats) : new Constant(Shape, Ints!);
}

/// <summary>
/// Shape inference for the ONNX operators of the bundled models, and value computation for the cheap ones (shape
/// arithmetic, reshapes and elementwise arithmetic on constants). It is what lets the builder fold constant subgraphs
/// and the Shape/Slice/Concat chains that compute reshape targets.
/// </summary>
internal static class Evaluator
{
    public static Value[] Evaluate(Node n, Value?[] inputs)
    {
        Value In(int i) => inputs[i] ?? throw new InvalidOperationException($"{n}: input {i} is unknown.");
        switch (n.Op)
        {
            case "Identity":
                return [In(0)];
            case "Cast":
            {
                var x = In(0);
                long to = n.Int("to", 1);
                if (!x.HasData) return [new Value(x.Shape)];
                if (to == 1) return [new Value(x.Shape, Enumerable.Range(0, (int)x.Count).Select(i => (float)x.At(i)).ToArray())];
                return [new Value(x.Shape, ints: Enumerable.Range(0, (int)x.Count).Select(i => (long)x.At(i)).ToArray())];
            }
            case "Shape":
            {
                var x = In(0);
                return [new Value([x.Shape.Length], ints: (long[])x.Shape.Clone())];
            }
            case "Add":
            case "Sub":
            case "Mul":
            case "Div":
            case "Pow":
                return [Binary(n.Op, In(0), In(1))];
            case "Relu":
            case "Sigmoid":
            case "HardSigmoid":
            case "Erf":
            case "Sqrt":
            case "Clip":
            case "Softmax":
            case "BatchNormalization":
                return [new Value(In(0).Shape)];
            case "Reshape":
            {
                var x = In(0);
                var target = In(1);
                if (target.Ints is null) throw new InvalidOperationException($"{n}: the target shape is unknown.");
                long[] shape = ReshapeTarget(x.Shape, target.Ints);
                return [new Value(shape, x.Floats, x.Ints)];
            }
            case "Squeeze":
            {
                var x = In(0);
                long[] axes = n.Ints("axes", []).Select(a => a < 0 ? a + x.Shape.Length : a).ToArray();
                var shape = x.Shape.Where((d, i) => axes.Length == 0 ? d != 1 : !axes.Contains(i)).ToArray();
                return [new Value(shape, x.Floats, x.Ints)];
            }
            case "Unsqueeze":
            {
                var x = In(0);
                int rank = x.Shape.Length + n.Ints("axes", []).Length;
                var axes = n.Ints("axes", []).Select(a => a < 0 ? a + rank : a).ToHashSet();
                var shape = new long[rank];
                int j = 0;
                for (int i = 0; i < rank; i++) shape[i] = axes.Contains(i) ? 1 : x.Shape[j++];
                return [new Value(shape, x.Floats, x.Ints)];
            }
            case "Concat":
                return [Concat(n, inputs.Select(v => v!).ToArray())];
            case "Slice":
                return [Slice(In(0), In(1), In(2), inputs.Length > 3 ? inputs[3] : null, inputs.Length > 4 ? inputs[4] : null)];
            case "Transpose":
            {
                var x = In(0);
                long[] perm = n.Ints("perm", Enumerable.Range(0, x.Shape.Length).Reverse().Select(i => (long)i).ToArray());
                return [new Value(perm.Select(p => x.Shape[p]).ToArray())];
            }
            case "ReduceMean":
            {
                var x = In(0);
                long[] axes = n.Ints("axes", Enumerable.Range(0, x.Shape.Length).Select(i => (long)i).ToArray()).Select(a => a < 0 ? a + x.Shape.Length : a).ToArray();
                bool keep = n.Int("keepdims", 1) != 0;
                var shape = new List<long>();
                for (int i = 0; i < x.Shape.Length; i++)
                {
                    if (!axes.Contains(i)) shape.Add(x.Shape[i]);
                    else if (keep) shape.Add(1);
                }
                return [new Value(shape.ToArray())];
            }
            case "GlobalAveragePool":
            {
                var x = In(0);
                return [new Value([x.Shape[0], x.Shape[1], 1, 1])];
            }
            case "MatMul":
            {
                long[] a = In(0).Shape, b = In(1).Shape;
                if (a.Length < 2 || b.Length < 2 || a[^1] != b[^2]) throw new InvalidOperationException($"{n}: matrix shapes do not match.");
                var batchA = a[..^2];
                var batchB = b[..^2];
                var batch = BroadcastShape(batchA, batchB);
                return [new Value([.. batch, a[^2], b[^1]])];
            }
            case "Conv":
                return [new Value(ConvShape(n, In(0).Shape, In(1).Shape))];
            case "ConvTranspose":
            {
                long[] x = In(0).Shape, w = In(1).Shape;
                long[] strides = n.Ints("strides", [1, 1]);
                long[] pads = n.Ints("pads", [0, 0, 0, 0]);
                long[] dil = n.Ints("dilations", [1, 1]);
                long[] outPad = n.Ints("output_padding", [0, 0]);
                long group = n.Int("group", 1);
                var shape = new long[] { x[0], w[1] * group, 0, 0 };
                for (int i = 0; i < 2; i++)
                    shape[2 + i] = strides[i] * (x[2 + i] - 1) + outPad[i] + (w[2 + i] - 1) * dil[i] + 1 - pads[i] - pads[i + 2];
                return [new Value(shape)];
            }
            case "MaxPool":
            case "AveragePool":
                return [new Value(PoolShape(n, In(0).Shape))];
            case "Resize":
            {
                var x = In(0);
                if (inputs.Length > 3 && inputs[3] is { Ints.Length: > 0 } sizes) return [new Value(sizes.Ints!)];
                var scales = In(2);
                if (scales.Floats is null) throw new InvalidOperationException($"{n}: the scales are unknown.");
                return [new Value(x.Shape.Select((d, i) => (long)Math.Floor(d * (double)scales.Floats[i])).ToArray())];
            }
            default:
                throw new NotSupportedException($"The model builder does not know the operator {n.Op}.");
        }
    }

    public static long[] ReshapeTarget(long[] input, long[] target)
    {
        var shape = new long[target.Length];
        long total = input.Aggregate(1L, (a, b) => a * b), known = 1;
        int infer = -1;
        for (int i = 0; i < target.Length; i++)
        {
            long d = target[i] == 0 ? input[i] : target[i];
            if (d == -1)
            {
                infer = i;
                continue;
            }
            shape[i] = d;
            known *= d;
        }
        if (infer >= 0) shape[infer] = total / known;
        return shape;
    }

    public static long[] BroadcastShape(long[] a, long[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var shape = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            long da = i - (rank - a.Length) >= 0 ? a[i - (rank - a.Length)] : 1;
            long db = i - (rank - b.Length) >= 0 ? b[i - (rank - b.Length)] : 1;
            if (da != db && da != 1 && db != 1) throw new InvalidOperationException("Shapes do not broadcast.");
            shape[i] = Math.Max(da, db);
        }
        return shape;
    }

    private static Value Binary(string op, Value a, Value b)
    {
        var shape = BroadcastShape(a.Shape, b.Shape);
        if (!a.HasData || !b.HasData) return new Value(shape);
        long count = shape.Aggregate(1L, (x, y) => x * y);
        bool ints = a.Ints is not null && b.Ints is not null;
        var fa = Indexer(a.Shape, shape);
        var fb = Indexer(b.Shape, shape);
        if (ints)
        {
            var result = new long[count];
            for (long i = 0; i < count; i++)
            {
                long x = a.Ints![fa(i)], y = b.Ints![fb(i)];
                result[i] = op switch { "Add" => x + y, "Sub" => x - y, "Mul" => x * y, "Div" => x / y, _ => (long)Math.Pow(x, y) };
            }
            return new Value(shape, ints: result);
        }
        var floats = new float[count];
        for (long i = 0; i < count; i++)
        {
            float x = (float)a.At(fa(i)), y = (float)b.At(fb(i));
            floats[i] = op switch { "Add" => x + y, "Sub" => x - y, "Mul" => x * y, "Div" => x / y, _ => MathF.Pow(x, y) };
        }
        return new Value(shape, floats);
    }

    /// <summary>Maps an index of the broadcast output to an index of an input.</summary>
    private static Func<long, long> Indexer(long[] input, long[] output)
    {
        int rank = output.Length;
        var strides = new long[rank];
        long stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            int j = i - (rank - input.Length);
            long d = j >= 0 ? input[j] : 1;
            strides[i] = d == 1 ? 0 : stride;
            stride *= d;
        }
        return index =>
        {
            long result = 0;
            for (int i = rank - 1; i >= 0; i--)
            {
                result += index % output[i] * strides[i];
                index /= output[i];
            }
            return result;
        };
    }

    private static Value Concat(Node n, Value[] inputs)
    {
        int rank = inputs[0].Shape.Length;
        long axis = n.Int("axis", 0);
        if (axis < 0) axis += rank;
        var shape = (long[])inputs[0].Shape.Clone();
        shape[axis] = inputs.Sum(v => v.Shape[axis]);
        if (!inputs.All(v => v.HasData)) return new Value(shape);
        long outer = shape.Take((int)axis).Aggregate(1L, (a, b) => a * b);
        bool ints = inputs.All(v => v.Ints is not null);
        var data = new List<double>();
        for (long o = 0; o < outer; o++)
        {
            foreach (var v in inputs)
            {
                long chunk = v.Count / outer;
                for (long i = 0; i < chunk; i++) data.Add(v.At(o * chunk + i));
            }
        }
        return ints ? new Value(shape, ints: data.Select(d => (long)d).ToArray()) : new Value(shape, data.Select(d => (float)d).ToArray());
    }

    private static Value Slice(Value x, Value starts, Value ends, Value? axes, Value? steps)
    {
        if (starts.Ints is null || ends.Ints is null) throw new InvalidOperationException("Slice bounds are unknown.");
        int rank = x.Shape.Length;
        var begin = new long[rank];
        var end = (long[])x.Shape.Clone();
        var step = Enumerable.Repeat(1L, rank).ToArray();
        for (int i = 0; i < starts.Ints.Length; i++)
        {
            int axis = (int)(axes?.Ints?[i] ?? i);
            if (axis < 0) axis += rank;
            long dim = x.Shape[axis];
            long s = steps?.Ints?[i] ?? 1;
            if (s != 1) throw new NotSupportedException("Slices with steps are not supported.");
            long b = starts.Ints[i], e = ends.Ints[i];
            if (b < 0) b += dim;
            if (e < 0) e += dim;
            begin[axis] = Math.Clamp(b, 0, dim);
            end[axis] = Math.Clamp(e, 0, dim);
        }
        var shape = new long[rank];
        for (int i = 0; i < rank; i++) shape[i] = Math.Max(0, end[i] - begin[i]);
        if (!x.HasData) return new Value(shape);
        long count = shape.Aggregate(1L, (a, b) => a * b);
        var values = new double[count];
        var srcStrides = new long[rank];
        long stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            srcStrides[i] = stride;
            stride *= x.Shape[i];
        }
        for (long o = 0; o < count; o++)
        {
            long rest = o, src = 0;
            for (int i = rank - 1; i >= 0; i--)
            {
                long idx = rest % shape[i];
                rest /= shape[i];
                src += (idx + begin[i]) * srcStrides[i];
            }
            values[o] = x.At(src);
        }
        return x.Ints is not null ? new Value(shape, ints: values.Select(v => (long)v).ToArray()) : new Value(shape, values.Select(v => (float)v).ToArray());
    }

    public static long[] ConvShape(Node n, long[] x, long[] w)
    {
        long[] kernel = [w[2], w[3]];
        long[] strides = n.Ints("strides", [1, 1]);
        long[] dil = n.Ints("dilations", [1, 1]);
        var pads = ResolvePads(n, x, kernel, strides, dil);
        var shape = new long[] { x[0], w[0], 0, 0 };
        for (int i = 0; i < 2; i++)
            shape[2 + i] = (x[2 + i] + pads[i] + pads[i + 2] - dil[i] * (kernel[i] - 1) - 1) / strides[i] + 1;
        return shape;
    }

    public static long[] PoolShape(Node n, long[] x)
    {
        long[] kernel = n.Ints("kernel_shape", []);
        long[] strides = n.Ints("strides", [1, 1]);
        var pads = ResolvePads(n, x, kernel, strides, [1, 1]);
        bool ceil = n.Int("ceil_mode", 0) != 0;
        var shape = new long[] { x[0], x[1], 0, 0 };
        for (int i = 0; i < 2; i++)
        {
            double size = (double)(x[2 + i] + pads[i] + pads[i + 2] - kernel[i]) / strides[i];
            shape[2 + i] = (long)(ceil ? Math.Ceiling(size) : Math.Floor(size)) + 1;
        }
        return shape;
    }

    /// <summary>
    /// The pads (top, left, bottom, right) of a convolution or pooling. SAME_UPPER and SAME_LOWER depend on the input
    /// size only through the stride, so with stride 1 they are constants the runtime can take as explicit pads.
    /// </summary>
    public static long[] ResolvePads(Node n, long[] x, long[] kernel, long[] strides, long[] dil)
    {
        string autoPad = n.Text("auto_pad", "NOTSET");
        if (autoPad is "NOTSET" or "") return n.Ints("pads", [0, 0, 0, 0]);
        if (autoPad == "VALID") return [0, 0, 0, 0];
        var pads = new long[4];
        for (int i = 0; i < 2; i++)
        {
            long input = x[2 + i];
            long output = (input + strides[i] - 1) / strides[i];
            long total = Math.Max(0, (output - 1) * strides[i] + (kernel[i] - 1) * dil[i] + 1 - input);
            long small = total / 2;
            pads[i] = autoPad == "SAME_UPPER" ? small : total - small;
            pads[i + 2] = total - pads[i];
        }
        return pads;
    }
}
