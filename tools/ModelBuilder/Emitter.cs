using PdfPecker.Ocr.Models;

namespace ModelBuilder;

/// <summary>Turns the simplified graph into the runtime's model definition.</summary>
internal static class Emitter
{
    /// <summary>Weights with at most this many values (biases, norms) are kept in full precision; they cost nothing.</summary>
    private const int SmallWeight = 1024;

    public static ModelDefinition Emit(Graph g, string name, string notice, WeightStorage precision, string[] characters)
    {
        var model = new ModelDefinition
        {
            Name = name,
            Notice = notice,
            InputShape = g.InputDims.Select(d => d > 0 ? (int)d : -1).ToArray(),
            Characters = characters,
        };
        var ids = new Dictionary<string, int>(StringComparer.Ordinal) { [g.Input] = 0 };
        int Value(string tensor) =>
            ids.TryGetValue(tensor, out int id) ? id : throw new NotSupportedException($"Tensor {tensor} is a constant or undefined where the runtime needs a computed value.");
        int AddWeight(float[] data, int[] shape)
        {
            model.Weights.Add(new WeightDefinition(shape, data, data.Length <= SmallWeight ? WeightStorage.Float32 : precision));
            return model.Weights.Count - 1;
        }
        int ConstantWeight(string tensor, int[]? shape = null)
        {
            var c = g.Constants[tensor];
            return AddWeight(c.Floats ?? throw new NotSupportedException($"{tensor} is not a float tensor."), shape ?? c.Dims.Select(d => (int)d).ToArray());
        }

        foreach (var n in g.Nodes)
        {
            var inputs = new List<int>();
            var ints = new List<int>();
            var floats = new List<float>();
            var weights = new List<int>();
            OpKind op;
            switch (n.Op)
            {
                case "Conv":
                {
                    op = OpKind.Conv;
                    var w = g.Constants[n.Inputs[1]];
                    long[] strides = n.Ints("strides", [1, 1]);
                    long[] dil = n.Ints("dilations", [1, 1]);
                    long[] pads = SampleInvariant(g, n, s => Evaluator.ResolvePads(n, s, [w.Dims[2], w.Dims[3]], strides, dil));
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange([(int)w.Dims[2], (int)w.Dims[3], (int)strides[0], (int)strides[1], (int)dil[0], (int)dil[1],
                        (int)pads[0], (int)pads[1], (int)pads[2], (int)pads[3], (int)n.Int("group", 1), (int)n.Activation, n.Residual is null ? 0 : 1]);
                    floats.AddRange([n.Alpha, n.Beta]);
                    weights.Add(ConstantWeight(n.Inputs[1]));
                    weights.Add(AddWeight(n.Bias ?? new float[w.Dims[0]], [(int)w.Dims[0]]));
                    if (n.Residual is not null) inputs.Add(Value(n.Residual));
                    break;
                }
                case "ConvTranspose":
                {
                    op = OpKind.ConvTranspose;
                    var w = g.Constants[n.Inputs[1]];
                    if (!n.Ints("strides", [1, 1]).SequenceEqual(new[] { w.Dims[2], w.Dims[3] }) || n.Ints("pads", [0, 0, 0, 0]).Any(p => p != 0)
                        || n.Ints("dilations", [1, 1]).Any(d => d != 1) || n.Int("group", 1) != 1 || n.Ints("output_padding", [0, 0]).Any(p => p != 0))
                        throw new NotSupportedException($"{n}: only transposed convolutions whose kernel equals the stride are supported.");
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange([(int)w.Dims[2], (int)w.Dims[3], (int)n.Activation]);
                    floats.AddRange([n.Alpha, n.Beta]);
                    weights.Add(ConstantWeight(n.Inputs[1]));
                    weights.Add(AddWeight(n.Bias ?? new float[w.Dims[1]], [(int)w.Dims[1]]));
                    break;
                }
                case "MaxPool":
                case "AveragePool":
                {
                    op = n.Op == "MaxPool" ? OpKind.MaxPool : OpKind.AveragePool;
                    long[] kernel = n.Ints("kernel_shape", []);
                    long[] strides = n.Ints("strides", [1, 1]);
                    long[] pads = SampleInvariant(g, n, s => Evaluator.ResolvePads(n, s, kernel, strides, [1, 1]));
                    if (n.Int("ceil_mode", 0) != 0) throw new NotSupportedException($"{n}: ceil_mode is not supported.");
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange([(int)kernel[0], (int)kernel[1], (int)strides[0], (int)strides[1], (int)pads[0], (int)pads[1], (int)pads[2], (int)pads[3]]);
                    if (op == OpKind.AveragePool) ints.Add((int)n.Int("count_include_pad", 0));
                    break;
                }
                case "GlobalAveragePool":
                case "ReduceMean" when g.Rank(n.Inputs[0]) == 4 && n.Ints("axes", []).Order().SequenceEqual(new long[] { 2, 3 }) && n.Int("keepdims", 1) != 0:
                    op = OpKind.GlobalAveragePool;
                    inputs.Add(Value(n.Inputs[0]));
                    break;
                case "SqueezeExcite":
                {
                    op = OpKind.SqueezeExcite;
                    var w1 = g.Constants[n.Inputs[1]];
                    var w2 = g.Constants[n.Inputs[2]];
                    int hidden = (int)w1.Dims[0], channels = (int)w2.Dims[0];
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange([(int)n.Activation, (int)n.Activation2, n.AddInput ? 1 : 0]);
                    floats.AddRange([n.Alpha, n.Beta, n.Alpha2, n.Beta2]);
                    weights.Add(ConstantWeight(n.Inputs[1], [hidden, channels]));
                    weights.Add(AddWeight(n.Bias![..hidden], [hidden]));
                    weights.Add(ConstantWeight(n.Inputs[2], [channels, hidden]));
                    weights.Add(AddWeight(n.Bias![hidden..], [channels]));
                    break;
                }
                case "Add":
                case "Mul":
                    op = n.Op == "Add" ? OpKind.Add : OpKind.Mul;
                    inputs.Add(Value(n.Inputs[0]));
                    inputs.Add(Value(n.Inputs[1]));
                    var (fh, fw) = n.Factors.Count > 0 ? n.Factors[0] : (1, 1);
                    ints.AddRange([fh, fw]);
                    break;
                case "Act":
                    op = OpKind.Activation;
                    inputs.Add(Value(n.Inputs[0]));
                    ints.Add((int)n.Activation);
                    floats.AddRange([n.Alpha, n.Beta]);
                    break;
                case "Concat":
                {
                    op = OpKind.Concat;
                    int rank = g.Rank(n.Output);
                    long axis = n.Int("axis", 0);
                    ints.Add((int)(axis < 0 ? axis + rank : axis));
                    for (int i = 0; i < n.Inputs.Count; i++)
                    {
                        inputs.Add(Value(n.Inputs[i]));
                        var f = i < n.Factors.Count ? n.Factors[i] : (1, 1);
                        ints.AddRange([f.Item1, f.Item2]);
                    }
                    break;
                }
                case "LayerNorm":
                    op = OpKind.LayerNorm;
                    inputs.Add(Value(n.Inputs[0]));
                    floats.Add(n.Alpha);
                    weights.Add(ConstantWeight(n.Inputs[1], [(int)g.Constants[n.Inputs[1]].Count]));
                    weights.Add(ConstantWeight(n.Inputs[2], [(int)g.Constants[n.Inputs[2]].Count]));
                    break;
                case "Linear":
                {
                    op = OpKind.Linear;
                    var w = g.Constants[n.Inputs[1]];
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange([(int)n.Activation, n.Residual is null ? 0 : 1]);
                    floats.AddRange([n.Alpha, n.Beta]);
                    weights.Add(ConstantWeight(n.Inputs[1]));
                    weights.Add(AddWeight(n.Bias ?? new float[w.Dims[1]], [(int)w.Dims[1]]));
                    if (n.Residual is not null) inputs.Add(Value(n.Residual));
                    break;
                }
                case "Attention":
                    op = OpKind.Attention;
                    inputs.Add(Value(n.Inputs[0]));
                    ints.Add(n.Heads);
                    floats.Add(n.Scale);
                    break;
                case "Softmax":
                {
                    op = OpKind.Softmax;
                    int rank = g.Rank(n.Output);
                    long axis = n.Int("axis", n.Int("axis", 1));
                    if ((axis < 0 ? axis + rank : axis) != rank - 1) throw new NotSupportedException($"{n}: only softmax over the last axis is supported.");
                    inputs.Add(Value(n.Inputs[0]));
                    break;
                }
                case "Transpose":
                    op = OpKind.Transpose;
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange(n.Ints("perm", Enumerable.Range(0, g.Rank(n.Inputs[0])).Reverse().Select(i => (long)i).ToArray()).Select(p => (int)p));
                    break;
                case "Reshape":
                case "Squeeze":
                case "Unsqueeze":
                case "Flatten":
                    op = OpKind.Reshape;
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange(ReshapeSpec(g, n.Inputs[0], n.Output));
                    break;
                case "Upsample":
                    op = OpKind.Upsample;
                    inputs.Add(Value(n.Inputs[0]));
                    ints.AddRange([n.Factors[0].H, n.Factors[0].W]);
                    break;
                default:
                    throw new NotSupportedException($"The runtime has no operator for {n}.");
            }
            int output = model.Nodes.Count + 1;
            model.Nodes.Add(new NodeDefinition(op, inputs.ToArray(), output, ints.ToArray(), floats.ToArray(), weights.ToArray()));
            ids[n.Output] = output;
        }
        if (!ids.TryGetValue(g.Output, out int last) || last != model.Nodes.Count)
            throw new InvalidOperationException("The graph output is not the last node's output.");
        return model;
    }

    /// <summary>A per-sample computation that must give the same result for every sample input shape.</summary>
    private static long[] SampleInvariant(Graph g, Node n, Func<long[], long[]> compute)
    {
        var results = g.ShapesOf(n.Inputs[0]).Select(compute).ToList();
        if (results.Any(r => !r.SequenceEqual(results[0]))) throw new NotSupportedException($"{n}: the padding depends on the input size.");
        return results[0];
    }

    /// <summary>
    /// The reshape target for the runtime, from the sample shapes: a dimension that never changes is a constant, one
    /// that always equals the input's at the same position is 0, and at most one other is -1. The result is checked
    /// against every sample.
    /// </summary>
    private static int[] ReshapeSpec(Graph g, string input, string output)
    {
        var ins = g.ShapesOf(input);
        var outs = g.ShapesOf(output);
        int rank = outs[0].Length;
        var spec = new int[rank];
        int inferred = 0;
        for (int i = 0; i < rank; i++)
        {
            if (outs.All(o => o[i] == outs[0][i])) spec[i] = (int)outs[0][i];
            else if (Enumerable.Range(0, outs.Length).All(s => i < ins[s].Length && outs[s][i] == ins[s][i])) spec[i] = 0;
            else
            {
                spec[i] = -1;
                inferred++;
            }
        }
        if (inferred > 1) throw new NotSupportedException($"Reshape to {output} has more than one dimension that varies independently.");
        for (int s = 0; s < ins.Length; s++)
            if (!Evaluator.ReshapeTarget(ins[s], spec.Select(d => (long)d).ToArray()).SequenceEqual(outs[s]))
                throw new NotSupportedException($"Reshape to {output} cannot be written with 0 and -1.");
        return spec;
    }
}
