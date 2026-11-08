using PdfPecker.Ocr.Models;

namespace ModelBuilder;

/// <summary>A constant tensor of the builder's graph: float32 or int64 values.</summary>
internal sealed class Constant
{
    public Constant(long[] dims, float[] values)
    {
        Dims = dims;
        Floats = values;
    }

    public Constant(long[] dims, long[] values)
    {
        Dims = dims;
        Ints = values;
    }

    public long[] Dims { get; }
    public float[]? Floats { get; }
    public long[]? Ints { get; }
    public long Count => Dims.Aggregate(1L, (a, b) => a * b);

    public double this[long i] => Floats is not null ? Floats[i] : Ints![i];

    public static Constant From(OnnxTensor t) =>
        t.FloatData is not null ? new Constant(t.Dims, t.FloatData) : new Constant(t.Dims, t.IntData!);
}

/// <summary>
/// A node of the builder's graph. It starts as an ONNX node; fusion passes reuse it for the runtime's operators and
/// fill in the fused properties.
/// </summary>
internal sealed class Node
{
    public Node(string op, IEnumerable<string> inputs, IEnumerable<string> outputs, Dictionary<string, OnnxAttribute>? attributes = null)
    {
        Op = op;
        Inputs = inputs.ToList();
        Outputs = outputs.ToList();
        Attributes = attributes ?? new Dictionary<string, OnnxAttribute>(StringComparer.Ordinal);
    }

    public string Op { get; set; }
    public List<string> Inputs { get; }
    public List<string> Outputs { get; }
    public Dictionary<string, OnnxAttribute> Attributes { get; }
    public string Output => Outputs[0];

    // Fused properties.
    public ActivationKind Activation { get; set; }
    public float Alpha { get; set; }
    public float Beta { get; set; }
    public string? Residual { get; set; }
    public float[]? Bias { get; set; }
    public int Heads { get; set; }
    public float Scale { get; set; }
    public bool AddInput { get; set; }
    public ActivationKind Activation2 { get; set; }
    public float Alpha2 { get; set; }
    public float Beta2 { get; set; }
    public List<(int H, int W)> Factors { get; } = [];

    public long Int(string name, long fallback) => Attributes.TryGetValue(name, out var a) ? a.I : fallback;

    public float Float(string name, float fallback) => Attributes.TryGetValue(name, out var a) ? a.F : fallback;

    public long[] Ints(string name, long[] fallback) => Attributes.TryGetValue(name, out var a) ? a.Ints.ToArray() : fallback;

    public string Text(string name, string fallback) => Attributes.TryGetValue(name, out var a) ? a.Text : fallback;

    public override string ToString() => $"{Op}({string.Join(", ", Inputs)}) -> {string.Join(", ", Outputs)}";
}

/// <summary>The graph the builder simplifies: nodes in topological order, constants by name, one input and one output.</summary>
internal sealed class Graph
{
    public List<Node> Nodes { get; } = [];
    public Dictionary<string, Constant> Constants { get; } = new(StringComparer.Ordinal);
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public long[] InputDims { get; set; } = [];

    /// <summary>The shape of each tensor for each sample input shape, filled by <see cref="ShapeSampler"/>.</summary>
    public Dictionary<string, long[][]> Shapes { get; set; } = new(StringComparer.Ordinal);

    private int _names;

    public static Graph FromOnnx(OnnxModel model)
    {
        var g = new Graph();
        foreach (var t in model.Initializers) g.Constants[t.Name] = Constant.From(t);
        var input = model.Inputs.Single(i => !g.Constants.ContainsKey(i.Name));
        g.Input = input.Name;
        g.InputDims = input.Dims.ToArray();
        g.Output = model.Outputs.Single().Name;
        foreach (var n in model.Nodes)
        {
            if (n.OpType == "Constant")
            {
                var t = n.Attributes["value"].T ?? throw new InvalidDataException("Constant without a tensor value.");
                g.Constants[n.Outputs[0]] = Constant.From(t);
                continue;
            }
            g.Nodes.Add(new Node(n.OpType, n.Inputs, n.Outputs, n.Attributes));
        }
        return g;
    }

    public bool IsConstant(string name) => Constants.ContainsKey(name);

    public string NewName(string hint) => $"{hint}#{_names++}";

    public string AddConstant(string hint, Constant value)
    {
        string name = NewName(hint);
        Constants[name] = value;
        return name;
    }

    /// <summary>The nodes reading each tensor, recomputed by the caller after changes.</summary>
    public Dictionary<string, List<Node>> Consumers()
    {
        var result = new Dictionary<string, List<Node>>(StringComparer.Ordinal);
        foreach (var n in Nodes)
        {
            foreach (string i in n.Inputs.Concat(n.Residual is null ? [] : [n.Residual]))
            {
                if (i.Length == 0) continue;
                if (!result.TryGetValue(i, out var list)) result[i] = list = [];
                if (!list.Contains(n)) list.Add(n);
            }
        }
        return result;
    }

    public Dictionary<string, Node> Producers()
    {
        var result = new Dictionary<string, Node>(StringComparer.Ordinal);
        foreach (var n in Nodes)
            foreach (string o in n.Outputs) result[o] = n;
        return result;
    }

    /// <summary>Replaces every use of a tensor (including the graph output) by another.</summary>
    public void Rename(string from, string to)
    {
        foreach (var n in Nodes)
        {
            for (int i = 0; i < n.Inputs.Count; i++)
                if (n.Inputs[i] == from) n.Inputs[i] = to;
            if (n.Residual == from) n.Residual = to;
        }
        if (Output == from) Output = to;
    }

    /// <summary>Removes nodes whose outputs nothing reads.</summary>
    public void RemoveDeadNodes()
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            var consumers = Consumers();
            for (int i = Nodes.Count - 1; i >= 0; i--)
            {
                var n = Nodes[i];
                if (n.Outputs.All(o => o != Output && !consumers.ContainsKey(o)))
                {
                    Nodes.RemoveAt(i);
                    changed = true;
                }
            }
        }
        var used = new HashSet<string>(Nodes.SelectMany(n => n.Inputs.Concat(n.Residual is null ? [] : [n.Residual])), StringComparer.Ordinal);
        foreach (string name in Constants.Keys.Where(k => !used.Contains(k)).ToList()) Constants.Remove(name);
    }

    /// <summary>The sample shapes of a tensor; throws when the sampler did not see it.</summary>
    public long[][] ShapesOf(string name) =>
        Shapes.TryGetValue(name, out var s) ? s : throw new InvalidOperationException($"No shape for {name}.");

    public bool SameShape(string a, string b)
    {
        var sa = ShapesOf(a);
        var sb = ShapesOf(b);
        return sa.Zip(sb).All(p => p.First.AsSpan().SequenceEqual(p.Second));
    }

    public int Rank(string name) => ShapesOf(name)[0].Length;
}
