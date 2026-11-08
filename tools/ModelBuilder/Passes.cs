using PdfPecker.Ocr.Models;

namespace ModelBuilder;

/// <summary>
/// The graph simplification passes, applied in order by <see cref="Run"/>. Each pass keeps the graph topologically
/// ordered: a fused node takes the place of the last node of the pattern it replaces and keeps its output name, so the
/// sample shapes stay valid.
/// </summary>
internal static class Passes
{
    public static void Run(Graph g, long[][] samples, TextWriter log)
    {
        RemoveIdentities(g);
        FoldConstants(g);
        SampleShapes(g, samples);
        g.RemoveDeadNodes();
        log.WriteLine($"  after folding: {Summary(g)}");
        FoldBatchNorm(g);
        FoldBiasAdds(g);
        MatMulToLinear(g);
        RecogniseActivations(g);
        FuseActivations(g);
        FuseLayerNorm(g);
        FuseSqueezeExcite(g);
        FuseAttention(g);
        FuseUpsampling(g);
        FuseResiduals(g);
        g.RemoveDeadNodes();
        log.WriteLine($"  after fusion:  {Summary(g)}");
    }

    public static string Summary(Graph g) =>
        $"{g.Nodes.Count} nodes (" + string.Join(", ", g.Nodes.GroupBy(n => n.Op).OrderByDescending(x => x.Count()).ThenBy(x => x.Key, StringComparer.Ordinal).Select(x => $"{x.Key} {x.Count()}")) + ")";

    private static void RemoveIdentities(Graph g)
    {
        foreach (var n in g.Nodes.Where(n => n.Op == "Identity").ToList())
        {
            g.Nodes.Remove(n);
            if (n.Output == g.Output)
            {
                // Keep the graph's output name: rename the producer's output instead.
                var producer = g.Producers()[n.Inputs[0]];
                string old = n.Inputs[0];
                producer.Outputs[producer.Outputs.IndexOf(old)] = n.Output;
                g.Rename(old, n.Output);
            }
            else
            {
                g.Rename(n.Output, n.Inputs[0]);
            }
        }
    }

    /// <summary>Evaluates nodes whose inputs are all constants.</summary>
    private static void FoldConstants(Graph g)
    {
        foreach (var n in g.Nodes.ToList())
        {
            var inputs = n.Inputs.Where(i => i.Length != 0).ToList();
            if (inputs.Count == 0 || !inputs.All(g.IsConstant) || n.Outputs.Count != 1) continue;
            if (n.Op is not ("Reshape" or "Cast" or "Concat" or "Slice" or "Squeeze" or "Unsqueeze" or "Add" or "Sub" or "Mul" or "Div" or "Shape")) continue;
            var values = n.Inputs.Select(i => i.Length == 0 ? null : Value.Of(g.Constants[i])).ToArray();
            var result = Evaluator.Evaluate(n, values)[0];
            if (!result.HasData) continue;
            g.Constants[n.Output] = result.ToConstant();
            g.Nodes.Remove(n);
        }
    }

    /// <summary>
    /// Runs shape inference for several input shapes, which differ in every dimension the caller chooses. A tensor whose
    /// values come out the same for all of them (the shape arithmetic of reshape targets) becomes a constant; the
    /// shapes are kept for the later passes and for writing reshape targets with 0 and -1.
    /// </summary>
    private static void SampleShapes(Graph g, long[][] samples)
    {
        var perSample = new List<Dictionary<string, Value>>();
        foreach (var sample in samples)
        {
            var values = new Dictionary<string, Value>(StringComparer.Ordinal) { [g.Input] = new Value(sample) };
            foreach (var (name, c) in g.Constants) values[name] = Value.Of(c);
            foreach (var n in g.Nodes)
            {
                var inputs = n.Inputs.Select(i => i.Length == 0 ? null : values[i]).ToArray();
                var outputs = Evaluator.Evaluate(n, inputs);
                for (int i = 0; i < n.Outputs.Count; i++) values[n.Outputs[i]] = outputs[i];
            }
            perSample.Add(values);
        }
        g.Shapes = perSample[0].Keys.ToDictionary(k => k, k => perSample.Select(s => s[k].Shape).ToArray(), StringComparer.Ordinal);
        // Reshape targets are rewritten from the sample shapes when the graph is emitted, so the shape arithmetic that
        // computes them at run time in ONNX is no longer needed.
        foreach (var n in g.Nodes.Where(n => n.Op == "Reshape")) n.Inputs.RemoveRange(1, n.Inputs.Count - 1);
        foreach (var n in g.Nodes.ToList())
        {
            if (n.Outputs.Count != 1) continue;
            var outputs = perSample.Select(s => s[n.Output]).ToList();
            if (!outputs.All(v => v.HasData)) continue;
            var first = outputs[0];
            bool same = outputs.All(v => v.Shape.SequenceEqual(first.Shape) && Enumerable.Range(0, (int)v.Count).All(i => v.At(i).Equals(first.At(i))));
            if (!same) continue;
            g.Constants[n.Output] = first.ToConstant();
            g.Nodes.Remove(n);
        }
    }

    private static Node? SingleConsumer(Graph g, Dictionary<string, List<Node>> consumers, string tensor) =>
        tensor != g.Output && consumers.TryGetValue(tensor, out var list) && list.Count == 1 && list[0].Inputs.Count(i => i == tensor) == 1 ? list[0] : null;

    private static int ConsumerCount(Graph g, Dictionary<string, List<Node>> consumers, string tensor) =>
        (tensor == g.Output ? 1 : 0) + (consumers.TryGetValue(tensor, out var list) ? list.Count : 0);

    private static bool IsScalar(Graph g, string name, out float value)
    {
        value = 0;
        if (!g.Constants.TryGetValue(name, out var c) || c.Count != 1) return false;
        value = (float)c[0];
        return true;
    }

    private static bool IsScalar(Graph g, string name, float expected, float tolerance = 1e-6f) =>
        IsScalar(g, name, out float v) && Math.Abs(v - expected) <= tolerance * Math.Max(1, Math.Abs(expected));

    /// <summary>The other input of a two-input node.</summary>
    private static string Other(Node n, string input) => n.Inputs[0] == input ? n.Inputs[1] : n.Inputs[0];

    /// <summary>Replaces a node's output name by the output of the last node of a fused pattern.</summary>
    private static void TakeOutput(Graph g, Node fused, Node last)
    {
        fused.Outputs[0] = last.Output;
        g.Nodes.Remove(fused);
        g.Nodes.Insert(g.Nodes.IndexOf(last), fused);
        g.Nodes.Remove(last);
    }

    /// <summary>Conv followed by BatchNormalization: scale the weights and shift the bias per output channel.</summary>
    private static void FoldBatchNorm(Graph g)
    {
        var consumers = g.Consumers();
        foreach (var conv in g.Nodes.Where(n => n.Op == "Conv").ToList())
        {
            var bn = SingleConsumer(g, consumers, conv.Output);
            if (bn is null || bn.Op != "BatchNormalization" || bn.Inputs[0] != conv.Output) continue;
            var w = g.Constants[conv.Inputs[1]];
            int cout = (int)w.Dims[0];
            long perChannel = w.Count / cout;
            var scale = g.Constants[bn.Inputs[1]];
            var shift = g.Constants[bn.Inputs[2]];
            var mean = g.Constants[bn.Inputs[3]];
            var variance = g.Constants[bn.Inputs[4]];
            double epsilon = bn.Float("epsilon", 1e-5f);
            var weights = new float[w.Count];
            var bias = new float[cout];
            for (int c = 0; c < cout; c++)
            {
                double s = scale[c] / Math.Sqrt(variance[c] + epsilon);
                for (long i = 0; i < perChannel; i++) weights[c * perChannel + i] = (float)(w[c * perChannel + i] * s);
                double b = conv.Bias is not null ? conv.Bias[c] : conv.Inputs.Count > 2 ? g.Constants[conv.Inputs[2]][c] : 0;
                bias[c] = (float)((b - mean[c]) * s + shift[c]);
            }
            conv.Inputs[1] = g.AddConstant(conv.Inputs[1], new Constant(w.Dims, weights));
            if (conv.Inputs.Count > 2) conv.Inputs.RemoveAt(2);
            conv.Bias = bias;
            TakeOutput(g, conv, bn);
        }
    }

    /// <summary>Conv, ConvTranspose or Linear followed by an Add of a per-channel constant: a bias.</summary>
    private static void FoldBiasAdds(Graph g)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            var consumers = g.Consumers();
            foreach (var n in g.Nodes.Where(n => n.Op is "Conv" or "ConvTranspose" or "Linear").ToList())
            {
                if (n.Inputs.Count > 2)
                {
                    n.Bias = g.Constants[n.Inputs[2]].Floats!.ToArray();
                    n.Inputs.RemoveAt(2);
                }
                var add = SingleConsumer(g, consumers, n.Output);
                if (add is null || add.Op != "Add") continue;
                string other = Other(add, n.Output);
                if (!g.Constants.TryGetValue(other, out var c)) continue;
                int channels = n.Op == "Linear" ? (int)g.Constants[n.Inputs[1]].Dims[1] : n.Op == "Conv" ? (int)g.Constants[n.Inputs[1]].Dims[0] : (int)g.Constants[n.Inputs[1]].Dims[1] * (int)n.Int("group", 1);
                // The constant must broadcast as one value per channel: [C] (or [1, .., C]) after a Linear, and
                // [C, 1, 1] (or [1, C, 1, 1]) after a convolution.
                var d = c.Dims;
                bool perChannel = c.Count == channels && (n.Op == "Linear"
                    ? d.Length >= 1 && d[^1] == channels
                    : d.Length >= 3 && d[^1] == 1 && d[^2] == 1 && d[^3] == channels);
                if (!perChannel) continue;
                n.Bias ??= new float[channels];
                for (int i = 0; i < channels; i++) n.Bias[i] += (float)c[i];
                TakeOutput(g, n, add);
                changed = true;
                break;
            }
        }
    }

    /// <summary>MatMul by a constant matrix becomes Linear, which the bias and activation passes then extend.</summary>
    private static void MatMulToLinear(Graph g)
    {
        foreach (var n in g.Nodes.Where(n => n.Op == "MatMul" && g.Constants.TryGetValue(n.Inputs[1], out var w) && w.Dims.Length == 2))
            n.Op = "Linear";
        FoldBiasAdds(g);
    }

    /// <summary>
    /// Turns the activation subgraphs into single Act nodes: Relu, Sigmoid, HardSigmoid, x * sigmoid(x) (Swish),
    /// x * hardsigmoid(x) and x * clip(x + 3, 0, 6) / 6 (HardSwish), and x * (erf(x / sqrt 2) + 1) * 0.5 (GELU).
    /// </summary>
    private static void RecogniseActivations(Graph g)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            var consumers = g.Consumers();
            foreach (var n in g.Nodes.ToList())
            {
                string x = n.Inputs.Count > 0 ? n.Inputs[0] : "";
                if (n.Op is "Sigmoid" or "HardSigmoid")
                {
                    bool hard = n.Op == "HardSigmoid";
                    float alpha = n.Float("alpha", 0.2f), beta = n.Float("beta", 0.5f);
                    var mul = SingleConsumer(g, consumers, n.Output);
                    if (mul is { Op: "Mul" } && Other(mul, n.Output) == x && ConsumerCount(g, consumers, x) == 2)
                    {
                        var act = Act(hard ? ActivationKind.HardSwish : ActivationKind.Swish, x, alpha, beta);
                        Replace(g, mul, act, n);
                        changed = true;
                        break;
                    }
                    var single = Act(hard ? ActivationKind.HardSigmoid : ActivationKind.Sigmoid, x, alpha, beta);
                    Replace(g, n, single);
                    changed = true;
                    break;
                }
                if (n.Op == "Relu")
                {
                    Replace(g, n, Act(ActivationKind.Relu, x, 0, 0));
                    changed = true;
                    break;
                }
                if (n.Op == "Add" && IsScalar(g, n.Inputs[1], 3f))
                {
                    // x * clip(x + 3, 0, 6) / 6
                    var clip = SingleConsumer(g, consumers, n.Output);
                    if (clip is not { Op: "Clip" } || !IsScalar(g, clip.Inputs[1], 0f) || !IsScalar(g, clip.Inputs[2], 6f)) continue;
                    var mul = SingleConsumer(g, consumers, clip.Output);
                    if (mul is not { Op: "Mul" } || Other(mul, clip.Output) != x || ConsumerCount(g, consumers, x) != 2) continue;
                    var div = SingleConsumer(g, consumers, mul.Output);
                    if (div is not { Op: "Div" } || div.Inputs[0] != mul.Output || !IsScalar(g, div.Inputs[1], 6f)) continue;
                    Replace(g, div, Act(ActivationKind.HardSwish, x, 1f / 6, 0.5f), n, clip, mul);
                    changed = true;
                    break;
                }
                if (n.Op == "Div" && IsScalar(g, n.Inputs[1], MathF.Sqrt(2f), 1e-5f))
                {
                    // x * (erf(x / sqrt 2) + 1) * 0.5
                    var erf = SingleConsumer(g, consumers, n.Output);
                    if (erf is not { Op: "Erf" }) continue;
                    var add = SingleConsumer(g, consumers, erf.Output);
                    if (add is not { Op: "Add" } || !IsScalar(g, Other(add, erf.Output), 1f)) continue;
                    var mul = SingleConsumer(g, consumers, add.Output);
                    if (mul is not { Op: "Mul" } || Other(mul, add.Output) != x || ConsumerCount(g, consumers, x) != 2) continue;
                    var half = SingleConsumer(g, consumers, mul.Output);
                    if (half is not { Op: "Mul" } || !IsScalar(g, Other(half, mul.Output), 0.5f)) continue;
                    Replace(g, half, Act(ActivationKind.Gelu, x, 0, 0), n, erf, add, mul);
                    changed = true;
                    break;
                }
            }
        }
    }

    private static Node Act(ActivationKind kind, string input, float alpha, float beta) =>
        new("Act", [input], [""]) { Activation = kind, Alpha = alpha, Beta = beta };

    /// <summary>Puts a fused node in place of the last node of a pattern and removes the pattern's other nodes.</summary>
    private static void Replace(Graph g, Node last, Node fused, params Node[] others)
    {
        fused.Outputs[0] = last.Output;
        g.Nodes[g.Nodes.IndexOf(last)] = fused;
        foreach (var o in others) g.Nodes.Remove(o);
    }

    /// <summary>An Act node right after a Conv, ConvTranspose or Linear that nothing else reads joins it.</summary>
    private static void FuseActivations(Graph g)
    {
        var consumers = g.Consumers();
        var producers = g.Producers();
        foreach (var act in g.Nodes.Where(n => n.Op == "Act").ToList())
        {
            if (!producers.TryGetValue(act.Inputs[0], out var p) || p.Op is not ("Conv" or "ConvTranspose" or "Linear")) continue;
            if (p.Activation != ActivationKind.None || p.Residual is not null || SingleConsumer(g, consumers, p.Output) != act) continue;
            p.Activation = act.Activation;
            p.Alpha = act.Alpha;
            p.Beta = act.Beta;
            TakeOutput(g, p, act);
            consumers = g.Consumers();
            producers = g.Producers();
        }
    }

    /// <summary>
    /// mean = ReduceMean(x), d = x - mean, var = ReduceMean(d^2), y = d / sqrt(var + eps) * gamma + beta, over the
    /// last axis.
    /// </summary>
    private static void FuseLayerNorm(Graph g)
    {
        var consumers = g.Consumers();
        foreach (var mean in g.Nodes.Where(n => n.Op == "ReduceMean").ToList())
        {
            if (!g.Nodes.Contains(mean)) continue;
            string x = mean.Inputs[0];
            int rank = g.Rank(x);
            if (!LastAxisOnly(mean, rank)) continue;
            var sub = SingleConsumer(g, consumers, mean.Output);
            if (sub is not { Op: "Sub" } || sub.Inputs[0] != x || sub.Inputs[1] != mean.Output) continue;
            var subUsers = consumers[sub.Output];
            var pow = subUsers.FirstOrDefault(u => u.Op == "Pow");
            var div = subUsers.FirstOrDefault(u => u.Op == "Div");
            if (subUsers.Count != 2 || pow is null || div is null || !IsScalar(g, pow.Inputs[1], 2f) || div.Inputs[0] != sub.Output) continue;
            var mean2 = SingleConsumer(g, consumers, pow.Output);
            if (mean2 is not { Op: "ReduceMean" } || !LastAxisOnly(mean2, rank)) continue;
            var addEps = SingleConsumer(g, consumers, mean2.Output);
            if (addEps is not { Op: "Add" } || !IsScalar(g, Other(addEps, mean2.Output), out float epsilon)) continue;
            var sqrt = SingleConsumer(g, consumers, addEps.Output);
            if (sqrt is not { Op: "Sqrt" } || SingleConsumer(g, consumers, sqrt.Output) != div || div.Inputs[1] != sqrt.Output) continue;
            var mulGamma = SingleConsumer(g, consumers, div.Output);
            if (mulGamma is not { Op: "Mul" } || !g.Constants.TryGetValue(Other(mulGamma, div.Output), out var gamma)) continue;
            var addBeta = SingleConsumer(g, consumers, mulGamma.Output);
            if (addBeta is not { Op: "Add" } || !g.Constants.TryGetValue(Other(addBeta, mulGamma.Output), out var beta)) continue;
            var ln = new Node("LayerNorm", [x, Other(mulGamma, div.Output), Other(addBeta, mulGamma.Output)], [""]) { Alpha = epsilon };
            if (gamma.Count != g.ShapesOf(x)[0][^1] || beta.Count != gamma.Count) continue;
            Replace(g, addBeta, ln, mean, sub, pow, mean2, addEps, sqrt, div, mulGamma);
            consumers = g.Consumers();
        }
    }

    private static bool LastAxisOnly(Node reduce, int rank)
    {
        var axes = reduce.Ints("axes", []);
        return axes.Length == 1 && (axes[0] == -1 || axes[0] == rank - 1) && reduce.Int("keepdims", 1) != 0;
    }

    /// <summary>
    /// Channel means, 1x1 convolution with an activation, another with an activation, and the product with the input
    /// (optionally added back to the input): a squeeze-and-excitation block.
    /// </summary>
    private static void FuseSqueezeExcite(Graph g)
    {
        var consumers = g.Consumers();
        foreach (var pool in g.Nodes.Where(n => n.Op is "GlobalAveragePool" or "ReduceMean").ToList())
        {
            if (!g.Nodes.Contains(pool)) continue;
            string x = pool.Inputs[0];
            if (g.Rank(x) != 4) continue;
            if (pool.Op == "ReduceMean" && !(pool.Ints("axes", []).Order().SequenceEqual(new long[] { 2, 3 }) && pool.Int("keepdims", 1) != 0)) continue;
            var c1 = SingleConsumer(g, consumers, pool.Output);
            if (c1 is null || !IsPointwiseConv(g, c1)) continue;
            var c2 = SingleConsumer(g, consumers, c1.Output);
            if (c2 is null || !IsPointwiseConv(g, c2)) continue;
            var mul = SingleConsumer(g, consumers, c2.Output);
            if (mul is not { Op: "Mul" } || Other(mul, c2.Output) != x) continue;
            var se = new Node("SqueezeExcite", [x, c1.Inputs[1], c2.Inputs[1]], [""])
            {
                Activation = c1.Activation,
                Alpha = c1.Alpha,
                Beta = c1.Beta,
                Activation2 = c2.Activation,
                Alpha2 = c2.Alpha,
                Beta2 = c2.Beta,
            };
            se.Factors.Clear();
            se.Bias = (c1.Bias ?? new float[g.Constants[c1.Inputs[1]].Dims[0]]).Concat(c2.Bias ?? new float[g.Constants[c2.Inputs[1]].Dims[0]]).ToArray();
            var add = SingleConsumer(g, consumers, mul.Output);
            if (add is { Op: "Add" } && Other(add, mul.Output) == x)
            {
                se.AddInput = true;
                Replace(g, add, se, pool, c1, c2, mul);
            }
            else
            {
                Replace(g, mul, se, pool, c1, c2);
            }
            consumers = g.Consumers();
        }
    }

    private static bool IsPointwiseConv(Graph g, Node n) =>
        n.Op == "Conv" && n.Residual is null && n.Int("group", 1) == 1 && g.Constants.TryGetValue(n.Inputs[1], out var w)
        && w.Dims.Length == 4 && w.Dims[2] == 1 && w.Dims[3] == 1 && n.Ints("strides", [1, 1]).All(s => s == 1)
        && n.Ints("pads", [0, 0, 0, 0]).All(p => p == 0);

    /// <summary>
    /// The multi-head attention core between the qkv projection and the output projection: Reshape [0, -1, 3, H, D],
    /// Transpose [2, 0, 3, 1, 4], three Slice/Squeeze pairs for q, k and v, q * scale (+ 0), q k^T, Softmax, times v,
    /// Transpose [0, 2, 1, 3] and Reshape back to [N, T, C].
    /// </summary>
    private static void FuseAttention(Graph g)
    {
        var consumers = g.Consumers();
        var producers = g.Producers();
        Node? P(string t, string op) => producers.TryGetValue(t, out var n) && n.Op == op ? n : null;
        foreach (var softmax in g.Nodes.Where(n => n.Op == "Softmax").ToList())
        {
            var scores = P(softmax.Inputs[0], "MatMul");
            var pv = SingleConsumer(g, consumers, softmax.Output);
            if (scores is null || pv is not { Op: "MatMul" } || pv.Inputs[0] != softmax.Output) continue;
            var outT = SingleConsumer(g, consumers, pv.Output);
            if (outT is not { Op: "Transpose" } || !outT.Ints("perm", []).SequenceEqual(new long[] { 0, 2, 1, 3 })) continue;
            var outR = SingleConsumer(g, consumers, outT.Output);
            if (outR is not { Op: "Reshape" }) continue;
            var kT = P(scores.Inputs[1], "Transpose");
            if (kT is null || !kT.Ints("perm", []).SequenceEqual(new long[] { 0, 1, 3, 2 })) continue;
            // q * scale, optionally + 0.
            string qs = scores.Inputs[0];
            var removed = new List<Node> { scores, pv, outT, kT, softmax };
            var addZero = P(qs, "Add");
            if (addZero is not null && IsScalar(g, addZero.Inputs[1], 0f))
            {
                removed.Add(addZero);
                qs = addZero.Inputs[0];
            }
            var mulScale = P(qs, "Mul");
            if (mulScale is null || !IsScalar(g, mulScale.Inputs[1], out float scale)) continue;
            removed.Add(mulScale);
            string? source = null;
            bool ok = true;
            string[] parts = [mulScale.Inputs[0], kT.Inputs[0], pv.Inputs[1]];
            for (int part = 0; part < 3 && ok; part++)
            {
                var squeeze = P(parts[part], "Squeeze");
                var slice = squeeze is null ? null : P(squeeze.Inputs[0], "Slice");
                if (squeeze is null || slice is null || !squeeze.Ints("axes", []).SequenceEqual(new long[] { 0 })
                    || !IsScalarInt(g, slice.Inputs[1], part) || !IsScalarInt(g, slice.Inputs[2], part + 1)
                    || (slice.Inputs.Count > 3 && !IsScalarInt(g, slice.Inputs[3], 0)))
                {
                    ok = false;
                    break;
                }
                if (source is not null && source != slice.Inputs[0]) ok = false;
                source = slice.Inputs[0];
                removed.Add(squeeze);
                removed.Add(slice);
            }
            if (!ok || source is null) continue;
            var transpose = P(source, "Transpose");
            if (transpose is null || !transpose.Ints("perm", []).SequenceEqual(new long[] { 2, 0, 3, 1, 4 })) continue;
            var reshape = P(transpose.Inputs[0], "Reshape");
            if (reshape is null) continue;
            var split = g.ShapesOf(reshape.Output)[0];
            if (split.Length != 5 || split[2] != 3) continue;
            string qkv = reshape.Inputs[0];
            var qkvShape = g.ShapesOf(qkv)[0];
            if (qkvShape.Length != 3 || qkvShape[2] != 3 * split[3] * split[4]) continue;
            removed.Add(transpose);
            removed.Add(reshape);
            var attention = new Node("Attention", [qkv], [""]) { Heads = (int)split[3], Scale = scale };
            Replace(g, outR, attention, removed.ToArray());
            consumers = g.Consumers();
            producers = g.Producers();
        }
    }

    private static bool IsScalarInt(Graph g, string name, long expected) =>
        g.Constants.TryGetValue(name, out var c) && c.Count == 1 && c[0] == expected;

    /// <summary>
    /// Nearest-neighbour Resize by integer factors becomes Upsample, which an Add or a Concat that is its only reader
    /// absorbs, so the enlarged map is never stored.
    /// </summary>
    private static void FuseUpsampling(Graph g)
    {
        foreach (var n in g.Nodes.Where(n => n.Op == "Resize").ToList())
        {
            if (n.Text("mode", "nearest") != "nearest" || n.Text("coordinate_transformation_mode", "half_pixel") != "asymmetric"
                || n.Text("nearest_mode", "round_prefer_floor") != "floor")
                throw new NotSupportedException($"{n}: only nearest, asymmetric, floor resizing is supported.");
            var scales = g.Constants[n.Inputs[2]];
            if (scales.Count != 4 || scales[0] != 1 || scales[1] != 1 || scales[2] != Math.Floor(scales[2]) || scales[3] != Math.Floor(scales[3]))
                throw new NotSupportedException($"{n}: only integer spatial scales are supported.");
            n.Op = "Upsample";
            n.Inputs.RemoveRange(1, n.Inputs.Count - 1);
            n.Factors.Add(((int)scales[2], (int)scales[3]));
        }
        var consumers = g.Consumers();
        var producers = g.Producers();
        foreach (var add in g.Nodes.Where(n => n.Op == "Add").ToList())
        {
            for (int side = 1; side >= 0; side--)
            {
                if (!producers.TryGetValue(add.Inputs[side], out var up) || up.Op != "Upsample" || SingleConsumer(g, consumers, up.Output) != add) continue;
                string other = add.Inputs[1 - side];
                if (g.IsConstant(other)) continue;
                // The runtime enlarges the second input, so the plain input goes first.
                add.Inputs[0] = other;
                add.Inputs[1] = up.Inputs[0];
                add.Factors.Add(up.Factors[0]);
                g.Nodes.Remove(up);
                break;
            }
        }
        consumers = g.Consumers();
        producers = g.Producers();
        foreach (var concat in g.Nodes.Where(n => n.Op == "Concat").ToList())
        {
            concat.Factors.Clear();
            for (int i = 0; i < concat.Inputs.Count; i++)
            {
                if (producers.TryGetValue(concat.Inputs[i], out var up) && up.Op == "Upsample" && SingleConsumer(g, consumers, up.Output) == concat)
                {
                    concat.Inputs[i] = up.Inputs[0];
                    concat.Factors.Add(up.Factors[0]);
                    g.Nodes.Remove(up);
                }
                else
                {
                    concat.Factors.Add((1, 1));
                }
            }
        }
    }

    /// <summary>
    /// An Add of a Conv or Linear result that nothing else reads and another tensor of the same shape becomes that
    /// layer's residual, added after its activation. The fused layer moves to the Add's place, after both inputs exist.
    /// </summary>
    private static void FuseResiduals(Graph g)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            var consumers = g.Consumers();
            var producers = g.Producers();
            foreach (var add in g.Nodes.Where(n => n.Op == "Add" && n.Factors.Count == 0).ToList())
            {
                if (add.Inputs[0] == add.Inputs[1] || add.Inputs.Any(g.IsConstant)) continue;
                Node? best = null;
                string? other = null;
                foreach (string side in add.Inputs)
                {
                    if (!producers.TryGetValue(side, out var p) || p.Op is not ("Conv" or "Linear") || p.Residual is not null) continue;
                    if (SingleConsumer(g, consumers, side) != add || !g.SameShape(add.Inputs[0], add.Inputs[1])) continue;
                    if (best is null || g.Nodes.IndexOf(p) > g.Nodes.IndexOf(best))
                    {
                        best = p;
                        other = Other(add, side);
                    }
                }
                if (best is null) continue;
                best.Residual = other;
                TakeOutput(g, best, add);
                changed = true;
                break;
            }
        }
    }
}
