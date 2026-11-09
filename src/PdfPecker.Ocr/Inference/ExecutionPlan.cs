namespace PdfPecker.Ocr.Inference;

/// <summary>Which buffer a value lives in.</summary>
internal enum BufferKind : byte
{
    /// <summary>The context's activation arena.</summary>
    Arena,

    /// <summary>The caller's input tensor, which is never written.</summary>
    Input,

    /// <summary>The context's output buffer for the model.</summary>
    Output,
}

/// <summary>Where a value lives: a buffer and an offset in floats.</summary>
internal readonly record struct Location(BufferKind Buffer, long Offset);

/// <summary>The base addresses of the buffers of one run; set at the start of each run.</summary>
internal sealed unsafe class Frame
{
    public float* Arena;
    public float* Input;
    public float* Output;

    public float* Resolve(Location location) => location.Buffer switch
    {
        BufferKind.Input => Input,
        BufferKind.Output => Output,
        _ => Arena,
    } + location.Offset;
}

/// <summary>A layer in a plan: its shapes for this input size and where its values live.</summary>
internal sealed unsafe class Step(Layer layer, int[][] inputShapes, int[] outputShape, Frame frame, InferenceContext context)
{
    public Layer Layer { get; } = layer;

    public int[][] InputShapes { get; } = inputShapes;

    public int[] OutputShape { get; } = outputShape;

    public Frame Frame { get; } = frame;

    /// <summary>The context the plan belongs to, for the kernels' scratch.</summary>
    public InferenceContext Context { get; } = context;

    public Location[] InputLocations { get; set; } = [];

    public Location OutputLocation { get; set; }

    /// <summary>The tile split of a matrix-product layer for this step's shapes, made on first use.</summary>
    public Layers.Tiling? Tiling { get; set; }

    /// <summary>Whatever else a layer keeps for this step between runs, made on first use.</summary>
    public object? State { get; set; }

    /// <summary>Where the inputs hold equal columns in this run (see <see cref="ColumnRuns"/>).</summary>
    public ColumnRuns[] InputRuns { get; set; } = [];

    /// <summary>Where the output holds equal columns in this run, set by the layer before the run.</summary>
    public ColumnRuns OutputRuns { get; set; } = new(0);

    public float* Input(int index) => Frame.Resolve(InputLocations[index]);

    public float* Output => Frame.Resolve(OutputLocation);
}

/// <summary>
/// The shapes and memory layout of a model for one input shape. Values are placed in one arena by their lifetimes:
/// views share their input's memory, elementwise layers may write over an input that is not needed afterwards, and the
/// inputs of a concatenation along a contiguous axis are written straight into their slice of its output. The rest are
/// placed greedily, largest first, at the lowest offset no value alive at the same time occupies.
/// </summary>
internal sealed class ExecutionPlan
{
    private const int Alignment = 16;

    private ExecutionPlan(int[] inputShape, Step[] steps, Frame frame, ColumnRuns inputRuns)
    {
        InputShape = inputShape;
        Steps = steps;
        Frame = frame;
        InputRuns = inputRuns;
    }

    /// <summary>Where the input holds equal columns in this run.</summary>
    private ColumnRuns InputRuns { get; }

    public int[] InputShape { get; }

    public Step[] Steps { get; }

    public Frame Frame { get; }

    public int[] OutputShape => Steps[^1].OutputShape;

    public int OutputLength { get; private set; }

    public long ArenaFloats { get; private set; }

    public long ScratchPerWorker { get; private set; }

    public long SharedScratch { get; private set; }

    public static ExecutionPlan Create(OcrModel model, int[] inputShape, InferenceContext context)
    {
        var layers = model.Layers;
        int valueCount = layers.Count + 1;
        var shapes = new int[valueCount][];
        var sizes = new long[valueCount];
        shapes[0] = (int[])inputShape.Clone();
        sizes[0] = Tensor.ElementCount(inputShape);
        var frame = new Frame();
        var steps = new Step[layers.Count];
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            var inputs = layer.Node.Inputs.Select(v => shapes[v]).ToArray();
            int[] output = layer.OutputShape(inputs);
            int v = i + 1;
            shapes[v] = output;
            sizes[v] = Tensor.ElementCount(output);
            steps[i] = new Step(layer, inputs, output, frame, context);
        }

        // The last step that reads each value; the model's output is needed after every step.
        var lastUse = new int[valueCount];
        for (int v = 0; v < valueCount; v++) lastUse[v] = v == 0 ? -1 : v - 1;
        for (int i = 0; i < layers.Count; i++)
            foreach (int input in layers[i].Node.Inputs) lastUse[input] = Math.Max(lastUse[input], i);
        int outputValue = valueCount - 1;
        lastUse[outputValue] = int.MaxValue;

        // Union-find with offsets: each value points at a parent whose memory contains it, at an offset.
        var parent = new int[valueCount];
        var offset = new long[valueCount];
        var start = new int[valueCount];
        var end = new int[valueCount];
        var members = new int[valueCount];
        for (int v = 0; v < valueCount; v++)
        {
            parent[v] = v;
            start[v] = v - 1;
            end[v] = lastUse[v];
            members[v] = 1;
        }
        int Find(int v, out long at)
        {
            at = 0;
            while (parent[v] != v)
            {
                at += offset[v];
                v = parent[v];
            }
            return v;
        }
        // Both are roots: the lifetimes and member counts that placement reads are kept on roots only.
        void Join(int child, int root, long at)
        {
            parent[child] = root;
            offset[child] = at;
            start[root] = Math.Min(start[root], start[child]);
            end[root] = Math.Max(end[root], end[child]);
            members[root] += members[child];
        }

        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            var node = layer.Node;
            int output = i + 1;
            if (layer.IsView)
            {
                int root = Find(node.Inputs[0], out long at);
                Join(output, root, at);
                continue;
            }
            int candidate = layer.InPlaceInput(steps[i].InputShapes, steps[i].OutputShape);
            if (candidate >= 0)
            {
                int v = node.Inputs[candidate];
                int root = Find(v, out long at);
                bool usedTwice = node.Inputs.Count(x => x == v) > 1;
                if (root != 0 && at == 0 && end[root] == i && !usedTwice && sizes[v] == sizes[output] && sizes[root] == sizes[output])
                {
                    Join(output, root, 0);
                    continue;
                }
            }
            if (layer is Layers.ConcatLayer concat && concat.CanAliasInputs(steps[i]))
            {
                // An input is written straight into its slice when it fills its memory alone: it owns it, or the
                // values sharing it (a layer that wrote over it in place, a view) have exactly its size. Two inputs
                // with the same memory cannot both move.
                var sources = new int[node.Inputs.Length];
                for (int j = 0; j < sources.Length; j++)
                {
                    int v = node.Inputs[j];
                    int root = Find(v, out long at);
                    sources[j] = root != 0 && at == 0 && sizes[root] == sizes[v] && concat.Factor(j) == (1, 1) ? root : -1;
                }
                long sliceOffset = 0;
                for (int j = 0; j < node.Inputs.Length; j++)
                {
                    int root = sources[j];
                    if (root > 0 && Array.IndexOf(sources, root) == j && Array.LastIndexOf(sources, root) == j)
                        Join(root, output, sliceOffset);
                    // An upsampled input's slice is larger than the input by its factors.
                    var (fh, fw) = concat.Factor(j);
                    sliceOffset += sizes[node.Inputs[j]] * fh * fw;
                }
            }
        }

        int outputRoot = Find(outputValue, out _);
        if (outputRoot == 0) throw new InvalidOperationException("A model whose output is its input is not supported.");

        // Greedy placement of the remaining groups, largest first.
        var roots = Enumerable.Range(1, valueCount - 1).Where(v => parent[v] == v && v != outputRoot)
            .OrderByDescending(v => sizes[v]).ThenBy(v => start[v]).ThenBy(v => v).ToList();
        var placed = new List<(long Offset, long Size, int Start, int End)>();
        var rootOffset = new long[valueCount];
        long arena = 0;
        foreach (int root in roots)
        {
            long size = (sizes[root] + Alignment - 1) / Alignment * Alignment;
            var conflicts = placed.Where(p => p.Start <= end[root] && start[root] <= p.End).OrderBy(p => p.Offset).ToList();
            long at = 0;
            foreach (var p in conflicts)
            {
                if (at + size <= p.Offset) break;
                at = Math.Max(at, p.Offset + p.Size);
            }
            placed.Add((at, size, start[root], end[root]));
            rootOffset[root] = at;
            arena = Math.Max(arena, at + size);
        }

        Location LocationOf(int v)
        {
            int root = Find(v, out long at);
            if (root == 0) return new Location(BufferKind.Input, at);
            if (root == outputRoot) return new Location(BufferKind.Output, at);
            return new Location(BufferKind.Arena, rootOffset[root] + at);
        }

        var runs = shapes.Select(s => new ColumnRuns(s.Length > 0 ? s[0] : 1)).ToArray();
        var plan = new ExecutionPlan((int[])inputShape.Clone(), steps, frame, runs[0])
        {
            ArenaFloats = arena,
            OutputLength = checked((int)sizes[outputRoot]),
        };
        long perWorker = 0, shared = 0;
        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            step.InputLocations = layers[i].Node.Inputs.Select(LocationOf).ToArray();
            step.OutputLocation = LocationOf(i + 1);
            step.InputRuns = layers[i].Node.Inputs.Select(v => runs[v]).ToArray();
            step.OutputRuns = runs[i + 1];
            perWorker = Math.Max(perWorker, layers[i].ScratchPerWorker(step, context));
            shared = Math.Max(shared, layers[i].SharedScratch(step));
        }
        plan.ScratchPerWorker = perWorker;
        plan.SharedScratch = shared;
        return plan;
    }

    /// <summary>
    /// Finds the columns of this run's values that are known to be equal (see <see cref="ColumnRuns"/>): the trailing
    /// columns of each batch item of a [N, C, H, W] input that are equal to its last column, and from them those of
    /// every layer's output.
    /// </summary>
    public unsafe void FindRuns(float* input)
    {
        var runs = InputRuns;
        runs.Clear();
        if (ColumnRuns.Enabled && InputShape.Length == 4 && InputShape[3] > 2)
        {
            int w = InputShape[3];
            long rows = (long)InputShape[1] * InputShape[2];
            runs.Reset(3);
            for (int n = 0; n < InputShape[0]; n++)
                runs.Set(n, TrailingRun((int*)input + (n * rows * w), rows, w), w);
        }

        bool any = runs.Any(3);
        foreach (var step in Steps)
        {
            if (any) step.Layer.FindRuns(step);
            else step.OutputRuns.Clear();
        }
    }

    /// <summary>The first column from which every column of the rows equals the last, compared as bits.</summary>
    private static unsafe int TrailingRun(int* item, long rows, int w)
    {
        int start = 0;
        for (long r = 0; r < rows; r++)
        {
            int* row = item + (r * w);
            int last = row[w - 1];
            int x = w - 1;
            while (x > start && row[x - 1] == last) x--;
            start = x;
            if (start == w - 1) break;
        }
        return start;
    }
}
