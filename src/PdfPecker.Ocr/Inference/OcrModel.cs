using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// A loaded neural network: its layers with weights expanded to float32 and packed for the kernels. It is immutable
/// after loading and can be shared by any number of threads, each running it with its own <see cref="InferenceContext"/>.
/// </summary>
internal sealed unsafe class OcrModel
{
    private readonly int[] _inputShape;

    private OcrModel(ModelDefinition definition, GemmKernel gemm)
    {
        Name = definition.Name;
        Notice = definition.Notice;
        _inputShape = definition.InputShape;
        Characters = definition.Characters;
        Gemm = gemm;
        var layers = new List<Layer>(definition.Nodes.Count);
        foreach (var node in definition.Nodes) layers.Add(Layer.Create(node, definition, gemm));
        Layers = layers;
        if (layers.Count == 0) throw new InvalidDataException("The model has no layers.");
        Inputs = layers.Select(l => l.Node.Inputs).ToArray();
    }

    private OcrModel(OcrModel source, List<Layer> layers, int[][] inputs, int[]? inputShape = null)
    {
        Name = source.Name;
        Notice = source.Notice;
        _inputShape = inputShape ?? source._inputShape;
        Characters = source.Characters;
        Gemm = source.Gemm;
        Layers = layers;
        Inputs = inputs;
    }

    public string Name { get; }

    public string Notice { get; }

    /// <summary>The input shape, with -1 where the caller chooses the size.</summary>
    public IReadOnlyList<int> InputShape => _inputShape;

    /// <summary>The recogniser's classes (empty for other models).</summary>
    public string[] Characters { get; }

    /// <summary>The matrix multiplication kernel the weights were packed for.</summary>
    public GemmKernel Gemm { get; }

    internal IReadOnlyList<Layer> Layers { get; }

    /// <summary>
    /// The values each layer reads, by layer: value 0 is the model's input and value i + 1 is what layer i writes. They
    /// are the nodes' inputs until a variant replaces several layers by one, which numbers the values afresh.
    /// </summary>
    internal int[][] Inputs { get; }

    /// <summary>Loads a model from the file format.</summary>
    public static OcrModel Load(ReadOnlySpan<byte> data, GemmKernel? gemm = null) => FromDefinition(ModelDefinition.Deserialize(data), gemm);

    /// <summary>Builds a model from a definition, as the tests and the model builder do.</summary>
    public static OcrModel FromDefinition(ModelDefinition definition, GemmKernel? gemm = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        return new OcrModel(definition, gemm ?? GemmKernel.Best);
    }

    /// <summary>
    /// The same model on a one-channel input [N, 1, H, W] that stands for three equal channels, as a gray page gives:
    /// bit for bit the output of the model on the plane repeated three times, without building the copies (see
    /// <see cref="Layers.GemmConvLayer.WithBroadcastInput"/>). The layers and their weights are shared with this model.
    /// Fails with <see cref="InvalidOperationException"/> for a model whose input is not three channels read only by a
    /// first convolution that can broadcast.
    /// </summary>
    public OcrModel WithGrayInput()
    {
        var layers = Layers;
        if (_inputShape.Length != 4 || _inputShape[1] != 3 || layers[0] is not Layers.GemmConvLayer { CanBroadcastInput: true } conv
            || Inputs[0] is not [0])
            throw new InvalidOperationException($"{Name} does not start with a convolution of a three-channel input.");
        for (int i = 1; i < layers.Count; i++)
            if (Array.IndexOf(Inputs[i], 0) >= 0)
                throw new InvalidOperationException($"{Name} reads its input after the first convolution.");
        var gray = new List<Layer>(layers) { [0] = conv.WithBroadcastInput() };
        return new OcrModel(this, gray, Inputs, [_inputShape[0], 1, _inputShape[2], _inputShape[3]]);
    }

    /// <summary>
    /// The same model with its final linear layer and softmax replaced by <see cref="Layers.ArgMaxHeadLayer"/>: the
    /// output is [.., 2], the index of the most probable class of each row (as a float, exact below 2^24) and its
    /// probability, bit for bit what the full output gives, without storing the probabilities of every class. The
    /// layers and their weights are shared with this model. Fails with <see cref="InvalidOperationException"/> for a
    /// model that does not end in a plain linear layer whose only reader is a softmax.
    /// </summary>
    public OcrModel WithArgMaxOutput()
    {
        var layers = Layers;
        if (layers.Count < 2 || layers[^1] is not Layers.SoftmaxLayer softmax || layers[^2] is not Layers.LinearLayer { IsPlain: true } linear)
            throw new InvalidOperationException($"{Name} does not end in a linear layer and a softmax.");
        int linearValue = layers.Count - 1;
        if (Inputs[^1] is not [var read] || read != linearValue)
            throw new InvalidOperationException($"{Name}'s softmax does not read its last linear layer.");
        if (linear.OutputFeatures > 1 << 24)
            throw new InvalidOperationException($"{Name} has too many classes for an index stored as a float.");
        var node = new NodeDefinition(OpKind.Linear, linear.Node.Inputs, linearValue, [(int)ActivationKind.None, 0], [0, 0], linear.Node.Weights);
        var fused = new List<Layer>(layers.Count - 1);
        for (int i = 0; i < layers.Count - 2; i++) fused.Add(layers[i]);
        fused.Add(new Layers.ArgMaxHeadLayer(linear, node));
        return new OcrModel(this, fused, Inputs[..^1]);
    }

    /// <summary>
    /// The same model with its first six layers, when they are PP-HGNetV2's stem, replaced by one
    /// <see cref="Layers.StemLayer"/>, which computes the stem's output block by block without its four inner maps: bit
    /// for bit the same output in much less working memory. The other layers and their weights are shared with this
    /// model. A model without that stem is returned as it is.
    /// </summary>
    public OcrModel WithFusedStem()
    {
        var stem = Inference.Layers.StemLayer.TryCreate(Layers, Inputs);
        if (stem is null) return this;
        const int Fused = 5;
        var layers = new List<Layer>(Layers.Count - Fused) { stem };
        var inputs = new List<int[]>(Layers.Count - Fused) { Inputs[0] };
        for (int i = Fused + 1; i < Layers.Count; i++)
        {
            layers.Add(Layers[i]);
            // Values after the stem move down by the five it no longer writes; TryCreate checked that none of them is read.
            inputs.Add(Inputs[i].Select(v => v > Fused ? v - Fused : v).ToArray());
        }

        return new OcrModel(this, layers, [.. inputs]);
    }

    /// <summary>
    /// Runs the model. The returned tensor's storage belongs to the context and is overwritten by the context's next
    /// run of this model; <see cref="Tensor.Clone"/> it to keep it longer. The input is read, never written.
    /// </summary>
    public Tensor Run(Tensor input, InferenceContext context)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(context);
        context.ThrowIfDisposed();
        CheckInputShape(input.Shape);
        using var pin = input.Data.Pin();
        var plan = RunCore((float*)pin.Pointer, input.Shape, context, out float* result);
        float[] output = context.OutputBuffer(this, plan.OutputLength, out int offset);
        new ReadOnlySpan<float>(result, plan.OutputLength).CopyTo(output.AsSpan(offset));
        return new Tensor(output.AsMemory(offset, plan.OutputLength), plan.OutputShape);
    }

    /// <summary>
    /// A block of the context's arena that a run of this model on <paramref name="shape"/> keeps for its input, for the
    /// caller to build the input in and pass to <see cref="Run(ReadOnlySpan{float}, ReadOnlySpan{int}, InferenceContext, out int[])"/>,
    /// which then reads it where it is. It saves the caller a buffer as large as the input (20 MB for a colour page at the
    /// detector's default size) on top of the arena, which already has room for it. The block is valid until the context
    /// runs a model again, and its contents are undefined until written.
    /// </summary>
    /// <param name="shape">The shape of the input the next run will read.</param>
    /// <param name="context">The context of that run.</param>
    public Span<float> InputBuffer(ReadOnlySpan<int> shape, InferenceContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        context.ThrowIfDisposed();
        CheckInputShape(shape);
        var plan = context.GetPlan(this, shape);
        // Allocating the arena now means the run that follows, with the same plan, keeps it rather than growing it.
        float* arena = context.Arena(plan.ArenaFloats);
        return new Span<float>(arena + plan.InputOffset, Tensor.ElementCount(shape));
    }

    /// <summary>
    /// Runs the model on values the caller holds in a span, without copying them, and returns the output where the last
    /// step wrote it, in the context's arena: it stays valid until the context runs a model again. This is the
    /// pipeline's path (see <see cref="ModelNetwork"/>), which saves a copy and a buffer for each input and output.
    /// </summary>
    /// <param name="input">At least as many values as <paramref name="shape"/> holds, in row-major order; read, never written.</param>
    /// <param name="shape">The input's shape.</param>
    /// <param name="context">The context to run in.</param>
    /// <param name="outputShape">The output's shape, a new array the caller may keep.</param>
    public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, InferenceContext context, out int[] outputShape)
    {
        ArgumentNullException.ThrowIfNull(context);
        context.ThrowIfDisposed();
        CheckInputShape(shape);
        if (input.Length < Tensor.ElementCount(shape))
            throw new ArgumentException("The input holds fewer values than its shape needs.", nameof(input));
        fixed (float* pointer = input)
        {
            var plan = RunCore(pointer, shape, context, out float* output);
            outputShape = (int[])plan.OutputShape.Clone();
            return new ReadOnlySpan<float>(output, plan.OutputLength);
        }
    }

    /// <summary>Runs every step of the plan for the input's shape; the output lands in the arena, where output points.</summary>
    private ExecutionPlan RunCore(float* input, ReadOnlySpan<int> shape, InferenceContext context, out float* output)
    {
        var plan = context.GetPlan(this, shape);
        context.EnsureScratch(plan.ScratchPerWorker);
        float* arena = context.Arena(plan.ArenaFloats);
        var frame = plan.Frame;
        frame.Arena = arena;
        frame.Input = input;
        var token = context.CancellationToken;
        plan.FindRuns(input);
        foreach (var step in plan.Steps)
        {
            token.ThrowIfCancellationRequested();
            step.Layer.Execute(step, context);
        }
        output = arena + plan.OutputOffset;
        return plan;
    }

    private void CheckInputShape(ReadOnlySpan<int> shape)
    {
        if (shape.Length != _inputShape.Length)
            throw new ArgumentException($"{Name} takes a {_inputShape.Length}-dimensional input.", "input");
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 0) throw new ArgumentException("Input dimensions must be positive.", "input");
            if (_inputShape[i] >= 0 && shape[i] != _inputShape[i])
                throw new ArgumentException($"{Name} needs dimension {i} of its input to be {_inputShape[i]}.", "input");
        }
    }
}
