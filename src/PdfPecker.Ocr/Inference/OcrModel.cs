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

    /// <summary>Loads a model from the file format.</summary>
    public static OcrModel Load(ReadOnlySpan<byte> data, GemmKernel? gemm = null) => FromDefinition(ModelDefinition.Deserialize(data), gemm);

    /// <summary>Builds a model from a definition, as the tests and the model builder do.</summary>
    public static OcrModel FromDefinition(ModelDefinition definition, GemmKernel? gemm = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        return new OcrModel(definition, gemm ?? GemmKernel.Best);
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
        var plan = context.GetPlan(this, input.Shape);
        context.EnsureScratch(plan.ScratchPerWorker, plan.SharedScratch);
        float* arena = context.Arena(plan.ArenaFloats);
        float[] output = context.OutputBuffer(this, plan.OutputLength);
        using var pin = input.Data.Pin();
        fixed (float* outputPointer = output)
        {
            var frame = plan.Frame;
            frame.Arena = arena;
            frame.Input = (float*)pin.Pointer;
            frame.Output = outputPointer;
            var token = context.CancellationToken;
            foreach (var step in plan.Steps)
            {
                token.ThrowIfCancellationRequested();
                step.Layer.Execute(step, context);
            }
        }
        return new Tensor(output.AsMemory(0, plan.OutputLength), plan.OutputShape);
    }

    private void CheckInputShape(int[] shape)
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
