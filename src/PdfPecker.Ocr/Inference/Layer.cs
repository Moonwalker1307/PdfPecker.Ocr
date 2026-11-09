using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Inference.Layers;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// One operator of a loaded model, with its weights packed for the kernels. A layer is immutable and shared by all
/// contexts; what depends on the input size lives in the <see cref="Step"/> that an execution plan makes for it.
/// </summary>
internal abstract class Layer
{
    protected Layer(NodeDefinition node)
    {
        Node = node;
    }

    public NodeDefinition Node { get; }

    /// <summary>The output shape for the given input shapes; unsupported shapes throw <see cref="ArgumentException"/>.</summary>
    public abstract int[] OutputShape(int[][] inputs);

    /// <summary>True when the output is the input under another shape, which costs nothing.</summary>
    public virtual bool IsView => false;

    /// <summary>
    /// The index of an input whose memory the output may reuse when that input is not needed afterwards, or -1. Only
    /// layers that read each input value before writing the output value at the same place may say so.
    /// </summary>
    public virtual int InPlaceInput(int[][] inputs, int[] output) => -1;

    /// <summary>Floats of scratch each worker needs for this step.</summary>
    public virtual long ScratchPerWorker(Step step, InferenceContext context) => 0;

    /// <summary>
    /// Floats of scratch shared by the workers of this step (<see cref="Step.SharedScratch"/>). The plan places it in
    /// the arena for the duration of the step, so it costs memory only where it is larger than what other steps need.
    /// </summary>
    public virtual long SharedScratch(Step step) => 0;

    /// <summary>Runs the layer for one step of a plan.</summary>
    public abstract void Execute(Step step, InferenceContext context);

    /// <summary>Creates the layer for a node of a model definition.</summary>
    public static Layer Create(NodeDefinition node, ModelDefinition model, GemmKernel gemm) => node.Op switch
    {
        OpKind.Conv => ConvLayer.CreateConv(node, model, gemm),
        OpKind.ConvTranspose => new ConvTransposeLayer(node, model, gemm),
        OpKind.MaxPool => new PoolLayer(node, max: true),
        OpKind.AveragePool => new PoolLayer(node, max: false),
        OpKind.GlobalAveragePool => new GlobalAveragePoolLayer(node),
        OpKind.SqueezeExcite => new SqueezeExciteLayer(node, model),
        OpKind.Add => new BinaryLayer(node, multiply: false),
        OpKind.Mul => new BinaryLayer(node, multiply: true),
        OpKind.Activation => new ActivationLayer(node),
        OpKind.Concat => new ConcatLayer(node),
        OpKind.LayerNorm => new LayerNormLayer(node, model),
        OpKind.Linear => new LinearLayer(node, model, gemm),
        OpKind.Attention => new AttentionLayer(node),
        OpKind.Softmax => new SoftmaxLayer(node),
        OpKind.Transpose => new TransposeLayer(node),
        OpKind.Reshape => new ReshapeLayer(node),
        OpKind.Upsample => new UpsampleLayer(node),
        _ => throw new InvalidDataException($"Unknown operator {(int)node.Op} in model data."),
    };

    /// <summary>Reads an integer attribute, failing on damaged model data.</summary>
    protected static int Int(NodeDefinition node, int index) =>
        index < node.Ints.Length ? node.Ints[index] : throw new InvalidDataException($"{node.Op} node lacks attribute {index}.");

    /// <summary>Reads a float attribute, failing on damaged model data.</summary>
    protected static float Float(NodeDefinition node, int index) =>
        index < node.Floats.Length ? node.Floats[index] : throw new InvalidDataException($"{node.Op} node lacks attribute {index}.");

    /// <summary>A weight of a node, failing on damaged model data.</summary>
    protected static WeightDefinition Weight(NodeDefinition node, ModelDefinition model, int index) =>
        index < node.Weights.Length ? model.Weights[node.Weights[index]] : throw new InvalidDataException($"{node.Op} node lacks weight {index}.");

    /// <summary>Throws the exception for an input this model cannot run on.</summary>
    protected static ArgumentException Unsupported(string message) => new(message, "input");

    protected static void RequireRank(int[] shape, int rank, string what)
    {
        if (shape.Length != rank) throw Unsupported($"{what} needs a {rank}-dimensional input.");
    }
}
