using System.Runtime.CompilerServices;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// An experimental 1x1 convolution in 8-bit integers (<see cref="Int8Gemm"/>), which a model gets in place of
/// <see cref="GemmConvLayer"/> only when <see cref="Experiments"/> asks for it and the processor has AVX-VNNI. Each tile
/// quantises the columns of its input planes into its scratch and multiplies them with the quantised weights; the
/// activation and the residual are applied in float, as in the float layer.
/// </summary>
internal sealed unsafe class Int8PointwiseConvLayer : ConvLayer
{
    private readonly Int8Gemm.Weights _weights;
    private readonly float[] _bias;
    private readonly int _k;

    public Int8PointwiseConvLayer(NodeDefinition node, ModelDefinition model)
        : base(node, model)
    {
        if (!IsPointwise(node)) throw new InvalidOperationException("Only a 1x1 convolution without groups, stride or padding runs in 8 bits.");
        _k = CinPerGroup;
        _weights = Int8Gemm.QuantizeWeights(Weight(node, model, 0).Data, Cout, _k);
        _bias = GC.AllocateArray<float>(GemmKernel.RoundUp(Cout, Int8Gemm.MR) + Int8Gemm.MR, pinned: true);
        Weight(node, model, 1).Data.AsSpan(0, Cout).CopyTo(_bias);
    }

    /// <summary>
    /// The fewest input channels worth 8 bits. Below 32 the integer loop is too short to pay for quantising the input
    /// (24 to 48 channels with ReLU took as long as the float layer), so such a layer only loses precision.
    /// </summary>
    public const int MinimumChannels = 32;

    /// <summary>Whether a convolution node is a 1x1 convolution this layer can run and gains from.</summary>
    public static bool Qualifies(NodeDefinition node, ModelDefinition model) =>
        IsPointwise(node) && node.Weights.Length > 0 && model.Weights[node.Weights[0]].Shape is { Length: 4 } shape && shape[1] >= MinimumChannels;

    /// <summary>Whether a convolution node is a 1x1 convolution this layer can run.</summary>
    public static bool IsPointwise(NodeDefinition node) =>
        node.Ints.Length > 10 && node.Ints[0] == 1 && node.Ints[1] == 1 && node.Ints[2] == 1 && node.Ints[3] == 1
        && node.Ints[6] == 0 && node.Ints[7] == 0 && node.Ints[8] == 0 && node.Ints[9] == 0 && node.Ints[10] == 1;

    /// <summary>The columns of one tile: the packed bytes stay around a quarter of a megabyte, as in the float layer.</summary>
    private int TileColumns(int columns)
    {
        int nr = Int8Gemm.NR;
        return Tiling.Balanced(columns, Math.Clamp(262144 / GemmKernel.RoundUp(_k, 4) / nr * nr, nr, 2048), nr);
    }

    public override long ScratchPerWorker(Step step, InferenceContext context)
    {
        int columns = step.OutputShape[2] * step.OutputShape[3];
        return (Int8Gemm.PackedBytes(_k, TileColumns(columns)) + 3) / 4;
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int columns = step.OutputShape[2] * step.OutputShape[3];
        step.Tiling ??= Tiling.For(step.OutputShape[0], columns, TileColumns(columns), Cout, Int8Gemm.MR);
        context.For(step.Tiling.Count, step, &Tile);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile(object state, int tile, int worker)
    {
        var step = (Step)state;
        var layer = (Int8PointwiseConvLayer)step.Layer;
        step.Tiling!.Decode(tile, out int n, out int n0, out int columns, out int m0, out int rows);
        int[] x = step.InputShapes[0];
        int planeOut = step.OutputShape[2] * step.OutputShape[3];
        float* input = step.Input(0) + (long)n * x[1] * x[2] * x[3];
        byte* packed = (byte*)step.Context.Scratch(worker);
        Int8Gemm.QuantizeColumns(input + n0, planeOut, layer._k, columns, packed);
        long baseOffset = (long)n * layer.Cout * planeOut + (long)m0 * planeOut + n0;
        var epilogue = new Epilogue(layer.Activation, layer.Alpha, layer.Beta, layer.HasResidual ? step.Input(1) + baseOffset : null);
        fixed (float* bias = layer._bias)
            Int8Gemm.Multiply(layer._weights, m0, rows, packed, columns, step.Output + baseOffset, planeOut, bias, epilogue);
    }
}
