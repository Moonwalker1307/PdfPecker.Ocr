using System.Numerics;
using System.Runtime.CompilerServices;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// Squeeze and excitation: per batch item, the channel means go through two small fully connected layers, and the
/// result scales every channel plane. The scaling reads each value before writing it, so it can run in place.
/// </summary>
internal sealed unsafe class SqueezeExciteLayer : Layer
{
    private static readonly int Width = Vector<float>.Count;
    private readonly float[] _w1, _b1, _w2, _b2;
    private readonly int _channels, _hidden, _hiddenStride, _channelStride;
    private readonly ActivationKind _act1, _act2;
    private readonly float _alpha1, _beta1, _alpha2, _beta2;
    private readonly bool _addInput;

    public SqueezeExciteLayer(NodeDefinition node, ModelDefinition model)
        : base(node)
    {
        _act1 = (ActivationKind)Int(node, 0);
        _act2 = (ActivationKind)Int(node, 1);
        _addInput = Int(node, 2) != 0;
        _alpha1 = Float(node, 0);
        _beta1 = Float(node, 1);
        _alpha2 = Float(node, 2);
        _beta2 = Float(node, 3);
        var w1 = Weight(node, model, 0);
        var w2 = Weight(node, model, 2);
        var b1 = Weight(node, model, 1).Data;
        var b2 = Weight(node, model, 3).Data;
        if (w1.Shape.Length != 2 || w2.Shape.Length != 2) throw new InvalidDataException("Squeeze-excitation weights must be matrices.");
        _hidden = w1.Shape[0];
        _channels = w2.Shape[0];
        if (w1.Shape[1] != _channels || w2.Shape[1] != _hidden || b1.Length != _hidden || b2.Length != _channels)
            throw new InvalidDataException("Squeeze-excitation weights do not match.");

        // Both small layers are kept transposed and padded with zeros to whole vectors, so that one vector computes
        // several outputs, each summing its inputs in order as a loop over one output would.
        _hiddenStride = (_hidden + Width - 1) / Width * Width;
        _channelStride = (_channels + Width - 1) / Width * Width;
        _w1 = Transpose(w1.Data, _hidden, _channels, _hiddenStride);
        _w2 = Transpose(w2.Data, _channels, _hidden, _channelStride);
        _b1 = new float[_hiddenStride];
        _b2 = new float[_channelStride];
        b1.CopyTo(_b1, 0);
        b2.CopyTo(_b2, 0);
    }

    /// <summary>A rows x cols matrix as cols rows of stride values each, the first rows of them the matrix's columns.</summary>
    private static float[] Transpose(float[] matrix, int rows, int cols, int stride)
    {
        var result = new float[(long)cols * stride];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[(long)c * stride + r] = matrix[(long)r * cols + c];
        return result;
    }

    public override int[] OutputShape(int[][] inputs)
    {
        RequireRank(inputs[0], 4, "Squeeze-excitation");
        if (inputs[0][1] != _channels) throw Unsupported($"Squeeze-excitation expects {_channels} channels.");
        return (int[])inputs[0].Clone();
    }

    public override int InPlaceInput(int[][] inputs, int[] output) => 0;

    // Each channel plane is scaled by one value, the same for all its columns; channels are scaled differently.
    public override void FindRuns(Step step)
    {
        if (step.InputRuns[0].Axis >= 2) KeepRuns(step, mixesLast: false);
        else step.OutputRuns.Clear();
    }

    public override long SharedScratch(Step step) => (long)step.OutputShape[0] * _channels * 2;

    public override long ScratchPerWorker(Step step, InferenceContext context) => _hiddenStride + _channelStride;

    public override void Execute(Step step, InferenceContext context)
    {
        int batch = step.InputShapes[0][0];
        context.For(batch * _channels, step, &Mean);
        context.For(batch, step, &Excite);
        context.For(batch * _channels, step, &ScalePlane);
    }

    /// <summary>The two small layers for one batch item: its channel means to its channel scales.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Excite(object state, int n, int worker)
    {
        var step = (Step)state;
        var layer = (SqueezeExciteLayer)step.Layer;
        int batch = step.InputShapes[0][0], channels = layer._channels, hiddenCount = layer._hidden;
        float* mean = step.Context.SharedScratch + (long)batch * channels + (long)n * channels;
        float* hidden = step.Context.Scratch(worker);
        float* scale = hidden + layer._hiddenStride;
        fixed (float* w1 = layer._w1, b1 = layer._b1, w2 = layer._w2, b2 = layer._b2)
        {
            for (int j = 0; j < layer._hiddenStride; j += Width)
            {
                var acc = VectorMath.Load(b1 + j);
                for (int c = 0; c < channels; c++) acc += VectorMath.Load(w1 + (long)c * layer._hiddenStride + j) * new Vector<float>(mean[c]);
                VectorMath.Store(VectorMath.Activate(acc, layer._act1, layer._alpha1, layer._beta1), hidden + j);
            }

            for (int c = 0; c < layer._channelStride; c += Width)
            {
                var acc = VectorMath.Load(b2 + c);
                for (int j = 0; j < hiddenCount; j++) acc += VectorMath.Load(w2 + (long)j * layer._channelStride + c) * new Vector<float>(hidden[j]);
                VectorMath.Store(VectorMath.Activate(acc, layer._act2, layer._alpha2, layer._beta2), scale + c);
            }
        }

        new Span<float>(scale, channels).CopyTo(new Span<float>(step.Context.SharedScratch + (long)n * channels, channels));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Mean(object state, int plane, int worker)
    {
        var step = (Step)state;
        var layer = (SqueezeExciteLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int size = x[2] * x[3];
        float* means = step.Context.SharedScratch + (long)x[0] * layer._channels;
        means[plane] = VectorMath.Sum(step.Input(0) + (long)plane * size, size) / size;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ScalePlane(object state, int plane, int worker)
    {
        var step = (Step)state;
        var layer = (SqueezeExciteLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int size = x[2] * x[3];
        float s = step.Context.SharedScratch[plane];
        float* input = step.Input(0) + (long)plane * size;
        float* output = step.Output + (long)plane * size;
        if (layer._addInput) VectorMath.AddScaled(input, s, output, size);
        else VectorMath.MultiplyScalar(input, s, output, size);
    }
}
