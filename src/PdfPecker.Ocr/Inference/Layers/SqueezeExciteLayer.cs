using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>
/// Squeeze and excitation: per batch item, the channel means go through two small fully connected layers, and the
/// result scales every channel plane. The scaling reads each value before writing it, so it can run in place.
/// </summary>
internal sealed unsafe class SqueezeExciteLayer : Layer
{
    private readonly float[] _w1, _b1, _w2, _b2;
    private readonly int _channels, _hidden;
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
        _w1 = w1.Data;
        _b1 = Weight(node, model, 1).Data;
        _w2 = w2.Data;
        _b2 = Weight(node, model, 3).Data;
        _hidden = w1.Shape[0];
        _channels = w2.Shape[0];
        if (w1.Shape.Length != 2 || w2.Shape.Length != 2 || w1.Shape[1] != _channels || w2.Shape[1] != _hidden || _b1.Length != _hidden || _b2.Length != _channels)
            throw new InvalidDataException("Squeeze-excitation weights do not match.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        RequireRank(inputs[0], 4, "Squeeze-excitation");
        if (inputs[0][1] != _channels) throw Unsupported($"Squeeze-excitation expects {_channels} channels.");
        return (int[])inputs[0].Clone();
    }

    public override int InPlaceInput(int[][] inputs, int[] output) => 0;

    public override long SharedScratch(Step step) => (long)step.OutputShape[0] * _channels * 2 + _hidden;

    public override void Execute(Step step, InferenceContext context)
    {
        int batch = step.InputShapes[0][0];
        float* scale = context.SharedScratch;
        float* hidden = scale + (long)batch * _channels * 2;
        context.For(batch * _channels, step, &Mean);
        for (int n = 0; n < batch; n++)
        {
            float* mean = scale + (long)batch * _channels + (long)n * _channels;
            for (int j = 0; j < _hidden; j++)
            {
                float acc = _b1[j];
                for (int c = 0; c < _channels; c++) acc += _w1[j * _channels + c] * mean[c];
                hidden[j] = VectorMath.Activate(acc, _act1, _alpha1, _beta1);
            }
            for (int c = 0; c < _channels; c++)
            {
                float acc = _b2[c];
                for (int j = 0; j < _hidden; j++) acc += _w2[c * _hidden + j] * hidden[j];
                scale[n * _channels + c] = VectorMath.Activate(acc, _act2, _alpha2, _beta2);
            }
        }
        context.For(batch * _channels, step, &ScalePlane);
    }

    private static void Mean(object state, int plane, int worker)
    {
        var step = (Step)state;
        var layer = (SqueezeExciteLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int size = x[2] * x[3];
        float* means = step.Context.SharedScratch + (long)x[0] * layer._channels;
        means[plane] = VectorMath.Sum(step.Input(0) + (long)plane * size, size) / size;
    }

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
