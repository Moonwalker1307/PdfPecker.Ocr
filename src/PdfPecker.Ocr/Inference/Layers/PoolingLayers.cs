using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Inference.Layers;

/// <summary>Max or average pooling over NCHW planes; padding never wins a maximum.</summary>
internal sealed unsafe class PoolLayer : Layer
{
    private readonly bool _max;
    private readonly int _kh, _kw, _sh, _sw, _pt, _pl, _pb, _pr;
    private readonly bool _countIncludePad;

    public PoolLayer(NodeDefinition node, bool max)
        : base(node)
    {
        _max = max;
        _kh = Int(node, 0);
        _kw = Int(node, 1);
        _sh = Int(node, 2);
        _sw = Int(node, 3);
        _pt = Int(node, 4);
        _pl = Int(node, 5);
        _pb = Int(node, 6);
        _pr = Int(node, 7);
        _countIncludePad = !max && Int(node, 8) != 0;
        if (_kh < 1 || _kw < 1 || _sh < 1 || _sw < 1 || _pt < 0 || _pl < 0 || _pb < 0 || _pr < 0 || _pt >= _kh || _pl >= _kw || _pb >= _kh || _pr >= _kw)
            throw new InvalidDataException("Bad pooling attributes.");
    }

    public override int[] OutputShape(int[][] inputs)
    {
        var x = inputs[0];
        RequireRank(x, 4, "Pooling");
        int oh = (x[2] + _pt + _pb - _kh) / _sh + 1;
        int ow = (x[3] + _pl + _pr - _kw) / _sw + 1;
        if (oh < 1 || ow < 1) throw Unsupported("The input is smaller than the pooling window.");
        return [x[0], x[1], oh, ow];
    }

    /// <summary>
    /// Windows without padding and a horizontal stride of 1 or 2 take the vector path; the rest, and the edges the
    /// vectors cannot reach, the generic one.
    /// </summary>
    private bool Vectorised => _pt == 0 && _pl == 0 && _pb == 0 && _pr == 0 && _sw <= 2 && !IsMax2x2Stride1;

    // The slack after a de-interleaved row, enough for a vector read that starts at its last value.
    private const int Slack = 16;

    private static int HalfLength(int w) => (w + 1) / 2 + Slack;

    // The stem's 2x2, stride 1 maximum may write over its input: each output row reads only its own row and the next,
    // rows go in order, and within a row a value or vector is stored only after every input it reads was loaded, so no
    // input is overwritten before its last reader. Writing in place lets the stem's input share memory with its output.
    public override int InPlaceInput(int[][] inputs, int[] output) => IsMax2x2Stride1 ? 0 : -1;

    public override void FindRuns(Step step) => WindowRuns(step.InputRuns[0], step.OutputRuns, step.OutputShape[3]);

    /// <summary>The runs of an output of the given width from those of the input, for this layer or one inside a fused layer.</summary>
    internal void WindowRuns(ColumnRuns input, ColumnRuns output, int outputWidth) =>
        output.FromWindow(input, 3, _kw, _sw, 1, _pl, outputWidth);

    private bool IsMax2x2Stride1 => _max && _kh == 2 && _kw == 2 && _sh == 1 && _sw == 1 && _pt == 0 && _pl == 0;

    public override long ScratchPerWorker(Step step, InferenceContext context) =>
        Vectorised && _sw == 2 ? (long)_kh * 2 * HalfLength(step.InputShapes[0][3]) : 0;

    public override void Execute(Step step, InferenceContext context) =>
        context.For(step.OutputShape[0] * step.OutputShape[1], step, &Plane);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Plane(object state, int plane, int worker)
    {
        var step = (Step)state;
        var layer = (PoolLayer)step.Layer;
        int[] x = step.InputShapes[0];
        int[] o = step.OutputShape;
        float* input = step.Input(0) + (long)plane * x[2] * x[3];
        float* output = step.Output + (long)plane * o[2] * o[3];
        if (layer.IsMax2x2Stride1)
            MaxPool2x2Stride1(input, x[2], x[3], output, o[2], o[3]);
        else if (layer.Vectorised)
            layer.Windows(input, x[2], x[3], output, o[2], o[3], step.Context.Scratch(worker));
        else
            layer.Generic(input, x[2], x[3], output, o[2], o[3], 0, 0, o[2]);
    }

    /// <summary>
    /// A window without padding, over a vector of output pixels at a time. A horizontal stride of 2 first splits the
    /// window's rows into their even and odd values, so every tap is one contiguous read. Each output takes its
    /// values in the same order and with the same operations as <see cref="Generic"/>, so either path gives the same
    /// bits.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Windows(float* input, int h, int w, float* output, int oh, int ow, float* scratch)
    {
        int width = Vector<float>.Count;
        int half = HalfLength(w);
        var divisor = new Vector<float>(_kh * _kw);
        var start = _max ? new Vector<float>(float.NegativeInfinity) : Vector<float>.Zero;
        for (int y = 0; y < oh; y++)
        {
            float* rows = input + (long)y * _sh * w;
            if (_sw == 2)
            {
                for (int ky = 0; ky < _kh; ky++)
                {
                    float* src = rows + (long)ky * w;
                    float* dst = scratch + (long)ky * 2 * half;
                    VectorMath.CopyEven(src, dst, (w + 1) / 2);
                    if (w > 1) VectorMath.CopyEven(src + 1, dst + half, w / 2);
                }
            }

            // Whole vectors only: without padding, the last of them reads at most the last value of each row (for a
            // stride of 1) or of its half (for 2), and the generic loop finishes the row.
            int done = ow / width * width;

            float* dstRow = output + (long)y * ow;
            for (int ox = 0; ox < done; ox += width)
            {
                var acc = start;
                for (int ky = 0; ky < _kh; ky++)
                {
                    for (int kx = 0; kx < _kw; kx++)
                    {
                        float* src = _sw == 1
                            ? rows + (long)ky * w + ox + kx
                            : scratch + (long)ky * 2 * half + ((kx & 1) == 0 ? 0 : half) + ox + (kx >> 1);
                        var v = VectorMath.Load(src);
                        acc = _max ? System.Numerics.Vector.Max(acc, v) : acc + v;
                    }
                }

                VectorMath.Store(_max ? acc : acc / divisor, dstRow + ox);
            }

            if (done < ow) Generic(input, h, w, output, oh, ow, y, done, y + 1);
        }
    }

    /// <summary>The stem's 2x2, stride 1 maximum with padding only at the bottom and right (SAME_UPPER).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MaxPool2x2Stride1(float* input, int h, int w, float* output, int oh, int ow)
    {
        int width = Vector<float>.Count;
        for (int y = 0; y < oh; y++)
        {
            float* r0 = input + (long)y * w;
            float* r1 = y + 1 < h ? r0 + w : r0;
            float* dst = output + (long)y * ow;
            int x = 0;
            // Vectors cover the pixels whose right neighbour is inside the row.
            for (; x + width < w && x + width <= ow; x += width)
            {
                var a = Vector.Max(VectorMath.Load(r0 + x), VectorMath.Load(r1 + x));
                var b = Vector.Max(VectorMath.Load(r0 + x + 1), VectorMath.Load(r1 + x + 1));
                VectorMath.Store(Vector.Max(a, b), dst + x);
            }
            for (; x < ow; x++)
            {
                float m = Math.Max(r0[x], r1[x]);
                if (x + 1 < w) m = Math.Max(m, Math.Max(r0[x + 1], r1[x + 1]));
                dst[x] = m;
            }
        }
    }

    /// <summary>True for the stem's 2x2, stride 1 maximum padded on the bottom and right, whose output has its input's size.</summary>
    internal bool IsStemMaximum => IsMax2x2Stride1 && _pb == 1 && _pr == 1;

    /// <summary>
    /// The stem's maximum (<see cref="IsStemMaximum"/>) on a block of output pixels, <paramref name="rows"/> rows of
    /// <paramref name="width"/> from (<paramref name="oy0"/>, <paramref name="ox0"/>), for <paramref name="channels"/>
    /// channels, as <see cref="GemmConvLayer.ComputeBlock"/> windows its input and output. Each value is the maximum of
    /// the same values, taken in the same order, as in <see cref="MaxPool2x2Stride1"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void MaxPool2x2Stride1Block(float* input, long inputPlane, long rowStride, int h, int w, int firstRow, int firstColumn,
        int oy0, int ox0, int rows, int width, float* output, long outputPlane, int channels)
    {
        int vector = Vector<float>.Count;
        for (int c = 0; c < channels; c++)
        {
            float* origin = input + c * inputPlane - (long)firstRow * rowStride - firstColumn;
            for (int ly = 0; ly < rows; ly++)
            {
                int y = oy0 + ly;
                float* r0 = origin + (long)y * rowStride;
                float* r1 = y + 1 < h ? r0 + rowStride : r0;
                float* dst = output + c * outputPlane + (long)ly * width - ox0;
                int x = ox0, end = ox0 + width;
                // As in MaxPool2x2Stride1, vectors cover the pixels whose right neighbour is inside the row.
                for (; x + vector < w && x + vector <= end; x += vector)
                {
                    var a = Vector.Max(VectorMath.Load(r0 + x), VectorMath.Load(r1 + x));
                    var b = Vector.Max(VectorMath.Load(r0 + x + 1), VectorMath.Load(r1 + x + 1));
                    VectorMath.Store(Vector.Max(a, b), dst + x);
                }
                for (; x < end; x++)
                {
                    float m = Math.Max(r0[x], r1[x]);
                    if (x + 1 < w) m = Math.Max(m, Math.Max(r0[x + 1], r1[x + 1]));
                    dst[x] = m;
                }
            }
        }
    }

    /// <summary>Output columns [xStart, ow) of rows [yStart, yEnd), by the definition.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Generic(float* input, int h, int w, float* output, int oh, int ow, int yStart, int xStart, int yEnd)
    {
        for (int y = yStart; y < yEnd; y++)
        {
            for (int x = xStart; x < ow; x++)
            {
                int y0 = y * _sh - _pt, x0 = x * _sw - _pl;
                float acc = _max ? float.NegativeInfinity : 0f;
                int count = 0;
                for (int ky = 0; ky < _kh; ky++)
                {
                    int iy = y0 + ky;
                    if ((uint)iy >= (uint)h) continue;
                    for (int kx = 0; kx < _kw; kx++)
                    {
                        int ix = x0 + kx;
                        if ((uint)ix >= (uint)w) continue;
                        float v = input[(long)iy * w + ix];
                        // maxss rather than Math.Max, whose branches mispredict on maps that are half zeros after
                        // HardSwish; values here are never NaN.
                        acc = _max ? Vector128.Max(Vector128.CreateScalarUnsafe(acc), Vector128.CreateScalarUnsafe(v)).ToScalar() : acc + v;
                        count++;
                    }
                }
                if (!_max)
                {
                    int divisor = _countIncludePad
                        ? (Math.Min(y0 + _kh, h + _pb) - Math.Max(y0, -_pt)) * (Math.Min(x0 + _kw, w + _pr) - Math.Max(x0, -_pl))
                        : count;
                    acc = divisor > 0 ? acc / divisor : 0f;
                }
                output[(long)y * ow + x] = acc;
            }
        }
    }
}

/// <summary>The mean of each channel plane.</summary>
internal sealed unsafe class GlobalAveragePoolLayer(NodeDefinition node) : Layer(node)
{
    public override int[] OutputShape(int[][] inputs)
    {
        RequireRank(inputs[0], 4, "Global average pooling");
        return [inputs[0][0], inputs[0][1], 1, 1];
    }

    public override void Execute(Step step, InferenceContext context)
    {
        int[] x = step.InputShapes[0];
        int planes = x[0] * x[1], size = x[2] * x[3];
        float* input = step.Input(0), output = step.Output;
        for (int p = 0; p < planes; p++) output[p] = VectorMath.Sum(input + (long)p * size, size) / size;
    }
}
