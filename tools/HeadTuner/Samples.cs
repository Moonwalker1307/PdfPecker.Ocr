using System.Text;
using PdfPecker.Ocr;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Recognition;

namespace HeadTuner;

/// <summary>
/// One time step of a recognised line: the features the network's last layer reads, what the step should be (one of
/// the tuned classes, or any other class with its fixed logit), and the summed exponentials of all classes that are not
/// tuned, which do not change.
/// </summary>
internal sealed record Sample(float[] Features, int Target, float FixedLogit, float LogRest, float Weight);

/// <summary>
/// Reads rendered pages with the converted recogniser, as the pipeline does, and turns each recognised line whose
/// text is known into samples: CTC forced alignment of the known text against the network's output says which class
/// each time step should be.
/// </summary>
internal sealed class SampleCollector
{
    private readonly ModelDefinition _definition;
    private readonly OcrModel _features;
    private readonly OcrModel _full;
    private readonly CharacterSet _characterSet;
    private readonly float[] _weights, _bias;
    private readonly int _k, _n;
    private readonly int[] _classes;
    private readonly Dictionary<int, int> _tuned = [];
    private readonly Dictionary<string, int> _classOf = new(StringComparer.Ordinal);
    public SampleCollector(ModelDefinition definition, int[] classes)
    {
        _definition = definition;
        var (weights, bias) = HeadPatch.FinalLayer(definition);
        _weights = weights.Data;
        _bias = bias.Data;
        _k = weights.Shape[0];
        _n = weights.Shape[1];
        _classes = classes;
        for (int j = 0; j < classes.Length; j++) _tuned[classes[j]] = j;
        for (int i = 1; i < definition.Characters.Length - 1; i++) _classOf.TryAdd(definition.Characters[i], i);
        _classOf[" "] = definition.Characters.Length - 1;
        _full = OcrModel.FromDefinition(definition).WithFusedStem();
        _characterSet = new CharacterSet(definition.Characters[1..^1]);

        // The same network without its last linear layer and softmax returns the features the last layer reads.
        var trunk = new ModelDefinition { Name = definition.Name, Notice = definition.Notice, InputShape = definition.InputShape, Characters = definition.Characters };
        trunk.Weights.AddRange(definition.Weights);
        trunk.Nodes.AddRange(definition.Nodes.Take(definition.Nodes.Count - 2));
        _features = OcrModel.FromDefinition(trunk).WithFusedStem();
    }

    public int Features => _k;

    /// <summary>
    /// Reads one page of <paramref name="lines"/> and returns the samples of every line it recognises, in the order the
    /// pipeline batches them. Each page runs in a context of its own, so pages can be read in parallel.
    /// </summary>
    public PageSamples Read(IReadOnlyList<string> lines, byte[] rgba, int width, int height, float dpi, Random sampling)
    {
        using var context = new InferenceContext(1);
        var capture = new CaptureNetwork(_full, _features, context);
        var pipeline = new OcrPipeline(
            new ModelNetwork(ModelStore.Detection, context), capture, new ModelNetwork(ModelStore.Classifier, context),
            _characterSet, OcrEngine.DetectionOptions, new RecognitionOptions(), null, context);
        pipeline.Run(new PixelView(rgba, width, height, 4), dpi);
        var page = new PageSamples();
        foreach (var (probabilities, features, steps) in capture.Rows) AddLine(page, lines, probabilities, features, steps, sampling);
        return page;
    }

    private void AddLine(PageSamples page, IReadOnlyList<string> lines, float[] p, float[] f, int steps, Random sampling)
    {
        int n = _n;
        string read = Greedy(p, steps);
        string? truth = null;
        int best = int.MaxValue;
        foreach (string line in lines)
        {
            int d = Levenshtein(read, line);
            if (d < best)
            {
                best = d;
                truth = line;
            }
        }

        // A line the detector split, merged or badly misread is no use as a label.
        if (truth is null || best > Math.Max(2, truth.Length / 5) || truth.Any(c => !_classOf.ContainsKey(c.ToString())))
        {
            page.LinesSkipped++;
            return;
        }

        int[] labels = [.. truth.Select(c => _classOf[c.ToString()])];
        int[]? path = Ctc.Align(p, steps, n, labels);
        if (path is null)
        {
            page.LinesSkipped++;
            return;
        }

        page.LinesUsed++;
        for (int t = 0; t < steps; t++)
        {
            var row = p.AsSpan(t * n, n);
            int target = path[t] < 0 ? 0 : labels[path[t]];
            int argMax = row.IndexOf(TensorMax(row));
            float maxTuned = 0;
            int maxClass = -1;
            double rest = 0;
            for (int c = 0; c < n; c++)
            {
                if (_tuned.ContainsKey(c))
                {
                    if (row[c] > maxTuned)
                    {
                        maxTuned = row[c];
                        maxClass = c;
                    }
                }
                else
                {
                    rest += row[c];
                }
            }

            // Steps where no tuned class is in play say little; a twenty-fifth of them is kept, weighted up to match.
            bool relevant = _tuned.ContainsKey(target) || _tuned.ContainsKey(argMax) || maxTuned > 1e-3f;
            float weight = 1;
            if (!relevant)
            {
                if (sampling.Next(25) != 0) continue;
                weight = 25;
            }

            if (maxTuned <= 1e-30f) continue;

            // The softmax's normaliser, from the logit and probability of the likeliest tuned class.
            var features = f.AsSpan(t * _k, _k).ToArray();
            double logit = _bias[maxClass];
            for (int k = 0; k < _k; k++) logit += features[k] * _weights[(k * n) + maxClass];
            double logZ = logit - Math.Log(maxTuned);
            int tuned = _tuned.TryGetValue(target, out int j) ? j : -1;
            float fixedLogit = tuned >= 0 ? 0 : (float)(logZ + Math.Log(Math.Max(row[target], 1e-30f)));
            page.Samples.Add(new Sample(features, tuned, fixedLogit, (float)(logZ + Math.Log(Math.Max(rest, 1e-30))), weight));
        }
    }

    private static float TensorMax(ReadOnlySpan<float> row)
    {
        float max = row[0];
        foreach (float v in row) if (v > max) max = v;
        return max;
    }

    private string Greedy(float[] p, int steps)
    {
        var text = new StringBuilder();
        int previous = -1;
        for (int t = 0; t < steps; t++)
        {
            var row = p.AsSpan(t * _n, _n);
            int best = row.IndexOf(TensorMax(row));
            if (best != previous && best != 0) text.Append(_definition.Characters[best]);
            previous = best;
        }

        return text.ToString().Trim();
    }

    internal static int Levenshtein(string a, string b)
    {
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) previous[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= b.Length; j++)
                current[j] = Math.Min(Math.Min(previous[j] + 1, current[j - 1] + 1), previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            (previous, current) = (current, previous);
        }

        return previous[b.Length];
    }

    /// <summary>
    /// The recogniser behind the pipeline's seam that also keeps, for each line of each batch, the softmax output and
    /// the features of the same input, so the samples see the batches (and their padding) the pipeline makes.
    /// </summary>
    private sealed class CaptureNetwork(OcrModel full, OcrModel features, InferenceContext context) : INetwork
    {
        public List<(float[] Probabilities, float[] Features, int Steps)> Rows { get; } = [];

        public Span<float> InputBuffer(ReadOnlySpan<int> shape) => new float[Tensor.ElementCount(shape)];

        public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
        {
            float[] copy = input.ToArray();
            int[] inputShape = shape.ToArray();
            float[] f = features.Run(copy, inputShape, context, out int[] featureShape).ToArray();
            float[] p = full.Run(copy, inputShape, context, out outputShape).ToArray();
            int n = outputShape[0], steps = outputShape[1], classes = outputShape[2], k = featureShape[2];
            if (featureShape[1] != steps) throw new InvalidOperationException("The features and the output have different time steps.");
            for (int i = 0; i < n; i++)
                Rows.Add((p.AsSpan(i * steps * classes, steps * classes).ToArray(), f.AsSpan(i * steps * k, steps * k).ToArray(), steps));
            return p;
        }
    }
}

/// <summary>The samples of one page and how many of its lines were used.</summary>
internal sealed class PageSamples
{
    public List<Sample> Samples { get; } = [];

    public int LinesUsed { get; set; }

    public int LinesSkipped { get; set; }
}

/// <summary>CTC forced alignment.</summary>
internal static class Ctc
{
    /// <summary>
    /// The most probable CTC path that spells <paramref name="labels"/>: for each time step, the position in the labels
    /// it emits or repeats, or -1 for a blank. Null when the line has fewer steps than labels.
    /// </summary>
    public static int[]? Align(float[] probabilities, int steps, int classes, int[] labels)
    {
        int length = labels.Length, states = (2 * length) + 1;
        if (length == 0 || steps < length) return null;
        var score = new double[states];
        var next = new double[states];
        var back = new int[steps * states];
        double Log(int t, int s) => Math.Log(Math.Max(probabilities[(t * classes) + (s % 2 == 0 ? 0 : labels[s / 2])], 1e-30));
        Array.Fill(score, double.NegativeInfinity);
        score[0] = Log(0, 0);
        score[1] = Log(0, 1);
        for (int t = 1; t < steps; t++)
        {
            for (int s = 0; s < states; s++)
            {
                double best = score[s];
                int from = s;
                if (s >= 1 && score[s - 1] > best)
                {
                    best = score[s - 1];
                    from = s - 1;
                }

                if (s >= 2 && s % 2 == 1 && labels[s / 2] != labels[(s / 2) - 1] && score[s - 2] > best)
                {
                    best = score[s - 2];
                    from = s - 2;
                }

                next[s] = best + Log(t, s);
                back[(t * states) + s] = from;
            }

            (score, next) = (next, score);
        }

        int state = score[states - 1] >= score[states - 2] ? states - 1 : states - 2;
        if (double.IsNegativeInfinity(score[state])) return null;
        var path = new int[steps];
        for (int t = steps - 1; t >= 0; t--)
        {
            path[t] = state % 2 == 0 ? -1 : state / 2;
            if (t > 0) state = back[(t * states) + state];
        }

        return path;
    }
}
