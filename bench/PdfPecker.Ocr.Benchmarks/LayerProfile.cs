#if OCR_INTERNALS
using System.Diagnostics;
using System.Globalization;
using PdfPecker;
using PdfPecker.Ocr;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;

namespace OcrBench;

/// <summary>
/// Where each network's time goes, layer by layer, on the inputs the engine gives it for the A4 invoice page at 300 dpi.
/// The pipeline runs once with each network behind a wrapper that keeps a copy of every input; the inputs are then run
/// again, one step of the execution plan at a time, and each step is timed. The table lists every layer with its share
/// of the page, so the next optimisation goes where the time is:
///   dotnet run -c Release -p:OcrInternals=true -- layers [--threads n] [--runs n] [--top n]
/// </summary>
static unsafe class LayerProfile
{
    public static void Run(int threads, int runs, int top)
    {
        byte[] pdf = HtmlToPdf.Convert(Stages.InvoiceHtml());
        var raster = PageRaster.Render(PdfRenderer.Open(pdf), 0, 300, 100_000_000, default);
        var view = new PixelView(raster.Pixels.Pixels.ToArray(), raster.Pixels.Width, raster.Pixels.Height, raster.Pixels.Channels);
        raster.Pixels.Dispose();

        using var context = new InferenceContext(threads);
        // The page is gray, so the detector runs on one plane, as in the engine.
        var det = new Recording(new ModelNetwork(ModelStore.DetectionGray, context));
        var detColour = new Recording(new ModelNetwork(ModelStore.Detection, context));
        var rec = new Recording(new ModelNetwork(ModelStore.RecognitionArgMax, context));
        var cls = new Recording(new ModelNetwork(ModelStore.Classifier, context));
        var pipeline = new OcrPipeline(detColour, rec, cls, OcrEngine.Characters, OcrEngine.DetectionOptions, OcrEngine.RecognitionOptions(new OcrOptions()), det, context);
        pipeline.Run(view, 300);

        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine($"PdfPecker.Ocr {OcrInfo.Version}, .NET {Environment.Version}, {Environment.ProcessorCount} cores, MaxDegreeOfParallelism {threads}, GEMM {GemmKernel.Best.Name}");
        Console.WriteLine($"A4 invoice page at 300 dpi, {runs} timed runs after one warm-up; per page, median over runs");
        double all = 0;
        var tables = new List<(string Name, Recording Net, double[] Ms)>();
        foreach (var (name, net) in new[] { ("detection", det), ("orientation", cls), ("recognition", rec) })
        {
            double[] ms = Profile(net, context, runs);
            tables.Add((name, net, ms));
            all += ms.Sum();
        }

        foreach (var (name, net, ms) in tables)
        {
            double total = ms.Sum();
            var layers = net.Model.Layers;
            Console.WriteLine();
            Console.WriteLine(string.Create(inv, $"{name}: {net.Inputs.Count} runs per page, {total:F1} ms ({100 * total / all:F0}% of the networks), inputs {Shapes(net)}"));
            if (name != "detection") Console.WriteLine(string.Create(inv, $"  columns with pixels (the rest is the batch's zero padding): {100 * ContentShare(net):F0}%"));
            Console.WriteLine($"  {"#",3} {"layer",-58} {"output (largest run)",-22} {"ms",8} {"share",6}");
            var order = Enumerable.Range(0, ms.Length).OrderByDescending(i => ms[i]).Take(top).OrderBy(i => i);
            foreach (int i in order)
            {
                Console.WriteLine(string.Create(inv, $"  {i,3} {Describe(layers[i], net.LargestPlanStep(context, i)),-58} {Shape(net.LargestPlanStep(context, i).OutputShape),-22} {ms[i],8:F2} {100 * ms[i] / total,5:F1}%"));
            }

            // The rest grouped by kind.
            var byKind = Enumerable.Range(0, ms.Length).GroupBy(i => Kind(layers[i])).Select(g => (g.Key, Ms: g.Sum(i => ms[i]), Count: g.Count())).OrderByDescending(g => g.Ms);
            Console.WriteLine("  by kind: " + string.Join(", ", byKind.Select(g => string.Create(inv, $"{g.Key} x{g.Count} {g.Ms:F1} ms"))));
        }
    }

    /// <summary>Runs every recorded input step by step and returns, per layer, the median over runs of its time per page.</summary>
    static double[] Profile(Recording net, InferenceContext context, int runs)
    {
        var model = net.Model;
        int count = model.Layers.Count;
        var samples = new List<double>[count];
        for (int i = 0; i < count; i++) samples[i] = [];
        for (int run = 0; run <= runs; run++)
        {
            var page = new double[count];
            foreach (var (input, shape) in net.Inputs)
            {
                var plan = context.GetPlan(model, shape);
                context.EnsureScratch(plan.ScratchPerWorker);
                float* arena = context.Arena(plan.ArenaFloats);
                float[] output = context.OutputBuffer(model, plan.OutputLength, out int offset);
                fixed (float* x = input, y = output)
                {
                    plan.Frame.Arena = arena;
                    plan.Frame.Input = x;
                    plan.Frame.Output = y + offset;
                    for (int i = 0; i < plan.Steps.Length; i++)
                    {
                        long s = Stopwatch.GetTimestamp();
                        plan.Steps[i].Layer.Execute(plan.Steps[i], context);
                        page[i] += Stopwatch.GetElapsedTime(s).TotalMilliseconds;
                    }
                }
            }

            if (run == 0) continue;
            for (int i = 0; i < count; i++) samples[i].Add(page[i]);
        }

        return samples.Select(s => s.Order().ElementAt(s.Count / 2)).ToArray();
    }

    /// <summary>The share of input columns that hold pixels of a line rather than the zeros a batch is padded with.</summary>
    static double ContentShare(Recording net)
    {
        long content = 0, total = 0;
        foreach (var (input, shape) in net.Inputs)
        {
            int n = shape[0], h = shape[2], w = shape[3];
            for (int k = 0; k < n; k++)
            {
                int last = 0;
                for (int y = 0; y < h; y++)
                {
                    var row = input.AsSpan((k * shape[1] * h + y) * w, w);
                    int i = row.LastIndexOfAnyExcept(0f);
                    last = Math.Max(last, i + 1);
                }

                content += last;
                total += w;
            }
        }

        return (double)content / total;
    }

    static string Kind(Layer layer) => layer switch
    {
        PdfPecker.Ocr.Inference.Layers.GemmConvLayer => layer.Node.Ints[0] == 1 && layer.Node.Ints[1] == 1 ? "conv1x1" : "conv",
        PdfPecker.Ocr.Inference.Layers.DepthwiseConvLayer => "depthwise",
        PdfPecker.Ocr.Inference.Layers.ArgMaxHeadLayer => "ArgMaxHead",
        _ => layer.Node.Op.ToString(),
    };

    static string Describe(Layer layer, Step step)
    {
        var n = layer.Node;
        int[] x = step.InputShapes[0];
        string extra = n.Op switch
        {
            OpKind.Conv => string.Create(CultureInfo.InvariantCulture, $" {n.Ints[0]}x{n.Ints[1]}/{n.Ints[2]}x{n.Ints[3]} g{n.Ints[10]} {x[1]}->{step.OutputShape[1]} act{n.Ints[11]}{(n.Ints[12] != 0 ? " +res" : "")}"),
            OpKind.Linear when layer is PdfPecker.Ocr.Inference.Layers.ArgMaxHeadLayer => string.Create(CultureInfo.InvariantCulture, $" (linear {x[^1]}->classes, softmax, argmax)"),
            OpKind.Linear => string.Create(CultureInfo.InvariantCulture, $" {x[^1]}->{step.OutputShape[^1]} act{n.Ints[0]}{(n.Ints[1] != 0 ? " +res" : "")}"),
            OpKind.MaxPool or OpKind.AveragePool => string.Create(CultureInfo.InvariantCulture, $" {n.Ints[0]}x{n.Ints[1]}/{n.Ints[2]}x{n.Ints[3]} pad {n.Ints[4]},{n.Ints[5]},{n.Ints[6]},{n.Ints[7]} on [{string.Join(",", x)}]"),
            _ => "",
        };
        return Kind(layer) + extra;
    }

    static string Shape(int[] s) => "[" + string.Join(",", s.Select(d => d.ToString(CultureInfo.InvariantCulture))) + "]";

    static string Shapes(Recording net)
    {
        var groups = net.Inputs.GroupBy(i => Shape(i.Shape)).Select(g => g.Count() > 1 ? $"{g.Count()} x {g.Key}" : g.Key);
        string text = string.Join(" ", groups);
        return text.Length > 160 ? text[..160] + "..." : text;
    }

    /// <summary>A network that keeps a copy of every input it runs, and the model behind it.</summary>
    sealed class Recording(ModelNetwork inner) : INetwork
    {
        public OcrModel Model => inner.Model;

        public List<(float[] Input, int[] Shape)> Inputs { get; } = [];

        public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
        {
            Inputs.Add((input[..Tensor.ElementCount(shape)].ToArray(), shape.ToArray()));
            return inner.Run(input, shape, out outputShape);
        }

        public Span<float> InputBuffer(ReadOnlySpan<int> shape) => inner.InputBuffer(shape);

        /// <summary>The step of layer i in the plan of the largest recorded input, for describing the layer.</summary>
        public Step LargestPlanStep(InferenceContext context, int i)
        {
            var largest = Inputs.MaxBy(x => x.Input.Length);
            return context.GetPlan(Model, largest.Shape).Steps[i];
        }
    }
}
#endif
