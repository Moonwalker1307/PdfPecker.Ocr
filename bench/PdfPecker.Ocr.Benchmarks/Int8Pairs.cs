#if OCR_INTERNALS
using System.Diagnostics;
using System.Globalization;
using PdfPecker;
using PdfPecker.Ocr;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Inference.Layers;
using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;

namespace OcrBench;

/// <summary>
/// The float and the experimental 8-bit networks in one process, on the inputs the engine gives them for the A4 invoice
/// page at 300 dpi, run in turns (float, 8-bit, float, ...) so that both see the same machine from one second to the
/// next: on a shared machine that pairs the runs far better than two processes minutes apart.
///   dotnet run -c Release -p:OcrInternals=true -- int8-pairs [--threads n] [--runs n]
/// </summary>
static unsafe class Int8Pairs
{
    public static void Run(int threads, int runs)
    {
        byte[] pdf = HtmlToPdf.Convert(Stages.InvoiceHtml());
        var raster = PageRaster.Render(PdfRenderer.Open(pdf), 0, 300, 100_000_000, default);
        var view = new PixelView(raster.Pixels.Pixels.ToArray(), raster.Pixels.Width, raster.Pixels.Height, raster.Pixels.Channels);
        raster.Pixels.Dispose();

        using var context = new InferenceContext(threads);
        var floats = new[] { ModelStore.Load("det", false).WithGrayInput(), ModelStore.Load("cls", false), ModelStore.Load("rec", false).WithArgMaxOutput() };
        var int8 = new[] { ModelStore.Load("det", true).WithGrayInput(), ModelStore.Load("cls", true), ModelStore.Load("rec", true).WithArgMaxOutput() };
        var recordings = floats.Select(m => new Recording(new ModelNetwork(m, context))).ToArray();
        var detColour = new ModelNetwork(ModelStore.Load("det", false), context);
        var pipeline = new OcrPipeline(detColour, recordings[2], recordings[1], OcrEngine.Characters, OcrEngine.DetectionOptions, OcrEngine.RecognitionOptions(new OcrOptions()), recordings[0], context);
        pipeline.Run(view, 300);

        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine($"PdfPecker.Ocr {OcrInfo.Version}, .NET {Environment.Version}, MaxDegreeOfParallelism {threads}, int8 supported {PdfPecker.Ocr.Inference.Kernels.Int8Gemm.IsSupported}");
        Console.WriteLine($"A4 invoice page at 300 dpi, float and 8-bit networks in turns, {runs} timed pairs after one warm-up; ms per page, median (min)");
        Console.WriteLine($"{"network",-12} {"float total",16} {"8-bit total",16} {"change",7} {"float 1x1",16} {"8-bit 1x1",16} {"change",7}");
        string[] names = ["detection", "orientation", "recognition"];
        double sumFloat = 0, sumInt8 = 0;
        for (int n = 0; n < 3; n++)
        {
            var inputs = recordings[n].Inputs;
            var f = new List<(double Total, double Pointwise)>();
            var q = new List<(double Total, double Pointwise)>();
            for (int run = 0; run <= runs; run++)
            {
                var a = Time(floats[n], inputs, context);
                var b = Time(int8[n], inputs, context);
                if (run == 0) continue;
                f.Add(a);
                q.Add(b);
            }

            double ft = Median(f.Select(x => x.Total)), qt = Median(q.Select(x => x.Total));
            double fp = Median(f.Select(x => x.Pointwise)), qp = Median(q.Select(x => x.Pointwise));
            sumFloat += ft;
            sumInt8 += qt;
            Console.WriteLine(string.Create(inv, $"{names[n],-12} {ft,8:F1} ({f.Min(x => x.Total),5:F0}) {qt,8:F1} ({q.Min(x => x.Total),5:F0}) {100 * (qt / ft - 1),6:F1}% {fp,8:F1} ({f.Min(x => x.Pointwise),5:F0}) {qp,8:F1} ({q.Min(x => x.Pointwise),5:F0}) {100 * (qp / fp - 1),6:F1}%"));
        }

        Console.WriteLine(string.Create(inv, $"{"all three",-12} {sumFloat,8:F1}         {sumInt8,8:F1}         {100 * (sumInt8 / sumFloat - 1),6:F1}%"));
    }

    /// <summary>One run of every recorded input, step by step: the whole time and the time of the 1x1 convolutions.</summary>
    static (double Total, double Pointwise) Time(OcrModel model, List<(float[] Input, int[] Shape)> inputs, InferenceContext context)
    {
        double total = 0, pointwise = 0;
        foreach (var (input, shape) in inputs)
        {
            var plan = context.GetPlan(model, shape);
            context.EnsureScratch(plan.ScratchPerWorker, plan.SharedScratch);
            float* arena = context.Arena(plan.ArenaFloats);
            float[] output = context.OutputBuffer(model, plan.OutputLength, out int offset);
            fixed (float* x = input, y = output)
            {
                plan.Frame.Arena = arena;
                plan.Frame.Input = x;
                plan.Frame.Output = y + offset;
                foreach (var step in plan.Steps)
                {
                    long s = Stopwatch.GetTimestamp();
                    step.Layer.Execute(step, context);
                    double ms = Stopwatch.GetElapsedTime(s).TotalMilliseconds;
                    total += ms;
                    if (step.Layer is Int8PointwiseConvLayer || (step.Layer is GemmConvLayer && step.Layer.Node.Ints[0] == 1 && step.Layer.Node.Ints[1] == 1)) pointwise += ms;
                }
            }
        }

        return (total, pointwise);
    }

    static double Median(IEnumerable<double> values)
    {
        var v = values.Order().ToArray();
        return v.Length % 2 == 1 ? v[v.Length / 2] : (v[v.Length / 2 - 1] + v[v.Length / 2]) / 2;
    }

    /// <summary>A network that keeps a copy of every input it runs.</summary>
    sealed class Recording(ModelNetwork inner) : INetwork
    {
        public List<(float[] Input, int[] Shape)> Inputs { get; } = [];

        public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
        {
            Inputs.Add((input[..Tensor.ElementCount(shape)].ToArray(), shape.ToArray()));
            return inner.Run(input, shape, out outputShape);
        }
    }
}
#endif
