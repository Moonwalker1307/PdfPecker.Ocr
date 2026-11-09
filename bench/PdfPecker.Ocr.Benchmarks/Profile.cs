#if OCR_INTERNALS
using System.Diagnostics;
using System.Globalization;
using PdfPecker.Ocr;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Recognition;
using TesseractComparison;

namespace OcrBench;

/// <summary>
/// Where the time of corpus pages goes, down to the layers of each network, so that a slow page group can be traced to an
/// operator rather than guessed at:
///   dotnet run -c Release -p:OcrInternals=true -- profile --corpus bench/TesseractComparison/corpus --only =letter-en.300 [--threads n] [--runs n]
/// Each page runs once to build the plans, then the given number of times with every layer timed. The networks run their
/// plans step by step exactly as <see cref="OcrModel"/> does, with a timestamp around each step, so the figures add up to
/// the network time of the engine plus the cost of reading the clock (well under a microsecond a step). It also prints the
/// shapes the recognizer fed its network, which shows how much of the recognition work is padding of short lines.
/// </summary>
static class Profile
{
    public static void Run(string corpus, string[]? only, int threads, int runs)
    {
        var inv = CultureInfo.InvariantCulture;
        using var context = new InferenceContext(threads);
        var det = new ProfiledNetwork("det", ModelStore.Detection, context);
        var rec = new ProfiledNetwork("rec", ModelStore.Recognition, context);
        var cls = new ProfiledNetwork("cls", ModelStore.Classifier, context);
        var options = new OcrOptions { MaxDegreeOfParallelism = threads };
        var pipeline = new OcrPipeline(det, rec, cls, OcrEngine.Characters, OcrEngine.DetectionOptions, OcrEngine.RecognitionOptions(options));
        Console.WriteLine($"PdfPecker.Ocr {OcrInfo.Version}, .NET {Environment.Version}, {Environment.ProcessorCount} cores, {threads} thread(s), {runs} profiled run(s) per page");
        foreach (var page in CorpusPage.Load(corpus, only))
        {
            var (pixels, w, h, c) = PngDecoder.Decode(File.ReadAllBytes(page.ImagePath));
            var view = new PixelView(pixels, w, h, c);
            double detMs = 0, recMs = 0, totalMs = 0;
            int lines = 0;
            for (int i = 0; i <= runs; i++)
            {
                bool measured = i > 0;
                foreach (var n in new[] { det, rec, cls }) n.Recording = measured;
                long t = Stopwatch.GetTimestamp();
                var quads = pipeline.Detector.Detect(view);
                double d = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
                var recognized = pipeline.Recognizer.Recognize(view, quads);
                double all = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
                lines = recognized.Count;
                if (!measured) continue;
                detMs += d;
                recMs += all - d;
                totalMs += all;
            }
            Console.WriteLine();
            Console.WriteLine(string.Create(inv, $"== {page.Id}: {w} x {h} px, {lines} lines; per run: detection {detMs / runs:F0} ms, orientation and recognition {recMs / runs:F0} ms, total {totalMs / runs:F0} ms"));
            foreach (var n in new[] { det, cls, rec })
            {
                n.Print(runs);
                n.Clear();
            }
        }
    }

    /// <summary>A model behind the pipeline's network seam that times each step of its plan.</summary>
    sealed unsafe class ProfiledNetwork(string name, OcrModel model, InferenceContext context) : INetwork
    {
        readonly Dictionary<string, (long Ticks, int Count)> _byOp = new(StringComparer.Ordinal);
        readonly Dictionary<int, (long Ticks, string Op, string Shape)> _byStep = [];
        readonly List<int[]> _shapes = [];
        long _contentSteps;
        long _ticks;
        int _calls;

        public bool Recording { get; set; }

        public Span<float> InputBuffer(ReadOnlySpan<int> shape) => model.InputBuffer(shape, context);

        public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
        {
            if (!Recording) return model.Run(input, shape, context, out outputShape);
            _calls++;
            _shapes.Add(shape.ToArray());
            if (shape.Length == 4) _contentSteps += ContentSteps(input, shape);
            long start = Stopwatch.GetTimestamp();
            var plan = context.GetPlan(model, shape);
            context.EnsureScratch(plan.ScratchPerWorker);
            float* arena = context.Arena(plan.ArenaFloats);
            fixed (float* inputPointer = input)
            {
                var frame = plan.Frame;
                frame.Arena = arena;
                frame.Input = inputPointer;
                for (int s = 0; s < plan.Steps.Length; s++)
                {
                    var step = plan.Steps[s];
                    long t = Stopwatch.GetTimestamp();
                    step.Layer.Execute(step, context);
                    long dt = Stopwatch.GetTimestamp() - t;
                    string op = OpName(step);
                    var (ticks, count) = _byOp.GetValueOrDefault(op);
                    _byOp[op] = (ticks + dt, count + 1);
                    var prev = _byStep.GetValueOrDefault(s);
                    _byStep[s] = (prev.Ticks + dt, op, string.Join('x', step.OutputShape));
                }
            }
            _ticks += Stopwatch.GetTimestamp() - start;
            outputShape = (int[])plan.OutputShape.Clone();
            return new ReadOnlySpan<float>(arena + plan.OutputOffset, plan.OutputLength);
        }

        /// <summary>
        /// The time steps of the lines themselves: the recognizer clears the batch to zero and writes each line from the left,
        /// and a normalised pixel is zero only by chance, so a line ends at the last column of its first channel that is not.
        /// </summary>
        static long ContentSteps(ReadOnlySpan<float> input, ReadOnlySpan<int> shape)
        {
            int n = shape[0], channels = shape[1], height = shape[2], width = shape[3];
            long steps = 0;
            for (int k = 0; k < n; k++)
            {
                var plane = input.Slice(k * channels * height * width, height * width);
                int last = 0;
                for (int y = 0; y < height; y++)
                {
                    var row = plane.Slice(y * width, width);
                    for (int x = width - 1; x >= last; x--)
                        if (row[x] != 0) { last = x + 1; break; }
                }
                steps += (last + 7) / 8;
            }
            return steps;
        }

        static string OpName(Step step)
        {
            string op = step.Layer.Node.Op.ToString();
            // The recognition model's last matrix product (120 features to every character class) is named on its own,
            // since it is the one layer whose cost grows with the size of the character set.
            if (step.Layer.Node.Op == OpKind.Linear && step.OutputShape[^1] > 10_000) return "Linear (to " + step.OutputShape[^1].ToString(CultureInfo.InvariantCulture) + " classes)";
            if (step.Layer.Node.Op == OpKind.Conv) op += step.InputShapes[0].Length == 4 ? " " + ConvKind(step) : "";
            return op;
        }

        static string ConvKind(Step step)
        {
            // Depthwise, 1x1 and other convolutions take different kernels, so they are told apart by their shapes.
            var ints = step.Layer.Node.Ints;
            int cin = step.InputShapes[0][1], group = ints[10];
            if (group > 1 && group == cin) return "depthwise";
            if (group > 1) return "grouped";
            return ints[0] == 1 && ints[1] == 1 ? "1x1" : ints[0].ToString(CultureInfo.InvariantCulture) + "x" + ints[1].ToString(CultureInfo.InvariantCulture);
        }

        public void Print(int runs)
        {
            var inv = CultureInfo.InvariantCulture;
            double total = Stopwatch.GetElapsedTime(0, _ticks).TotalMilliseconds / runs;
            Console.WriteLine(string.Create(inv, $"  {name}: {total:F1} ms per run in {_calls / runs} call(s)"));
            if (_calls == 0) return;
            foreach (var (op, (ticks, count)) in _byOp.OrderByDescending(k => k.Value.Ticks))
            {
                double ms = Stopwatch.GetElapsedTime(0, ticks).TotalMilliseconds / runs;
                if (ms < total * 0.01) continue;
                Console.WriteLine(string.Create(inv, $"    {op,-32} {ms,8:F1} ms {100 * ms / total,5:F1}%  ({count / runs} steps)"));
            }
            Console.WriteLine("    slowest steps:");
            foreach (var (step, (ticks, op, shape)) in _byStep.OrderByDescending(k => k.Value.Ticks).Take(6))
                Console.WriteLine(string.Create(inv, $"      #{step,-4} {op,-32} {Stopwatch.GetElapsedTime(0, ticks).TotalMilliseconds / runs,8:F1} ms  last output {shape}"));
            if (name == "rec")
            {
                // Each batch is padded to its widest line, so the steps the network computes can exceed the lines' own.
                var perRun = _shapes.Take(_shapes.Count / runs).ToList();
                long steps = perRun.Sum(s => (long)s[0] * (s[3] / 8));
                Console.WriteLine(string.Create(inv, $"    input batches: {string.Join(' ', perRun.Select(s => $"{s[0]}x{s[3]}"))}"));
                Console.WriteLine(string.Create(inv, $"    time steps computed per run: {steps} (batch x width / 8), of which the lines' own {_contentSteps / runs} ({100.0 * _contentSteps / runs / steps:F0}%)"));
            }
        }

        public void Clear()
        {
            _byOp.Clear();
            _byStep.Clear();
            _shapes.Clear();
            _ticks = 0;
            _calls = 0;
            _contentSteps = 0;
        }
    }
}
#endif
