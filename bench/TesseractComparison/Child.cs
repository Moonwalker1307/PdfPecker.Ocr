using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace TesseractComparison;

/// <summary>Everything the parent tells a measuring child process.</summary>
sealed class ChildOptions
{
    public string Engine = "tesseract";
    public int Threads = 1;
    public string CorpusDir = "";
    public string OutFile = "";

    /// <summary>Where each page's recognised text is written for inspection (one folder per configuration).</summary>
    public string DumpDir = "";
    public int Runs = 5;
    public int Warmup = 1;
    public string[]? Only;
    public string TesseractLib = "";
    public string Tessdata = "";
    public string TessLang = "auto";
    public string TessPsm = "auto";
    public string Python = "python3";
    public string RapidScript = "";
}

/// <summary>
/// Measures one engine at one thread count on the corpus, in this process: the process is fresh, so its peak memory is the
/// engine's own, and it exits when done. Timing is the median of N runs after warm-up runs; peak memory is the process's
/// resident peak since the start of the timed runs of that page (so it includes the loaded models, which is what the engine
/// needs resident); CPU time is process-wide, so it counts every thread.
/// </summary>
static class Child
{
    public static readonly string[] Header =
    [
        "engine", "threads", "page", "category", "lang", "dpi", "degradation", "rotation", "width", "height", "channels", "runs",
        "wall_ms_median", "wall_ms_min", "wall_ms_max", "cpu_ms_median", "peak_rss_mb", "alloc_mb",
        "ref_chars", "hyp_chars", "char_edits", "cer", "ref_words", "hyp_words", "word_edits", "wer", "bag_f1",
        "gt_boxes", "hyp_boxes", "box_recall_iou50", "box_precision_iou50", "mean_iou", "mean_conf", "error",
    ];

    public static int Run(ChildOptions o)
    {
        using var engine = Create(o);
        var pages = CorpusPage.Load(o.CorpusDir, o.Only);
        var init = Stopwatch.StartNew();
        engine.Initialize(o.Threads);
        double initMs = init.Elapsed.TotalMilliseconds;
        bool canReset = ProcMem.ResetPeak();
        var rows = new List<string>();
        int firstPrepareDone = 0;
        double firstPrepareMs = 0;
        long rssAfterInit = 0;
        foreach (var page in pages)
        {
            string error = "";
            OcrResult? last = null;
            var wall = new List<double>();
            var cpu = new List<double>();
            long peakKb = 0, allocBytes = 0;
            try
            {
                var prep = Stopwatch.StartNew();
                engine.Prepare(page);
                if (firstPrepareDone++ == 0) { firstPrepareMs = prep.Elapsed.TotalMilliseconds; rssAfterInit = ProcMem.RssKb(); }
                var (pixels, w, h, c) = PngDecoder.Decode(File.ReadAllBytes(page.ImagePath));
                var format = c == 1 ? PixelFormat.Gray8 : PixelFormat.Rgb24;
                for (int i = 0; i < o.Warmup; i++) engine.Recognize(pixels, w, h, format);
                if (canReset) ProcMem.ResetPeak();
                engine.BeginMeasurement();
                long alloc0 = GC.GetTotalAllocatedBytes(true);
                for (int i = 0; i < o.Runs; i++)
                {
                    var cpu0 = Environment.CpuUsage.TotalTime;
                    long t0 = Stopwatch.GetTimestamp();
                    last = engine.Recognize(pixels, w, h, format);
                    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    double cpuMs = last.ExternalCpuMs ?? (Environment.CpuUsage.TotalTime - cpu0).TotalMilliseconds;
                    wall.Add(ms);
                    cpu.Add(cpuMs);
                }
                allocBytes = (GC.GetTotalAllocatedBytes(true) - alloc0) / o.Runs;
                peakKb = last!.ExternalPeakKb ?? ProcMem.PeakKb();
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                error = e.Message.Replace('\t', ' ').Replace('\n', ' ');
            }
            if (last is not null && o.DumpDir.Length > 0)
            {
                Directory.CreateDirectory(o.DumpDir);
                File.WriteAllText(Path.Combine(o.DumpDir, page.Id + ".txt"), last.Text, new UTF8Encoding(false));
            }
            rows.Add(Row(o, engine, page, wall, cpu, peakKb, allocBytes, last, error));
            Console.Error.WriteLine($"{engine.Name} t={o.Threads} {page.Id}: {(error.Length > 0 ? "ERROR " + error : $"{Median(wall):F0} ms")}");
        }
        var all = new List<string> { string.Join(',', Header) };
        all.AddRange(rows);
        File.WriteAllLines(o.OutFile, all, new UTF8Encoding(false));
        File.WriteAllLines(o.OutFile + ".meta", new[]
        {
            $"engine\t{engine.Name}", $"threads\t{o.Threads}", $"description\t{engine.Description}",
            $"init_ms\t{initMs:F0}", $"first_prepare_ms\t{firstPrepareMs:F0}", $"rss_after_load_mb\t{(engine.InProcess ? (rssAfterInit / 1024.0).ToString("F0", CultureInfo.InvariantCulture) : "n/a")}",
            $"process_peak_mb\t{ProcMem.PeakKb() / 1024.0:F0}", $"peak_reset\t{canReset}", $"runs\t{o.Runs}", $"warmup\t{o.Warmup}",
        });
        return 0;
    }

    static IOcrEngineAdapter Create(ChildOptions o)
    {
        switch (o.Engine)
        {
            case "tesseract":
                TesseractNative.SetLibraryPath(o.TesseractLib);
                return new TesseractAdapter(o.Tessdata, o.TessLang, o.TessPsm);
            case "rapidocr":
                return new RapidOcrAdapter(o.Python, o.RapidScript);
            case "pdfpecker-ocr":
                return new PdfPeckerOcrAdapter();
            default:
                throw new ArgumentException($"unknown engine {o.Engine}");
        }
    }

    static string Row(ChildOptions o, IOcrEngineAdapter engine, CorpusPage page, List<double> wall, List<double> cpu, long peakKb, long allocBytes,
        OcrResult? result, string error)
    {
        string F(double v, string fmt = "0.####") => double.IsNaN(v) ? "" : v.ToString(fmt, CultureInfo.InvariantCulture);
        var f = new List<string>
        {
            engine.Name, o.Threads.ToString(CultureInfo.InvariantCulture), page.Id, page.Category, page.Lang, page.Dpi.ToString(CultureInfo.InvariantCulture),
            page.Degradation, page.Rotation.ToString(CultureInfo.InvariantCulture), page.Width.ToString(CultureInfo.InvariantCulture),
            page.Height.ToString(CultureInfo.InvariantCulture), page.Channels.ToString(CultureInfo.InvariantCulture), wall.Count.ToString(CultureInfo.InvariantCulture),
        };
        if (result is null || wall.Count == 0)
        {
            f.AddRange(Enumerable.Repeat("", Header.Length - f.Count - 1));
            f.Add(Csv(error));
            return string.Join(',', f);
        }
        string reference = Metrics.Normalize(string.Join('\n', page.LoadLines()));
        string hypothesis = Metrics.Normalize(result.Text);
        int charEdits = Metrics.Distance<char>(reference, hypothesis);
        var refWords = Metrics.Words(reference);
        var hypWords = Metrics.Words(hypothesis);
        int wordEdits = Metrics.Distance<string>(refWords, hypWords);
        var (recall, precision, meanIou) = Metrics.MatchBoxes(page.LoadBoxes(), result.Words);
        double meanConf = result.Words.Count == 0 ? double.NaN : result.Words.Average(w => w.Confidence);
        wall.Sort();
        cpu.Sort();
        f.Add(F(Median(wall), "0.##")); f.Add(F(wall[0], "0.##")); f.Add(F(wall[^1], "0.##")); f.Add(F(Median(cpu), "0.##"));
        f.Add(F(peakKb / 1024.0, "0.#")); f.Add(engine.Managed ? F(allocBytes / 1048576.0, "0.##") : "");
        f.Add(reference.Length.ToString(CultureInfo.InvariantCulture)); f.Add(hypothesis.Length.ToString(CultureInfo.InvariantCulture)); f.Add(charEdits.ToString(CultureInfo.InvariantCulture));
        f.Add(F(charEdits / (double)Math.Max(1, reference.Length)));
        f.Add(refWords.Length.ToString(CultureInfo.InvariantCulture)); f.Add(hypWords.Length.ToString(CultureInfo.InvariantCulture)); f.Add(wordEdits.ToString(CultureInfo.InvariantCulture));
        f.Add(F(wordEdits / (double)Math.Max(1, refWords.Length)));
        f.Add(F(Metrics.BagF1(refWords, hypWords)));
        f.Add(page.LoadBoxes().Count.ToString(CultureInfo.InvariantCulture)); f.Add(result.Words.Count.ToString(CultureInfo.InvariantCulture));
        f.Add(F(recall)); f.Add(F(precision)); f.Add(F(meanIou)); f.Add(F(meanConf));
        f.Add(Csv(error));
        return string.Join(',', f);
    }

    static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;

    static double Median(List<double> sorted)
    {
        if (sorted.Count == 0) return double.NaN;
        var copy = sorted.OrderBy(v => v).ToList();
        return copy.Count % 2 == 1 ? copy[copy.Count / 2] : (copy[copy.Count / 2 - 1] + copy[copy.Count / 2]) / 2;
    }
}
