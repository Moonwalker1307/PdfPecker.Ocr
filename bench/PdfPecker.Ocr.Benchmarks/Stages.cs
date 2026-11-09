#if OCR_INTERNALS
using System.Diagnostics;
using System.Globalization;
using System.Text;
using PdfPecker;
using PdfPecker.Ocr;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Recognition;

namespace OcrBench;

/// <summary>
/// One A4 invoice page at 300 dpi end to end, and where its time goes: rendering the PDF page to pixels, detection,
/// orientation, recognition and layout. Run it in a fresh process per thread count, since the first call is reported
/// as the cold one (models loaded, code compiled, the engine's worker built):
///   dotnet run -c Release -p:OcrInternals=true -- stages --threads 1 [--runs 5]
/// The stage split runs the pipeline the way the engine does, with each network behind a timing wrapper, so the
/// orientation figure is the classifier's network time and the recognition figure is the rest of the recognizer
/// (crops, recognition network, decoding, words). Peak memory is the process's resident peak while reading warm
/// pages through the public API, reset after the cold call where the kernel allows it.
/// </summary>
static class Stages
{
    public static void Run(int threads, int runs)
    {
        byte[] pdf = HtmlToPdf.Convert(InvoiceHtml());
        var options = new OcrOptions { MaxDegreeOfParallelism = threads, PageMode = OcrPageMode.Force };
        double rssStart = Runner.PeakRssMb();

        // Cold: the first call of the process.
        long t0 = Stopwatch.GetTimestamp();
        OcrDocument first = PdfOcr.Recognize(pdf, options);
        double cold = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        double rssCold = Runner.PeakRssMb();
        int lines = first.Pages[0].Lines.Count;

        // Warm, through the public API: a scanned PDF page rendered and read.
        bool reset = ResetPeak();
        var warm = new List<double>();
        long alloc0 = GC.GetTotalAllocatedBytes(true);
        for (int i = 0; i < runs; i++)
        {
            long s = Stopwatch.GetTimestamp();
            PdfOcr.Recognize(pdf, options);
            warm.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }

        double allocEndToEnd = (GC.GetTotalAllocatedBytes(true) - alloc0) / (double)runs / 1048576;
        double rssWarm = Runner.PeakRssMb();
        var gc = GC.GetGCMemoryInfo();

        // Warm OCR alone on pixels already in memory, through the public API.
        var raster = PageRaster.Render(PdfRenderer.Open(pdf), 0, 300, options.MaxPixels, default);
        byte[] gray = raster.Pixels.Pixels.ToArray();
        int width = raster.Pixels.Width, height = raster.Pixels.Height;
        bool isGray = raster.Pixels.Format == PixelFormat.Gray8;
        raster.Pixels.Dispose();
        var ocrOnly = new List<double>();
        alloc0 = GC.GetTotalAllocatedBytes(true);
        for (int i = 0; i < runs; i++)
        {
            long s = Stopwatch.GetTimestamp();
            if (isGray) ImageOcr.RecognizeGray(gray, width, height, options);
            else OcrEngine.Recognize(new PixelView(gray, width, height, 3), 0, 300, options, default);
            ocrOnly.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }

        double allocOcr = (GC.GetTotalAllocatedBytes(true) - alloc0) / (double)runs / 1048576;
        long arena = PeekArena(threads);

        // The stage split, with the engine's settings in a context of the same size.
        var split = StageSplit(pdf, threads, runs);

        // Model loading alone: each model decompressed and expanded once more.
        var load = new Dictionary<string, double>();
        foreach (string name in new[] { "det", "rec", "cls" })
        {
            long s = Stopwatch.GetTimestamp();
            ModelStore.Load(name);
            load[name] = Stopwatch.GetElapsedTime(s).TotalMilliseconds;
        }

        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine($"PdfPecker.Ocr {OcrInfo.Version}, .NET {Environment.Version}, {Environment.ProcessorCount} cores, MaxDegreeOfParallelism {threads}");
        Console.WriteLine($"A4 invoice page at 300 dpi: {width} x {height} px ({(isGray ? "gray" : "RGB")}), {lines} lines, {runs} warm runs");
        Console.WriteLine();
        Console.WriteLine(string.Create(inv, $"Cold first call (PdfOcr.Recognize, includes model load, JIT, worker): {cold:F0} ms"));
        Console.WriteLine(string.Create(inv, $"Warm PdfOcr.Recognize (render + OCR):  median {Median(warm):F0} ms, min {warm.Min():F0} ms, {allocEndToEnd:F1} MB allocated per page"));
        Console.WriteLine(string.Create(inv, $"Warm OCR on pixels (RecognizeGray):     median {Median(ocrOnly):F0} ms, min {ocrOnly.Min():F0} ms, {allocOcr:F2} MB allocated per page"));
        Console.WriteLine();
        Console.WriteLine("Stage split (warm, median of runs):");
        foreach (var (stage, times) in split)
            Console.WriteLine(string.Create(inv, $"  {stage,-34} {Median(times),8:F1} ms"));
        Console.WriteLine();
        Console.WriteLine(string.Create(inv, $"Model load alone: det {load["det"]:F0} ms, rec {load["rec"]:F0} ms, cls {load["cls"]:F0} ms"));
        Console.WriteLine(string.Create(inv, $"Peak resident memory: {rssStart:F0} MB before the first call, {rssCold:F0} MB after it, {rssWarm:F0} MB peak over the warm calls{(reset ? " (peak reset after the cold call)" : " (running peak: reset refused)")}"));
        Console.WriteLine(string.Create(inv, $"Managed heap after the warm calls: {gc.HeapSizeBytes / 1048576.0:F0} MB in use, {gc.TotalCommittedBytes / 1048576.0:F0} MB committed; after the last GC, large objects {gc.GenerationInfo[3].SizeAfterBytes / 1048576.0:F0} MB and pinned ones (arenas, outputs) {gc.GenerationInfo[4].SizeAfterBytes / 1048576.0:F0} MB"));
        Console.WriteLine(string.Create(inv, $"Engine worker arena: {arena / 1048576.0:F0} MB"));
    }

    /// <summary>
    /// The memory of OCR alone: the page is rendered to gray pixels first and the renderer's garbage collected, then
    /// the resident peak is reset, so the peaks below are those of reading the page through ImageOcr.RecognizeGray,
    /// cold (models loaded, worker built) and warm. Run it in a fresh process: dotnet ... stages --memory --threads n
    /// </summary>
    public static void Memory(int threads, int runs)
    {
        byte[] pdf = HtmlToPdf.Convert(InvoiceHtml());
        var raster = PageRaster.Render(PdfRenderer.Open(pdf), 0, 300, 100_000_000, default);
        byte[] gray = raster.Pixels.Pixels.ToArray();
        int width = raster.Pixels.Width, height = raster.Pixels.Height;
        raster.Pixels.Dispose();
        pdf = [];
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        bool reset = ResetPeak();
        double before = Runner.PeakRssMb();
        var options = new OcrOptions { MaxDegreeOfParallelism = threads };
        long alloc0 = GC.GetTotalAllocatedBytes(true);
        ImageOcr.RecognizeGray(gray, width, height, options);
        double coldAlloc = (GC.GetTotalAllocatedBytes(true) - alloc0) / 1048576.0;
        double cold = Runner.PeakRssMb();
        for (int i = 0; i < runs; i++) ImageOcr.RecognizeGray(gray, width, height, options);
        double warm = Runner.PeakRssMb();
        var gc = GC.GetGCMemoryInfo();
        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine($"PdfPecker.Ocr {OcrInfo.Version}, .NET {Environment.Version}, MaxDegreeOfParallelism {threads}, GC gen0 budget {Environment.GetEnvironmentVariable("DOTNET_GCgen0size") ?? "default"}");
        Console.WriteLine(string.Create(inv, $"OCR alone on a {width} x {height} gray page (held by the caller, {gray.Length / 1048576.0:F1} MB):"));
        Console.WriteLine(string.Create(inv, $"  resident before the first call {before:F0} MB{(reset ? "" : " (peak reset refused: running peak)")}"));
        Console.WriteLine(string.Create(inv, $"  peak during the cold call {cold:F0} MB (it allocated {coldAlloc:F0} MB: models, plans, buffers)"));
        Console.WriteLine(string.Create(inv, $"  peak over {runs} warm calls {warm:F0} MB"));
        Console.WriteLine(string.Create(inv, $"  managed heap {gc.HeapSizeBytes / 1048576.0:F0} MB, committed {gc.TotalCommittedBytes / 1048576.0:F0} MB, pinned (arenas, outputs) {gc.GenerationInfo[4].SizeAfterBytes / 1048576.0:F0} MB, worker arena {PeekArena(threads) / 1048576.0:F0} MB"));
    }

    static List<(string Stage, List<double> Times)> StageSplit(byte[] pdf, int threads, int runs)
    {
        var renderer = PdfRenderer.Open(pdf);
        using var context = new InferenceContext(threads);
        // The page is gray, so the engine's detector runs on one plane (DetectionGray); the colour network is there for a
        // colour page, and both are timed.
        var det = new TimedNetwork(new ModelNetwork(ModelStore.DetectionGray, context));
        var detColour = new TimedNetwork(new ModelNetwork(ModelStore.Detection, context));
        var rec = new TimedNetwork(new ModelNetwork(ModelStore.RecognitionArgMax, context));
        var cls = new TimedNetwork(new ModelNetwork(ModelStore.Classifier, context));
        var options = new OcrOptions { MaxDegreeOfParallelism = threads };
        var pipeline = new OcrPipeline(detColour, rec, cls, OcrEngine.Characters, OcrEngine.DetectionOptions, OcrEngine.RecognitionOptions(options), det);
        string[] names = ["render to pixels (PdfRenderer)", "detection", "  of which the network", "orientation (classifier network)", "recognition (crops, network, CTC)", "  of which the network", "layout (reading order, OcrPage)", "total"];
        var times = names.Select(n => (n, new List<double>())).ToList();
        for (int i = 0; i <= runs; i++)
        {
            det.Reset();
            detColour.Reset();
            rec.Reset();
            cls.Reset();
            long t = Stopwatch.GetTimestamp();
            var raster = PageRaster.Render(renderer, 0, 300, options.MaxPixels, default);
            double render = Lap(ref t);
            var view = new PixelView(raster.Pixels.Pixels, raster.Pixels.Width, raster.Pixels.Height, raster.Pixels.Channels);
            List<DetectedQuad> quads = pipeline.Detector.Detect(view);
            double detection = Lap(ref t);
            List<RecognizedLine> recognized = pipeline.Recognizer.Recognize(view, quads);
            double recognition = Lap(ref t);
            PageResult result = PageAssembler.Assemble(view.Width, view.Height, raster.Dpi, recognized);
            OcrEngine.ToPage(result, 0);
            double layout = Lap(ref t);
            raster.Pixels.Dispose();
            if (i == 0) continue; // the first pass warms this context's plans and buffers
            double[] values = [render, detection, det.Milliseconds + detColour.Milliseconds, cls.Milliseconds, recognition - cls.Milliseconds, rec.Milliseconds, layout, render + detection + recognition + layout];
            for (int k = 0; k < values.Length; k++) times[k].Item2.Add(values[k]);
        }

        return times;
    }

    static double Lap(ref long t)
    {
        long now = Stopwatch.GetTimestamp();
        double ms = Stopwatch.GetElapsedTime(t, now).TotalMilliseconds;
        t = now;
        return ms;
    }

    static long PeekArena(int threads)
    {
        EngineWorker worker = EnginePool.Rent(threads);
        long bytes = worker.ArenaBytes;
        EnginePool.Return(worker);
        return bytes;
    }

    static double Median(List<double> values)
    {
        var sorted = values.Order().ToList();
        return sorted.Count % 2 == 1 ? sorted[sorted.Count / 2] : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2;
    }

    /// <summary>Resets the kernel's resident peak (VmHWM) of this process, so the next reading is the peak from here on.</summary>
    static bool ResetPeak()
    {
        try
        {
            File.WriteAllText("/proc/self/clear_refs", "5");
            return true;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>An invoice of about forty text lines and table cells on an A4 page.</summary>
    public static string InvoiceHtml()
    {
        var sb = new StringBuilder();
        sb.Append("<!doctype html><html><head><meta charset=\"utf-8\"><style>@page{size:A4;margin:18mm}body{font:10.5pt/1.45 sans-serif}")
            .Append("h1{font-size:20pt;margin:0 0 6mm}table{border-collapse:collapse;width:100%;margin:5mm 0}td,th{border:1px solid #999;padding:3px 6px;text-align:left}")
            .Append("td.n,th.n{text-align:right}.cols{display:flex;gap:20mm}</style></head><body>")
            .Append("<h1>Invoice 2026-0412</h1><div class=\"cols\"><p>Example Supplies Ltd.<br>12 Harbour Road<br>8000 Aarhus, Denmark<br>VAT DK 12 34 56 78</p>")
            .Append("<p>Bill to: Example Customer Ltd.<br>Storgata 12<br>0155 Oslo, Norway<br>Customer no. 55-1093</p></div>")
            .Append("<p>Invoice date: 8 November 2026<br>Due date: 8 December 2026<br>Order reference: PO-77812</p>")
            .Append("<table><tr><th>Item</th><th class=\"n\">Qty</th><th class=\"n\">Unit price</th><th class=\"n\">Amount</th></tr>");
        string[] items = ["Consulting, October", "Travel expenses", "Software licence", "Support hours", "Training day", "Hardware: docking station", "Shipping and handling", "Extended warranty", "Cloud storage, 1 TB", "Printer toner, black"];
        for (int i = 0; i < items.Length; i++)
        {
            int qty = 1 + (i * 7 % 13);
            decimal price = 49.5m + (i * 137.25m);
            sb.Append(CultureInfo.InvariantCulture, $"<tr><td>{items[i]}</td><td class=\"n\">{qty}</td><td class=\"n\">{price:N2}</td><td class=\"n\">{qty * price:N2}</td></tr>");
        }

        sb.Append("</table><p>Subtotal EUR 28,093.75<br>VAT 25% EUR 7,023.44<br><b>Total EUR 35,117.19</b></p>")
            .Append("<p>Please pay within 30 days to IBAN DK50 0040 0440 1162 43, BIC NDEADKKK, quoting the invoice number.<br>")
            .Append("Late payments carry interest of 0.8% a month. Questions about this invoice: invoices@example.com.<br>")
            .Append("Thank you for your order. Rødgrød med fløde, Größe 42, smørrebrød og æblekage.</p></body></html>");
        return sb.ToString();
    }

    /// <summary>A network that adds up the time of its runs.</summary>
    sealed class TimedNetwork(INetwork inner) : INetwork
    {
        long _ticks;

        public double Milliseconds => Stopwatch.GetElapsedTime(0, _ticks).TotalMilliseconds;

        public void Reset() => _ticks = 0;

        public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
        {
            long s = Stopwatch.GetTimestamp();
            var output = inner.Run(input, shape, out outputShape);
            _ticks += Stopwatch.GetTimestamp() - s;
            return output;
        }
    }
}
#endif
