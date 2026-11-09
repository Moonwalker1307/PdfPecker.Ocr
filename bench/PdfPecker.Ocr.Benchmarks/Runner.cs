using System.Diagnostics;
using System.Globalization;
using System.Text.Json;

namespace OcrBench;

sealed record CaseResult(string Name, string? Pending, int Runs, double FirstMs, double MedianMs, double MeanMs, double MinMs, double P95Ms,
    double AllocMb, double CpuMs, double MegapixelsPerSecond);

sealed record RunResult(string Version, int Cores, double PeakRssMb, List<CaseResult> Cases);

static class Runner
{
    public static RunResult RunAll(List<BenchCase> cases, bool quiet)
    {
        var results = new List<CaseResult>();
        foreach (var c in cases)
        {
            if (c.Run is null)
            {
                results.Add(new CaseResult(c.Name, c.Pending ?? "pending", 0, 0, 0, 0, 0, 0, 0, 0, 0));
                continue;
            }
            if (!quiet) Console.Error.WriteLine($"running {c.Name}...");
            results.Add(Measure(c));
        }
        return new RunResult(PdfPecker.Ocr.OcrInfo.Version, Environment.ProcessorCount, PeakRssMb(), results);
    }

    /// <summary>
    /// The first run is timed on its own (it includes model loading and JIT), a second run warms up, and then runs are timed
    /// until the case has its minimum number and its time budget is used up, at most 200. CPU time counts every thread of the
    /// process; allocation is the process-wide managed allocation per run, which includes the worker threads' share.
    /// </summary>
    static CaseResult Measure(BenchCase c)
    {
        var run = c.Run!;
        long t0 = Stopwatch.GetTimestamp();
        run();
        double first = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        run();
        GC.Collect();
        var times = new List<double>();
        long alloc0 = GC.GetTotalAllocatedBytes(true);
        var cpu0 = Environment.CpuUsage.TotalTime;
        var budget = Stopwatch.StartNew();
        while (times.Count < c.MinRuns || (times.Count < 200 && budget.Elapsed.TotalSeconds < c.BudgetSeconds))
        {
            long s = Stopwatch.GetTimestamp();
            run();
            times.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }
        double cpu = (Environment.CpuUsage.TotalTime - cpu0).TotalMilliseconds / times.Count;
        double alloc = (GC.GetTotalAllocatedBytes(true) - alloc0) / (double)times.Count / 1048576.0;
        times.Sort();
        double median = times.Count % 2 == 1 ? times[times.Count / 2] : (times[times.Count / 2 - 1] + times[times.Count / 2]) / 2;
        double mean = times.Average();
        double mpix = c.Pixels > 0 ? c.Pixels / 1e6 / (median / 1000) : 0;
        return new CaseResult(c.Name, null, times.Count, first, median, mean, times[0], times[(int)(times.Count * 0.95)], alloc, cpu, mpix);
    }

    public static double PeakRssMb()
    {
        try
        {
            foreach (var line in File.ReadLines("/proc/self/status"))
                if (line.StartsWith("VmHWM:", StringComparison.Ordinal))
                    return long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1], CultureInfo.InvariantCulture) / 1024.0;
        }
        catch (IOException) { }
        return Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0;
    }

    public static string Ms(double ms) => ms >= 100 ? $"{ms:F0} ms" : ms >= 10 ? $"{ms:F1} ms" : $"{ms:F2} ms";

    public static void Print(RunResult r)
    {
        Console.WriteLine($"PdfPecker.Ocr {r.Version} benchmark, {r.Cores} cores, .NET {Environment.Version}");
        Console.WriteLine();
        Console.WriteLine($"{"Case",-46} {"Runs",5} {"First",10} {"Median",10} {"Mean",10} {"P95",10} {"Alloc/run",11} {"CPU/run",10} {"MPix/s",8}");
        foreach (var c in r.Cases)
        {
            if (c.Pending is not null) { Console.WriteLine($"{c.Name,-46} pending: {c.Pending}"); continue; }
            Console.WriteLine($"{c.Name,-46} {c.Runs,5} {Ms(c.FirstMs),10} {Ms(c.MedianMs),10} {Ms(c.MeanMs),10} {Ms(c.P95Ms),10} {c.AllocMb,9:F1}MB {Ms(c.CpuMs),10} {(c.MegapixelsPerSecond > 0 ? c.MegapixelsPerSecond.ToString("F2", CultureInfo.InvariantCulture) : ""),8}");
        }
        Console.WriteLine();
        Console.WriteLine($"Peak resident memory of the process: {r.PeakRssMb:F0} MB");
    }
}

/// <summary>
/// Compares two builds of this program, each run in a fresh process in turns, so that a slow patch on a shared VM hits both.
/// A change is judged only against its base version on one machine: cloud VMs differ too much for a number from one to say
/// anything about another.
/// </summary>
static class Versions
{
    const string GcBudget = "0x1000000";

    public static void Compare(string otherAssembly, int rounds)
    {
        var builds = new[] { ("other", otherAssembly), ("this", typeof(Versions).Assembly.Location) };
        var runs = builds.ToDictionary(b => b.Item1, _ => new List<RunResult>());
        for (int round = 1; round <= rounds; round++)
            foreach (var (label, assembly) in builds)
            {
                Console.Error.WriteLine($"Round {round}/{rounds}: {label} build...");
                runs[label].Add(RunInChild(assembly) ?? throw new InvalidOperationException($"The {label} build failed."));
            }

        Console.WriteLine($"{Environment.ProcessorCount} cores, {CpuName()}, .NET {Environment.Version}, GC gen0 budget {Environment.GetEnvironmentVariable("DOTNET_GCgen0size") ?? GcBudget}");
        Console.WriteLine($"Range over {rounds} rounds; other = {otherAssembly}");
        Console.WriteLine();
        Console.WriteLine("| | Other build | This build |");
        Console.WriteLine("| --- | --- | --- |");
        var names = runs["this"].SelectMany(r => r.Cases).Concat(runs["other"].SelectMany(r => r.Cases)).Where(c => c.Pending is null).Select(c => c.Name).Distinct().ToList();
        foreach (var name in names)
        {
            Row($"{name}: median", r => Pick(r, name)?.MedianMs, Runner.Ms);
            Row($"{name}: allocation per run", r => Pick(r, name)?.AllocMb, v => $"{v:F1} MB");
            Row($"{name}: CPU per run", r => Pick(r, name)?.CpuMs, Runner.Ms);
        }
        Row("Peak resident memory", r => r.PeakRssMb, v => $"{v:F0} MB");

        static CaseResult? Pick(RunResult r, string name)
        {
            var c = r.Cases.FirstOrDefault(x => x.Name == name);
            return c is null || c.Pending is not null ? null : c;
        }

        void Row(string label, Func<RunResult, double?> value, Func<double, string> format) =>
            Console.WriteLine($"| {label} | {Range(runs["other"], value, format)} | {Range(runs["this"], value, format)} |");
    }

    static string Range(List<RunResult> results, Func<RunResult, double?> value, Func<double, string> format)
    {
        var values = results.Select(value).Where(v => v.HasValue).Select(v => v!.Value).ToList();
        if (values.Count == 0) return "n/a";
        double min = values.Min(), max = values.Max();
        return format(min) == format(max) ? format(min) : $"{format(min)} – {format(max)}";
    }

    static RunResult? RunInChild(string assembly)
    {
        var psi = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true };
        psi.ArgumentList.Add(assembly);
        psi.ArgumentList.Add("--json");
        // A fixed gen0 budget keeps peak memory comparable between machines with different caches, as in the library's harness.
        if (Environment.GetEnvironmentVariable("DOTNET_GCgen0size") == null) psi.Environment["DOTNET_GCgen0size"] = GcBudget;
        using var p = Process.Start(psi)!;
        var output = p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        var line = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).LastOrDefault(l => l.StartsWith('{'));
        return p.ExitCode == 0 && line != null ? JsonSerializer.Deserialize<RunResult>(line) : null;
    }

    static string CpuName()
    {
        try { return File.ReadLines("/proc/cpuinfo").First(l => l.StartsWith("model name", StringComparison.Ordinal)).Split(':')[1].Trim(); }
        catch (Exception e) when (e is IOException or InvalidOperationException) { return "unknown CPU"; }
    }
}
