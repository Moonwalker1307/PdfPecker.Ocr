using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace PdfPecker.Ocr.Fuzz;

internal sealed class Options
{
    public string Target = "all";
    public TimeSpan Duration = TimeSpan.FromMinutes(1);
    public string Out = "fuzz-findings";
    public long Seed = 1;
    public TimeSpan Timeout = TimeSpan.FromSeconds(10);
    public long MaxMemoryMb = 1536;
    public string? Repo;
    public bool NoMinimize;
    public bool Worker;
    public string? Repro;
    public bool List;
    public int SkipSeeds;
    public int Jobs = 1;
    public int Lane;

    public static Options Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException("Missing value for " + args[i]);
            switch (args[i])
            {
                case "--target": o.Target = Value(); break;
                case "--duration": o.Duration = ParseDuration(Value()); break;
                case "--out": o.Out = Value(); break;
                case "--seed": o.Seed = long.Parse(Value(), CultureInfo.InvariantCulture); break;
                case "--timeout": o.Timeout = TimeSpan.FromSeconds(double.Parse(Value(), CultureInfo.InvariantCulture)); break;
                case "--max-memory-mb": o.MaxMemoryMb = long.Parse(Value(), CultureInfo.InvariantCulture); break;
                case "--repo": o.Repo = Value(); break;
                case "--no-minimize": o.NoMinimize = true; break;
                case "--worker": o.Worker = true; break;
                case "--repro": o.Repro = Value(); break;
                case "--list": o.List = true; break;
                case "--skip-seeds": o.SkipSeeds = int.Parse(Value(), CultureInfo.InvariantCulture); break;
                case "--jobs": o.Jobs = Math.Max(1, int.Parse(Value(), CultureInfo.InvariantCulture)); break;
                case "--lane": o.Lane = int.Parse(Value(), CultureInfo.InvariantCulture); break;
                default: throw new ArgumentException("Unknown option " + args[i]);
            }
        }
        return o;
    }

    /// <summary>Seconds, or a number with an s, m or h suffix.</summary>
    public static TimeSpan ParseDuration(string s)
    {
        var m = Regex.Match(s.Trim(), @"^(\d+(?:\.\d+)?)([smh]?)$");
        if (!m.Success) throw new ArgumentException("Bad duration " + s);
        double n = double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        return m.Groups[2].Value switch { "m" => TimeSpan.FromMinutes(n), "h" => TimeSpan.FromHours(n), _ => TimeSpan.FromSeconds(n) };
    }

    private static string Invariant(long n) => n.ToString(CultureInfo.InvariantCulture);

    public string TimeoutText => Timeout.TotalSeconds.ToString(CultureInfo.InvariantCulture);

    public string[] ToWorkerArgs(string target, TimeSpan duration, long seed, bool worker, int skipSeeds = 0, int lane = 0)
    {
        var a = new List<string>
        {
            "--target", target, "--duration", Invariant((long)Math.Max(1, duration.TotalSeconds)) + "s", "--out", Out, "--seed", Invariant(seed),
            "--timeout", TimeoutText, "--max-memory-mb", Invariant(MaxMemoryMb),
        };
        if (Repo != null) { a.Add("--repo"); a.Add(Repo); }
        if (NoMinimize) a.Add("--no-minimize");
        if (worker) a.Add("--worker");
        if (skipSeeds > 0) { a.Add("--skip-seeds"); a.Add(Invariant(skipSeeds)); }
        if (Jobs > 1) { a.Add("--jobs"); a.Add(Invariant(Jobs)); a.Add("--lane"); a.Add(Invariant(lane)); }
        return a.ToArray();
    }
}

/// <summary>Exit codes shared by the supervisor, its worker processes and --repro runs.</summary>
internal static class Exit
{
    public const int Done = 0;
    public const int Findings = 1;
    public const int Usage = 2;
    public const int Restart = 3; // the worker had to abandon a thread that ignored cancellation
    public const int ReproFails = 10;
}

internal static class Program
{
    private static int Main(string[] args)
    {
        Options o;
        try { o = Options.Parse(args); }
        catch (Exception ex) when (ex is ArgumentException or FormatException or OverflowException)
        {
            Console.Error.WriteLine(ex.Message);
            Console.Error.WriteLine("Usage: PdfPecker.Ocr.Fuzz --target <name|all> [--duration 2m] [--out dir] [--seed 1] [--timeout 10] [--max-memory-mb 1536] [--no-minimize] [--repo path] [--jobs n] | --list | --repro <file> --target <name>");
            return Exit.Usage;
        }
        if (o.List)
        {
            foreach (var t in Targets.All) Console.WriteLine(t.Name);
            return Exit.Done;
        }
        if (o.Repro != null) return Repro(o);
        if (o.Worker) return new Worker(o).Run();
        return Supervise(o);
    }

    private static List<Target> Select(Options o)
    {
        if (o.Target == "all") return Targets.All.ToList();
        var t = Targets.Find(o.Target);
        if (t == null)
        {
            Console.Error.WriteLine("Unknown target " + o.Target + ". Targets: " + string.Join(", ", Targets.All.Select(x => x.Name)));
            return new List<Target>();
        }
        return new List<Target> { t };
    }

    /// <summary>Runs one file through a target once, for reproducing and for minimising crashes from outside the process.</summary>
    private static int Repro(Options o)
    {
        var target = Targets.Find(o.Target);
        if (target == null)
        {
            Console.Error.WriteLine("Unknown target " + o.Target);
            return Exit.Usage;
        }
        var executor = new Executor(target.TimeoutFor(o.Timeout), o.MaxMemoryMb << 20);
        var verdict = executor.Run(target, File.ReadAllBytes(o.Repro!), out _);
        if (verdict != null) Console.WriteLine($"{verdict.Kind}: {verdict.Signature}\n{verdict.Detail}");
        return executor.AbandonedThreads > 0 ? Exit.Restart : verdict != null ? Exit.ReproFails : Exit.Done;
    }

    /// <summary>
    /// Runs each target in a child process. A crash the process cannot survive (stack overflow, fatal error) kills only
    /// the child; the supervisor saves the input the child had written down and starts a new child for the rest of the time.
    /// With --jobs, each target runs that many children side by side, each with its own random seed and its own share of
    /// the seed inputs, so a machine with several cores tries several times as many inputs in the same time.
    /// </summary>
    private static int Supervise(Options o)
    {
        var targets = Select(o);
        if (targets.Count == 0) return Exit.Usage;
        int total = 0;
        foreach (var target in targets)
        {
            var end = DateTime.UtcNow + o.Duration;
            var codes = new int[o.Jobs];
            var lanes = Enumerable.Range(0, o.Jobs).Select(lane => new Thread(() => codes[lane] = RunLane(o, target, lane, end))).ToArray();
            foreach (var t in lanes) t.Start();
            foreach (var t in lanes) t.Join();
            if (codes.Contains(Exit.Usage)) return Exit.Usage;
            var store = new FindingStore(o.Out, target.Name);
            int count = store.Count;
            Console.WriteLine($"[{target.Name}] done: {count} finding(s) in {store.Directory}");
            total += count;
        }
        return total > 0 ? Exit.Findings : Exit.Done;
    }

    /// <summary>Runs and restarts the children of one lane until the time is up.</summary>
    private static int RunLane(Options o, Target target, int lane, DateTime end)
    {
        var store = new FindingStore(o.Out, target.Name, lane);
        int restarts = 0;
        store.SeedsDone = 0;
        while (DateTime.UtcNow < end)
        {
            var remaining = end - DateTime.UtcNow;
            // A restarted worker skips the seeds it already ran, so a seed that hangs is not run again on every restart.
            var seed = o.Seed + lane * 7_919L + restarts * 1_000_003L;
            var (code, stderr) = RunChild(o.ToWorkerArgs(target.Name, remaining, seed, worker: true, store.SeedsDone, lane));
            if (code == Exit.Done) break;
            restarts++;
            if (code == Exit.Restart) continue;
            if (code == Exit.Usage)
            {
                Console.Error.Write(stderr);
                return Exit.Usage;
            }
            CrashFound(o, target, store, code, stderr);
        }
        if (File.Exists(store.CurrentInputPath)) File.Delete(store.CurrentInputPath);
        store.SeedsDone = 0;
        return Exit.Done;
    }

    private static (int Code, string Stderr) RunChild(string[] args)
    {
        var psi = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false, RedirectStandardError = true };
        if (Path.GetFileNameWithoutExtension(Environment.ProcessPath) == "dotnet") psi.ArgumentList.Add(typeof(Program).Assembly.Location);
        foreach (var a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        var err = new StringBuilder();
        int lines = 0;
        p.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null || lines > 200) return;
            lines++;
            lock (err) err.AppendLine(e.Data);
        };
        p.BeginErrorReadLine();
        p.WaitForExit();
        p.WaitForExit(); // flushes the asynchronous error handler
        lock (err) return (p.ExitCode, err.ToString());
    }

    private static void CrashFound(Options o, Target target, FindingStore store, int code, string stderr)
    {
        var frame = Regex.Match(stderr, @"at (PdfPecker\.(?!Ocr\.Fuzz)[^\r\n(]*)");
        var signature = "crash|exit " + code.ToString(CultureInfo.InvariantCulture) + "|" + (frame.Success ? frame.Groups[1].Value : "no-library-frame");
        if (store.Repeat(signature)) { Console.WriteLine($"[{target.Name}] crash (exit {code}) repeated"); return; }
        if (!File.Exists(store.CurrentInputPath))
        {
            Console.WriteLine($"[{target.Name}] the worker died with exit code {code} before running an input:\n{stderr}");
            return;
        }
        var input = File.ReadAllBytes(store.CurrentInputPath);
        string[] ReproArgs(string file) => new[] { "--repro", file, "--target", target.Name, "--timeout", o.TimeoutText, "--max-memory-mb", o.MaxMemoryMb.ToString(CultureInfo.InvariantCulture) };
        // A crash that does not happen again with the input alone came from state built up over many inputs, such as a
        // heap that filled up; there is nothing to minimise or to put in a regression test.
        if (RunChild(ReproArgs(store.CurrentInputPath)).Code != code)
        {
            Console.WriteLine($"[{target.Name}] crash (exit {code}) did not reproduce with the input alone; not saved");
            return;
        }
        var minimised = input;
        if (!o.NoMinimize)
        {
            var scratch = Path.Combine(store.Directory, ".trial");
            minimised = Minimizer.Minimize(input, target.Text, candidate =>
            {
                File.WriteAllBytes(scratch, candidate);
                var (trialCode, _) = RunChild(ReproArgs(scratch));
                return trialCode == code;
            }, TimeSpan.FromSeconds(120), maxTrials: 150);
            File.Delete(scratch);
        }
        var detail = $"The process died with exit code {code} (a stack overflow or another fatal error that cannot be caught).\nStandard error:\n{stderr}";
        var name = store.Save(target, new Verdict("crash", signature, detail), minimised, input.Length, o.Seed, 0);
        Console.WriteLine($"[{target.Name}] FINDING crash exit {code}: {name} ({minimised.Length} bytes)");
    }
}

/// <summary>The in-process fuzzing loop of one target: seeds first, then mutants until the time is up.</summary>
internal sealed class Worker
{
    private readonly Options _o;

    public Worker(Options o) => _o = o;

    public int Run()
    {
        var target = Targets.Find(_o.Target);
        if (target == null) { Console.Error.WriteLine("Unknown target " + _o.Target); return Exit.Usage; }
        var root = SeedContext.FindRoot(_o.Repo);
        var seeds = target.Seeds(new SeedContext(root)).Where(s => s.Data.Length <= target.MaxSize).ToList();
        if (seeds.Count == 0) { Console.Error.WriteLine($"[{target.Name}] no seed inputs found under {root}"); return Exit.Usage; }

        var store = new FindingStore(_o.Out, target.Name, _o.Lane);
        var executor = new Executor(target.TimeoutFor(_o.Timeout), _o.MaxMemoryMb << 20);
        long startRetained = executor.Retained;
        bool leakReported = false, shed = false;
        var rng = new Random(unchecked((int)_o.Seed));
        var mutator = new Mutator(rng, target.MaxSize, target.Dictionary, target.Nesters);
        var corpus = seeds.Select(s => s.Data).ToList();
        var survivors = new List<byte[]>();
        var tolerated = new SortedDictionary<string, long>(StringComparer.Ordinal);
        long executions = 0, duplicates = 0, flaky = 0;
        var clock = Stopwatch.StartNew();
        var tag = _o.Jobs > 1 ? target.Name + "#" + (_o.Lane + 1).ToString(CultureInfo.InvariantCulture) : target.Name;
        var nextReport = TimeSpan.FromSeconds(20);
        Console.WriteLine($"[{tag}] {seeds.Count} seeds, seed {_o.Seed}, {_o.Duration.TotalSeconds:0}s");

        // Returns true when the input ran without any exception, which makes the mutant worth mutating further.
        bool Execute(byte[] input, string? label)
        {
            File.WriteAllBytes(store.CurrentInputPath, input); // survives a crash that takes the process down
            executions++;
            var verdict = executor.Run(target, input, out var tol);
            if (tol != null) { tolerated[tol] = tolerated.GetValueOrDefault(tol) + 1; return false; }
            if (verdict == null) return true;
            if (store.Repeat(verdict.Signature)) { duplicates++; return false; }

            // A slow or memory-hungry input is often just a victim of a busy machine or of state left by earlier inputs, so
            // it only counts when it fails again on its own.
            if (verdict.Kind is "timeout" or "memory" && executor.AbandonedThreads == 0)
            {
                // The retry measures the input against the heap as it is now, so a cache that earlier inputs filled
                // does not count; heap that keeps growing between inputs is a leak, reported on its own.
                long now = executor.MeasureRetained();
                if (now - startRetained > (_o.MaxMemoryMb << 20) / 4 && !leakReported)
                {
                    leakReported = true;
                    Console.WriteLine($"[{tag}] LEAK: the heap kept between inputs grew from {startRetained >> 20} MB to {now >> 20} MB");
                }
                // Once the leak alone is past the ceiling, the hard heap limit is close and every input would fail;
                // a fresh worker sheds it.
                if (now - startRetained > _o.MaxMemoryMb << 20)
                {
                    flaky++;
                    shed = true;
                    return false;
                }
                if (executor.Run(target, input, out _) is not { Kind: var repeatKind } || repeatKind != verdict.Kind)
                {
                    flaky++;
                    return false;
                }
            }

            var minimised = input;
            if (!_o.NoMinimize && executor.AbandonedThreads == 0)
            {
                bool Same(Verdict v) => v.Kind == verdict.Kind && (v.Kind != "exception" || v.Signature == verdict.Signature);
                minimised = Minimizer.Minimize(input, target.Text,
                    candidate => executor.AbandonedThreads == 0 && executor.Run(target, candidate, out _) is { } v && Same(v),
                    TimeSpan.FromSeconds(90));
            }
            var final = verdict;
            if (minimised.Length != input.Length && executor.AbandonedThreads == 0 && executor.Run(target, minimised, out _) is { } again && again.Kind == verdict.Kind)
                final = again with { Signature = verdict.Signature };
            var name = store.Save(target, final, minimised, input.Length, _o.Seed, executions);
            Console.WriteLine($"[{tag}] FINDING {verdict.Kind}: {name} ({minimised.Length} bytes{(label != null ? ", from seed " + label : "")}) {verdict.Signature}");
            return false;
        }

        for (int s = _o.SkipSeeds; s < seeds.Count; s++)
        {
            if (clock.Elapsed > _o.Duration) break;
            if (s % _o.Jobs != _o.Lane) continue; // the other lanes run the other seeds
            store.SeedsDone = s + 1; // written first: a seed that takes the process down is skipped as well
            Execute(seeds[s].Data, seeds[s].Name);
            if (executor.AbandonedThreads > 0 || shed) return Exit.Restart;
        }
        store.SeedsDone = seeds.Count;
        while (clock.Elapsed < _o.Duration)
        {
            var basis = survivors.Count > 0 && rng.Next(5) == 0 ? survivors[rng.Next(survivors.Count)] : corpus[rng.Next(corpus.Count)];
            var input = mutator.Mutate(basis, corpus, target.Text);
            if (Execute(input, null) && rng.Next(10) == 0)
            {
                if (survivors.Count < 200) survivors.Add(input);
                else survivors[rng.Next(survivors.Count)] = input;
            }
            if (executor.AbandonedThreads > 0 || shed) return Exit.Restart;
            if (clock.Elapsed > nextReport)
            {
                nextReport += TimeSpan.FromSeconds(20);
                Console.WriteLine($"[{tag}] {executions} executions ({executions / clock.Elapsed.TotalSeconds:0.#}/s), {store.Count} finding(s), {duplicates} duplicate(s), {flaky} unconfirmed, "
                    + $"tolerated: {(tolerated.Count == 0 ? "none" : string.Join(", ", tolerated.Select(kv => kv.Key + "=" + kv.Value.ToString(CultureInfo.InvariantCulture))))}");
            }
        }
        Console.WriteLine($"[{tag}] finished: {executions} executions, {store.Count} finding(s), {duplicates} duplicate(s), {flaky} unconfirmed slow or large input(s)");
        File.Delete(store.CurrentInputPath);
        return Exit.Done;
    }
}
