using System.Diagnostics;
using System.Globalization;
using System.Text;
using TesseractComparison;

// Compares OCR engines on the synthetic corpus. Run with Native AOT (dotnet publish -r linux-x64 -c Release) or dotnet run.
//
//   TesseractComparison run    [--engines tesseract,rapidocr] [--threads 1,max] [--runs 5] [--warmup 1] [--corpus dir]
//                              [--out dir] [--lib libtesseract.so] [--tessdata dir] [--tess-lang auto|eng|multi|<codes>]
//                              [--tess-psm auto|1|3|...] [--python path] [--only id,id] [--title text]
//   TesseractComparison report --in dir          (writes results.md again from results.csv)
//   TesseractComparison child ...                (internal: measures one engine at one thread count in this process)
//
// "run" starts one child process per engine and thread count, so every engine's peak memory is its own and a crash cannot
// take the others down, then merges the children's CSV files into results.csv and results.md in the output folder.
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("usage: TesseractComparison run|report|child [options]; see the comment at the top of Program.cs");
    return 2;
}

var opts = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 1; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length) { Console.Error.WriteLine($"bad argument {args[i]}"); return 2; }
    opts[args[i][2..]] = args[++i];
}

string root = FindRepoRoot();
string Opt(string key, string fallback) => opts.GetValueOrDefault(key, fallback);
int OptInt(string key, int fallback) => int.Parse(Opt(key, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);

var childOptions = new ChildOptions
{
    Engine = Opt("engine", "tesseract"),
    Threads = int.TryParse(Opt("threads", "1"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var threadsValue) ? threadsValue : 1,
    CorpusDir = Opt("corpus", Path.Combine(root, "bench", "TesseractComparison", "corpus")),
    OutFile = Opt("out", "result.csv"),
    DumpDir = Opt("dump", ""),
    Runs = OptInt("runs", 5),
    Warmup = OptInt("warmup", 1),
    Only = opts.TryGetValue("only", out var only) ? only.Split(',') : null,
    TesseractLib = Opt("lib", Path.Combine(root, "tools", "Tesseract", "out", "linux-x64", "libtesseract.so")),
    Tessdata = Opt("tessdata", Path.Combine(root, "tools", "Tesseract", "out", "tessdata")),
    TessLang = Opt("tess-lang", "auto"),
    TessPsm = Opt("tess-psm", "auto"),
    Python = Opt("python", Environment.GetEnvironmentVariable("PDFPECKER_PYTHON") ?? "python3"),
    RapidScript = Path.Combine(AppContext.BaseDirectory, "rapidocr_server.py"),
};

switch (args[0])
{
    case "child":
        return Child.Run(childOptions);
    case "report":
        return WriteReport(Opt("in", "."), Opt("title", "OCR comparison"));
    case "run":
        return RunAll();
    default:
        Console.Error.WriteLine($"unknown command {args[0]}");
        return 2;
}

int RunAll()
{
    string outDir = Opt("out", Path.Combine(root, "bench", "TesseractComparison", "results", DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)));
    Directory.CreateDirectory(outDir);
    var engines = Opt("engines", PdfPeckerOcrAdapter.Available ? "tesseract,rapidocr,pdfpecker-ocr" : "tesseract,rapidocr").Split(',');
    var threadList = Opt("threads", "1,max").Split(',').Select(t => t == "max" ? Environment.ProcessorCount : int.Parse(t, CultureInfo.InvariantCulture)).Distinct().ToArray();
    string self = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
    bool viaDotnet = Path.GetFileNameWithoutExtension(self) == "dotnet";
    foreach (var engine in engines)
        foreach (int threads in threadList)
        {
            string csv = Path.Combine(outDir, $"{engine}-t{threads}.csv");
            var psi = new ProcessStartInfo(self) { RedirectStandardError = true, RedirectStandardOutput = true };
            if (viaDotnet) psi.ArgumentList.Add(Environment.GetCommandLineArgs()[0]);
            psi.ArgumentList.Add("child");
            void Add(string k, string v) { psi.ArgumentList.Add("--" + k); psi.ArgumentList.Add(v); }
            Add("engine", engine); Add("threads", threads.ToString(CultureInfo.InvariantCulture)); Add("out", csv);
            Add("dump", Path.Combine(outDir, "hyp", $"{engine}-t{threads}"));
            Add("corpus", childOptions.CorpusDir); Add("runs", childOptions.Runs.ToString(CultureInfo.InvariantCulture)); Add("warmup", childOptions.Warmup.ToString(CultureInfo.InvariantCulture));
            Add("lib", childOptions.TesseractLib); Add("tessdata", childOptions.Tessdata); Add("tess-lang", childOptions.TessLang); Add("tess-psm", childOptions.TessPsm);
            Add("python", childOptions.Python);
            if (childOptions.Only is not null) Add("only", string.Join(',', childOptions.Only));
            // Tesseract's OpenMP runtime reads its thread limit once, at load, so it has to be in the environment of the process.
            psi.Environment["OMP_THREAD_LIMIT"] = threads.ToString(CultureInfo.InvariantCulture);
            psi.Environment["OMP_NUM_THREADS"] = threads.ToString(CultureInfo.InvariantCulture);
            Console.Error.WriteLine($"== {engine}, {threads} thread(s)");
            using var p = Process.Start(psi)!;
            var log = new StringBuilder();
            p.ErrorDataReceived += (_, e) => { if (e.Data is not null) { lock (log) log.AppendLine(e.Data); if (e.Data.Contains(" ms") || e.Data.Contains("ERROR")) Console.Error.WriteLine(e.Data); } };
            p.BeginErrorReadLine();
            p.StandardOutput.ReadToEnd();
            p.WaitForExit();
            File.WriteAllText(Path.Combine(outDir, $"{engine}-t{threads}.log"), log.ToString());
            if (p.ExitCode != 0) Console.Error.WriteLine($"   child exited with {p.ExitCode}; see {engine}-t{threads}.log");
        }
    return WriteReport(outDir, Opt("title", "OCR comparison"));
}

int WriteReport(string dir, string title)
{
    var all = new List<ResultRow>();
    var meta = new Dictionary<string, Dictionary<string, string>>();
    foreach (var csv in Directory.GetFiles(dir, "*.csv").Where(f => !Path.GetFileName(f).Equals("results.csv", StringComparison.Ordinal)).OrderBy(f => f, StringComparer.Ordinal))
    {
        all.AddRange(Report.ReadRows(csv));
        if (File.Exists(csv + ".meta"))
        {
            var m = File.ReadAllLines(csv + ".meta").Select(l => l.Split('\t', 2)).Where(a => a.Length == 2).ToDictionary(a => a[0], a => a[1]);
            meta[$"{m.GetValueOrDefault("engine")}/{m.GetValueOrDefault("threads")}"] = m;
        }
    }
    if (all.Count == 0) { Console.Error.WriteLine("no result rows found"); return 1; }
    var merged = new List<string> { string.Join(',', Child.Header) };
    foreach (var csv in Directory.GetFiles(dir, "*.csv").Where(f => !Path.GetFileName(f).Equals("results.csv", StringComparison.Ordinal)).OrderBy(f => f, StringComparer.Ordinal))
        merged.AddRange(File.ReadLines(csv).Skip(1));
    File.WriteAllLines(Path.Combine(dir, "results.csv"), merged, new UTF8Encoding(false));
    var header = new StringBuilder(Report.Build(dir, all, meta, title));
    File.WriteAllText(Path.Combine(dir, "results.md"), MachineLine() + header, new UTF8Encoding(false));
    Console.WriteLine($"wrote {Path.Combine(dir, "results.md")} and results.csv");
    return 0;
}

static string MachineLine()
{
    string cpu = "unknown CPU";
    try { cpu = File.ReadLines("/proc/cpuinfo").First(l => l.StartsWith("model name", StringComparison.Ordinal)).Split(':')[1].Trim(); } catch (Exception e) when (e is IOException or InvalidOperationException) { }
    return $"<!-- {Environment.ProcessorCount} cores, {cpu}, {DateTime.UtcNow:yyyy-MM-dd} -->\n";
}

static string FindRepoRoot()
{
    foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "PdfPecker.Ocr.slnx"))) return dir.FullName;
    return Directory.GetCurrentDirectory();
}
