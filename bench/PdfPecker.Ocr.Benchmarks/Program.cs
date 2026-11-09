using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using PdfPecker.Ocr;
using OcrBench;

// A dependency-free benchmark harness for PdfPecker.Ocr: warm up, then time runs of every case and report the median, the
// mean, the allocation and CPU time per run, and the peak resident memory of the process.
// Usage: dotnet run -c Release [-- [--corpus dir] [--threads n] [--filter text]]
//        dotnet run -c Release -- --json            one pass in this process, the result as one line of JSON (what the
//                                                   version comparison starts a fresh process for)
//        dotnet run -c Release -- --perf <file>     the table on standard output and the result as JSON in <file>, which
//                                                   perf.yml keeps
//        dotnet run -c Release -- versions <other.dll> [rounds]   this build against another build of this program, in turns
//                                                   (bench/compare-versions.sh builds the other one from a git ref)
//        dotnet run -c Release -p:OcrInternals=true -- stages [--memory] [--threads n] [--runs n]
//                                                   an A4 invoice page at 300 dpi end to end, cold and warm, and its time
//                                                   by stage (or, with --memory, the resident peak of OCR alone); run it
//                                                   in a fresh process per thread count
//        dotnet run -c Release -p:OcrInternals=true -- profile [--corpus dir] [--only id,=id] [--threads n] [--runs n]
//                                                   corpus pages with every network layer timed (Profile.cs)
//        dotnet run -c Release -p:OcrInternals=true -- layers [--threads n] [--runs n] [--top n]
//                                                   each network's time on that page layer by layer
//        the API cases are on by default (-p:OcrApi=false turns them off); -p:OcrInternals=true adds the stage cases.
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

if (args.Length > 1 && args[0] == "versions")
{
    Versions.Compare(Path.GetFullPath(args[1]), args.Length > 2 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 3);
    return 0;
}

if (args.Length > 0 && args[0] == "profile")
{
#if OCR_INTERNALS
    int pt = int.TryParse(ArgValue("--threads"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var t2) ? t2 : Environment.ProcessorCount;
    int pr = int.TryParse(ArgValue("--runs"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var r2) ? r2 : 3;
    Profile.Run(ArgValue("--corpus") ?? "bench/TesseractComparison/corpus", ArgValue("--only")?.Split(','), pt, pr);
    return 0;
#else
    Console.Error.WriteLine("The profile command needs a build with -p:OcrInternals=true.");
    return 2;
#endif
}

if (args.Length > 0 && args[0] == "stages")
{
#if OCR_INTERNALS
    int st = int.TryParse(ArgValue("--threads"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var t1) ? t1 : Environment.ProcessorCount;
    int sr = int.TryParse(ArgValue("--runs"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var r1) ? r1 : 5;
    if (args.Contains("--memory")) Stages.Memory(st, sr);
    else Stages.Run(st, sr);
    return 0;
#else
    Console.Error.WriteLine("The stages command needs a build with -p:OcrInternals=true.");
    return 2;
#endif
}

if (args.Length > 0 && args[0] == "layers")
{
#if OCR_INTERNALS
    int lt = int.TryParse(ArgValue("--threads"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var t2) ? t2 : Environment.ProcessorCount;
    int lr = int.TryParse(ArgValue("--runs"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var r2) ? r2 : 3;
    int top = int.TryParse(ArgValue("--top"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var n2) ? n2 : 15;
    LayerProfile.Run(lt, lr, top);
    return 0;
#else
    Console.Error.WriteLine("The layers command needs a build with -p:OcrInternals=true.");
    return 2;
#endif
}

bool json = args.Contains("--json");
string? corpus = ArgValue("--corpus");
int threads = int.TryParse(ArgValue("--threads"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var t) ? t : Environment.ProcessorCount;
string? filter = ArgValue("--filter");
string? perf = ArgValue("--perf");

var cases = Cases.Create(corpus, threads).Where(c => filter is null || c.Name.Contains(filter, StringComparison.Ordinal)).ToList();
var result = Runner.RunAll(cases, quiet: json);
if (perf is not null) File.WriteAllText(perf, JsonSerializer.Serialize(result));
if (json) Console.WriteLine(JsonSerializer.Serialize(result));
else Runner.Print(result);
return 0;

string? ArgValue(string name)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}
