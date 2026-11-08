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
//        dotnet run -c Release -- versions <other.dll> [rounds]   this build against another build of this program, in turns
//                                                   (bench/compare-versions.sh builds the other one from a git ref)
//        add -p:OcrApi=true (and -p:OcrInternals=true) to the build to switch the real cases on; see the csproj.
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

if (args.Length > 1 && args[0] == "versions")
{
    Versions.Compare(Path.GetFullPath(args[1]), args.Length > 2 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 3);
    return 0;
}

bool json = args.Contains("--json");
string? corpus = ArgValue("--corpus");
int threads = int.TryParse(ArgValue("--threads"), NumberStyles.Integer, CultureInfo.InvariantCulture, out var t) ? t : Environment.ProcessorCount;
string? filter = ArgValue("--filter");

var cases = Cases.Create(corpus, threads).Where(c => filter is null || c.Name.Contains(filter, StringComparison.Ordinal)).ToList();
var result = Runner.RunAll(cases, quiet: json);
if (json) Console.WriteLine(JsonSerializer.Serialize(result));
else Runner.Print(result);
return 0;

string? ArgValue(string name)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}
