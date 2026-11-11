using System.Diagnostics;
using System.Globalization;
using HeadTuner;
using PdfPecker.Ocr.Models;

// Writes src/PdfPecker.Ocr/Models/rec-head.bin: retrained output weights for the recogniser's classes that PP-OCRv6
// confuses on Danish and Norwegian text (Å read as A, Æ as A and E, Ø as O or a zero). It renders pages of generated
// text in PdfPecker's bundled fonts, clean and degraded like scans, reads them with the converted recogniser, aligns the
// known text with the network's output, and refits only the last layer's weights of the tuned classes. Usage, from
// the repository root:
//   dotnet run -c Release --project tools/HeadTuner -- [--pages 800] [--out path]
// See README.md. Every random choice is seeded, so a run on the same processor writes the same file.

// The classes that compete in the refit. 0, O and o take part, so the others learn not to take their places, but keep
// their weights: a zero or an O next to letters must not change.
string[] tunedCharacters = ["A", "Å", "Æ", "E", "É", "È", "Ê", "e", "O", "Ø", "0", "o", "ø", "a", "å", "æ", "∅", "Ä", "Ö", "ö", "ä"];
string[] frozenCharacters = ["0", "O", "o"];
const int Seed = 1;
const int LinesPerPage = 12;
const int Epochs = 8;
const double LearningRate = 1e-3;
const double Lambda = 1e-4;

int pages = 800;
string outPath = Path.Combine("src", "PdfPecker.Ocr", "Models", "rec-head.bin");
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--pages": pages = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--out": outPath = args[++i]; break;
        default:
            Console.Error.WriteLine("Usage: HeadTuner [--pages n] [--out path]");
            return 2;
    }
}

var watch = Stopwatch.StartNew();
ModelDefinition definition;
using (var data = ModelStore.Inflate("rec"))
{
    definition = ModelDefinition.Deserialize(data.GetBuffer().AsSpan(0, (int)data.Length));
}

var classes = tunedCharacters.Select(c => Array.IndexOf(definition.Characters, c)).ToArray();
if (classes.Any(c => c <= 0)) throw new InvalidDataException("A tuned character is not in the recogniser's character list.");
var collector = new SampleCollector(definition, classes);

// Pages are read in parallel, each in an inference context of its own, and their samples joined in page order, so
// the result does not depend on the number of threads.
var styles = new PageStyle[pages];
var styleRandom = new Random(Seed);
for (int p = 0; p < pages; p++) styles[p] = PageStyle.Draw(styleRandom);
var perPage = new PageSamples[pages];
int done = 0;
Parallel.For(0, pages, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, p =>
{
    var lines = TrainingText.Lines(LinesPerPage, (Seed * 100_000) + p);
    var (rgba, width, height) = Pages.Render(lines, styles[p]);
    perPage[p] = collector.Read(lines, rgba, width, height, styles[p].Dpi, new Random((Seed * 100_000) + p));
    int n = Interlocked.Increment(ref done);
    if (n % 100 == 0) Console.WriteLine($"  {n} of {pages} pages read ({watch.Elapsed.TotalSeconds:0} s)");
});

var samples = perPage.SelectMany(p => p.Samples).ToList();
Console.WriteLine($"{pages} pages: {perPage.Sum(p => p.LinesUsed):N0} lines used, {perPage.Sum(p => p.LinesSkipped):N0} skipped, {samples.Count:N0} samples");

var (weights, bias) = HeadPatch.FinalLayer(definition);
int k = weights.Shape[0], n = weights.Shape[1];
double[][] original = [.. classes.Select(c =>
{
    var row = new double[k + 1];
    for (int i = 0; i < k; i++) row[i] = weights.Data[(i * n) + c];
    row[k] = bias.Data[c];
    return row;
})];
var frozen = frozenCharacters.Select(c => Array.IndexOf(tunedCharacters, c)).ToHashSet();
double[][] fitted = Fit.Run(samples, original, frozen, Epochs, LearningRate, Lambda, Seed, Console.Out);

// The patch holds only the classes whose weights changed.
var columns = new List<HeadPatch.Column>();
for (int j = 0; j < classes.Length; j++)
{
    if (frozen.Contains(j)) continue;
    columns.Add(new HeadPatch.Column(classes[j], tunedCharacters[j], [.. fitted[j].Take(k).Select(v => (float)v)], (float)fitted[j][k]));
}

byte[] patch = HeadPatch.Create(k, columns).Write();
File.WriteAllBytes(outPath, patch);
Console.WriteLine($"written {outPath}: {columns.Count} classes, {patch.Length:N0} bytes ({watch.Elapsed.TotalSeconds:0} s)");
return 0;
