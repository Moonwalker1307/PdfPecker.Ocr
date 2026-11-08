using System.Globalization;
using System.Text;
using Corpus;
using PdfPecker;

// Generates the synthetic OCR corpus: HTML pages with known text, drawn by PdfPecker, rendered to pixels at 300 and 150
// dpi, plus degraded copies. Writes, for each page, <id>.png (8-bit gray or RGB), <id>.txt (the ground truth lines in
// reading order) and <id>.boxes.tsv (ground truth word boxes in pixels), and manifest.tsv listing them. See README.md.
//
// Usage: dotnet run -c Release --project tools/Corpus -- [--out dir] [--seed n] [--dpis 300,150] [--only a,b] [--no-degraded] [--tesseract-samples]
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

string? outDir = null;
ulong seed = 20261108;
var dpis = new List<int> { 300, 150 };
string[]? only = null;
bool degraded = true;
bool tesseractSamples = false;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--out": outDir = args[++i]; break;
        case "--seed": seed = ulong.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--dpis": dpis = args[++i].Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToList(); break;
        case "--only": only = args[++i].Split(','); break;
        case "--no-degraded": degraded = false; break;
        case "--tesseract-samples": tesseractSamples = true; break;
        default: Console.Error.WriteLine($"unknown argument {args[i]}"); return 2;
    }
}

outDir ??= Path.Combine(FindRepoRoot(), "bench", "TesseractComparison", "corpus");
Directory.CreateDirectory(outDir);

// Pages that also get the degraded variants (at 300 dpi), chosen to cover a table of numbers, a serif letter and columns.
string[] degradeBases = ["invoice-en", "letter-de", "article-en"];

var manifest = new List<string>
{
    string.Join('\t', "id", "file", "width", "height", "channels", "dpi", "lang", "category", "font", "degradation", "rotation", "text", "boxes"),
};
int count = 0;

foreach (var spec in Pages.All(seed))
{
    var pdf = HtmlToPdf.Convert(spec.Html);
    var textPages = PdfText.ExtractPages(pdf);
    if (textPages.Count != 1) throw new InvalidOperationException($"{spec.Id} does not fit one page ({textPages.Count} pages): shorten its text.");
    var tp = textPages[0];
    CheckGroundTruth(spec, tp);
    foreach (var line in tp.Lines)
        foreach (var w in line.Words)
            if (w.X + w.Width > tp.Width - 20) throw new InvalidOperationException($"{spec.Id}: the word '{w.Text}' runs past the right margin.");

    var renderer = PdfRenderer.Open(pdf);
    foreach (int dpi in dpis)
    {
        string id = $"{spec.Id}.{dpi}";
        bool wanted = only is null || only.Any(o => id.Contains(o, StringComparison.Ordinal));
        var rendered = renderer.RenderPage(0, new ImageOptions { Dpi = dpi });
        var bitmap = Bitmap.FromRgba(rendered.Rgba, rendered.Width, rendered.Height, spec.Colour);
        double scale = dpi / 72.0;
        var boxes = TightBoxes(tp, scale, bitmap);
        if (wanted) { Write(spec, id, dpi, bitmap, boxes, "none", 0); count++; }

        if (!degraded || dpi != 300 || !degradeBases.Contains(spec.Id) || bitmap.Channels != 1) continue;
        foreach (var (name, make) in Variants(spec.Id, seed))
        {
            string vid = $"{id}.{name}";
            if (only is not null && !only.Any(o => vid.Contains(o, StringComparison.Ordinal))) continue;
            var (img, moved, rotation) = make(bitmap, boxes);
            Write(spec, vid, dpi, img, moved, name, rotation);
            count++;
        }
    }
}

if (tesseractSamples)
{
    foreach (var (name, lang, image, lines) in TesseractSamples.Load())
    {
        if (only is not null && !only.Any(o => name.Contains(o, StringComparison.Ordinal))) continue;
        // No ground truth boxes, and no resolution: the images say none, and Tesseract then estimates it itself.
        Write(new PageSpec(name, "tesseract-sample", lang, "photo", false, "", lines), name, 0, image, [], "none", 0);
        count++;
    }
}

File.WriteAllLines(Path.Combine(outDir, "manifest.tsv"), manifest, new UTF8Encoding(false));
Console.WriteLine($"{count} pages written to {outDir}");
return 0;

void Write(PageSpec spec, string id, int dpi, Bitmap bitmap, List<WordBox> boxes, string degradation, int rotation)
{
    File.WriteAllBytes(Path.Combine(outDir, id + ".png"), bitmap.ToPng());
    File.WriteAllText(Path.Combine(outDir, id + ".txt"), string.Join('\n', spec.Lines) + "\n", new UTF8Encoding(false));
    var sb = new StringBuilder();
    foreach (var b in boxes)
        sb.Append(b.X.ToString("0.0", CultureInfo.InvariantCulture)).Append('\t').Append(b.Y.ToString("0.0", CultureInfo.InvariantCulture)).Append('\t')
          .Append(b.Width.ToString("0.0", CultureInfo.InvariantCulture)).Append('\t').Append(b.Height.ToString("0.0", CultureInfo.InvariantCulture)).Append('\t')
          .Append(b.Text).Append('\n');
    File.WriteAllText(Path.Combine(outDir, id + ".boxes.tsv"), sb.ToString(), new UTF8Encoding(false));
    manifest.Add(string.Join('\t', id, id + ".png", bitmap.Width, bitmap.Height, bitmap.Channels, dpi, spec.Lang, spec.Category, spec.Font, degradation, rotation, id + ".txt", id + ".boxes.tsv"));
}

// The degraded variants: each takes the clean gray page and its ground truth boxes and returns the new page, the moved
// boxes and the rotation of the page in degrees (180 when the page is upside down, which the text order is unaffected by).
static IEnumerable<(string Name, Func<Bitmap, List<WordBox>, (Bitmap, List<WordBox>, int)> Make)> Variants(string id, ulong seed)
{
    yield return ("noise", (b, w) => (Degrade.Noise(b, 28, Rng.For(seed, id + "noise")), w, 0));
    yield return ("blur", (b, w) => (Degrade.Blur(b, 1.6), w, 0));
    foreach (var (name, deg) in new[] { ("skew1", 1.0), ("skew2", -2.0), ("skew3", 3.0) })
        yield return (name, (b, w) => (Degrade.Rotate(b, deg), w.Select(x => Degrade.RotateBox(x, b.Width, b.Height, deg)).ToList(), 0));
    yield return ("jpeg30", (b, w) => (Degrade.JpegQuality(b, 30), w, 0));
    yield return ("lowcontrast", (b, w) => (Degrade.LowContrast(b, 120, 175, 3, Rng.For(seed, id + "low")), w, 0));
    yield return ("rot180", (b, w) => (Degrade.Rotate180(b), w.Select(x => Degrade.Rotate180Box(x, b.Width, b.Height)).ToList(), 180));
    // A scan-like page: slight skew, noise and JPEG compression together.
    yield return ("scan", (b, w) =>
    {
        var r = Degrade.Rotate(b, 1.5);
        r = Degrade.Blur(r, 0.8);
        r = Degrade.Noise(r, 14, Rng.For(seed, id + "scan"));
        r = Degrade.JpegQuality(r, 40);
        return (r, w.Select(x => Degrade.RotateBox(x, b.Width, b.Height, 1.5)).ToList(), 0);
    });
}

// The ground truth is the text the HTML holds, and the words PdfPecker reads back from the PDF must be the same words:
// a difference means the layout broke a line, dropped a character or the template is wrong.
static void CheckGroundTruth(PageSpec spec, PdfTextPage page)
{
    static Dictionary<string, int> Bag(IEnumerable<string> words)
    {
        var d = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var w in words) d[w] = d.GetValueOrDefault(w) + 1;
        return d;
    }
    var expected = Bag(spec.Lines.SelectMany(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries)));
    var actual = Bag(page.Lines.SelectMany(l => l.Words.Select(w => w.Text)));
    var diff = new List<string>();
    foreach (var (w, n) in expected) if (actual.GetValueOrDefault(w) != n) diff.Add($"'{w}' expected {n}, read {actual.GetValueOrDefault(w)}");
    foreach (var (w, n) in actual) if (!expected.ContainsKey(w)) diff.Add($"'{w}' read {n}, not expected");
    if (diff.Count > 0) throw new InvalidOperationException($"{spec.Id}: ground truth differs from the PDF's text:\n  " + string.Join("\n  ", diff.Take(12)));
}

// PdfPecker's word boxes span the font's ascent and descent. The ground truth boxes are cut down to the dark pixels inside,
// the way an OCR engine's boxes hug the ink, so that IoU measures the engine and not the font's metrics.
static List<WordBox> TightBoxes(PdfTextPage page, double scale, Bitmap bitmap)
{
    var result = new List<WordBox>();
    foreach (var line in page.Lines)
        foreach (var w in line.Words)
        {
            int x0 = Math.Max(0, (int)Math.Floor(w.X * scale)), y0 = Math.Max(0, (int)Math.Floor(w.Y * scale));
            int x1 = Math.Min(bitmap.Width, (int)Math.Ceiling((w.X + w.Width) * scale)), y1 = Math.Min(bitmap.Height, (int)Math.Ceiling((w.Y + w.Height) * scale));
            int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                {
                    if (Luma(bitmap, x, y) >= 140) continue;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            if (maxX < 0) continue; // white on white, or a space: nothing to find
            result.Add(new WordBox(w.Text, minX, minY, maxX - minX + 1, maxY - minY + 1));
        }
    return result;
}

static int Luma(Bitmap b, int x, int y)
{
    if (b.Channels == 1) return b.Data[y * b.Width + x];
    int o = (y * b.Width + x) * 3;
    return (b.Data[o] * 77 + b.Data[o + 1] * 150 + b.Data[o + 2] * 29) >> 8;
}

static string FindRepoRoot()
{
    for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
        if (File.Exists(Path.Combine(dir.FullName, "PdfPecker.Ocr.slnx"))) return dir.FullName;
    throw new InvalidOperationException("Run this from inside the repository, or pass --out.");
}
