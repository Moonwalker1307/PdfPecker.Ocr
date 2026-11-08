namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Where the seed corpus comes from: the test fixtures and golden inputs, read in place from the repository, plus pages
/// of printed text that PdfPecker writes and renders at start-up, so the image and PDF targets have seeds that hold
/// real text even before the tests have fixtures of their own.
/// </summary>
internal sealed class SeedContext
{
    public string Root { get; }

    public SeedContext(string root) => Root = root;

    public static string FindRoot(string? explicitRoot)
    {
        if (explicitRoot != null) return Path.GetFullPath(explicitRoot);
        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
            for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, "PdfPecker.Ocr.slnx"))) return dir.FullName;
        throw new InvalidOperationException("Cannot find the repository root (PdfPecker.Ocr.slnx); pass --repo.");
    }

    private string Tests => Path.Combine(Root, "tests", "PdfPecker.Ocr.Tests");
    public string FixtureDir => Path.Combine(Tests, "Fixtures");
    public string GoldenDir => Path.Combine(Tests, "Golden");

    // Build output and findings folders are never seeds: a saved finding read back as a seed would hang every restarted
    // worker on the same input.
    private static readonly string[] SkippedFolders = ["bin", "obj", ".git", "node_modules", "TestResults"];

    /// <summary>Files under a repository-relative folder whose names match any of the extensions, skipping huge ones.</summary>
    public IEnumerable<(string Name, byte[] Data)> Files(string relativeDir, long maxBytes, params string[] extensions)
    {
        var dir = Path.Combine(Root, relativeDir);
        if (!Directory.Exists(dir)) yield break;
        foreach (var path in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories).OrderBy(p => p, StringComparer.Ordinal))
        {
            if (!extensions.Any(e => path.EndsWith(e, StringComparison.OrdinalIgnoreCase))) continue;
            var folders = Path.GetRelativePath(Root, Path.GetDirectoryName(path)!).Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (folders.Any(f => SkippedFolders.Contains(f, StringComparer.OrdinalIgnoreCase) || f.StartsWith("fuzz-findings", StringComparison.OrdinalIgnoreCase))) continue;
            if (new FileInfo(path).Length > maxBytes) continue;
            yield return (Path.GetRelativePath(Root, path), File.ReadAllBytes(path));
        }
    }

    // ---- Generated pages ----

    // Short pages of the kinds of text the models meet: a letter, a table of figures, mixed scripts, a rotated line and
    // small print. They are kept small, because every seed is run once before mutation starts and OCR is not cheap.
    private static readonly (string Name, string Html)[] Pages =
    [
        ("letter", "<body style=\"font:14px serif;margin:24px\"><h2>Invoice 2026-0042</h2><p>Example Customer Ltd.<br>1 Sample Street<br>Springfield</p><p>Thank you for your order.</p></body>"),
        ("table", "<body style=\"font:12px sans-serif;margin:24px\"><table border=1 cellspacing=0 cellpadding=4><tr><th>Item</th><th>Qty</th><th>Price</th></tr><tr><td>Paper A4</td><td>10</td><td>4.99</td></tr><tr><td>Toner</td><td>2</td><td>61.50</td></tr></table></body>"),
        ("scripts", "<body style=\"font:16px sans-serif;margin:24px\"><p>Grüße aus Århus, Ærø und Łódź.</p><p>Съешь же ещё этих мягких булок.</p><p>Ελληνικά 0123456789</p></body>"),
        ("rotated", "<body style=\"font:18px serif;margin:40px\"><p style=\"transform:rotate(180deg)\">Upside down line</p><p style=\"transform:rotate(7deg)\">Slightly skewed scan</p></body>"),
        ("small", "<body style=\"font:7px sans-serif;margin:12px;column-count:2\">" + string.Concat(Enumerable.Repeat("Small print that runs on in two columns. ", 30)) + "</body>"),
    ];

    private List<(string Name, byte[] Data)>? _pdfs;

    /// <summary>One-page PDFs of <see cref="Pages"/>, written by PdfPecker; text PDFs, so they also cover the skip-pages-with-text path.</summary>
    public IReadOnlyList<(string Name, byte[] Data)> GeneratedPdfs()
        => _pdfs ??= Pages.Select(p => ("generated:" + p.Name + ".pdf", HtmlToPdf.Convert(p.Html, new PdfOptions { PageSize = PageSize.A5 }))).ToList();

    /// <summary>
    /// <see cref="GeneratedPdfs"/> rendered by PdfPecker as images of the given format at a low resolution, so seeds stay
    /// small and fast to recognise.
    /// </summary>
    public IEnumerable<(string Name, byte[] Data)> RenderedImages(PdfImageFormat format, float dpi = 72)
    {
        foreach (var (name, pdf) in GeneratedPdfs())
            yield return (Path.ChangeExtension(name, RenderedImage.GetFileExtension(format)), PdfToImage.ConvertPage(pdf, 0, new ImageOptions { Dpi = dpi, Format = format }));
    }
}
