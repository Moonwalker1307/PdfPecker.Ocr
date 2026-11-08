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

    /// <summary>
    /// <see cref="GeneratedPdfs"/> as scans: each page rendered at 72 dpi and put back into a PDF as an image, with no
    /// text, so the PDF targets reach rendering and the models with the default page mode.
    /// </summary>
    public IEnumerable<(string Name, byte[] Data)> ScannedPdfs()
    {
        foreach (var (name, png) in RenderedImages(PdfImageFormat.Png).Take(3))
            yield return (Path.ChangeExtension(name, ".scan.pdf"), ImageToPdf.Convert([png], null, new ImageToPdfOptions { FitPageToImage = true, Dpi = 72 }));
    }

    /// <summary>
    /// The first generated page as a GIF and a BMP, which PdfPecker reads but does not write: an uncompressed 24-bit BMP,
    /// and a GIF whose LZW stream sends a clear code before its codes outgrow nine bits.
    /// </summary>
    public IEnumerable<(string Name, byte[] Data)> GeneratedGifAndBmp()
    {
        var (name, pdf) = GeneratedPdfs()[0];
        var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = 50 });
        int w = image.Width, h = image.Height;
        var gray = new byte[w * h];
        for (int i = 0; i < gray.Length; i++) gray[i] = image.Rgba[(i * 4) + 1];
        yield return (Path.ChangeExtension(name, ".gif"), Gif(gray, w, h));
        yield return (Path.ChangeExtension(name, ".bmp"), Bmp(gray, w, h));
    }

    private static byte[] Bmp(byte[] gray, int w, int h)
    {
        int stride = ((w * 3) + 3) & ~3;
        var bmp = new byte[54 + (stride * h)];
        var s = bmp.AsSpan();
        "BM"u8.CopyTo(s);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(s[2..], bmp.Length);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(s[10..], 54);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(s[14..], 40);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(s[18..], w);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(s[22..], h);
        System.Buffers.Binary.BinaryPrimitives.WriteInt16LittleEndian(s[26..], 1);
        System.Buffers.Binary.BinaryPrimitives.WriteInt16LittleEndian(s[28..], 24);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                bmp.AsSpan(54 + ((h - 1 - y) * stride) + (x * 3), 3).Fill(gray[(y * w) + x]);
        return bmp;
    }

    private static byte[] Gif(byte[] gray, int w, int h)
    {
        var gif = new List<byte>();
        gif.AddRange("GIF89a"u8.ToArray());
        gif.AddRange([(byte)w, (byte)(w >> 8), (byte)h, (byte)(h >> 8), 0xF7, 0, 0]);
        for (int i = 0; i < 256; i++) gif.AddRange([(byte)i, (byte)i, (byte)i]);
        gif.AddRange([0x2C, 0, 0, 0, 0, (byte)w, (byte)(w >> 8), (byte)h, (byte)(h >> 8), 0, 8]);
        var codes = new List<byte>();
        int buffer = 0, bits = 0, sinceClear = 0;
        void Emit(int code)
        {
            buffer |= code << bits;
            for (bits += 9; bits >= 8; bits -= 8)
            {
                codes.Add((byte)buffer);
                buffer >>= 8;
            }
        }

        Emit(256);
        foreach (byte g in gray)
        {
            if (sinceClear++ == 254)
            {
                Emit(256);
                sinceClear = 1;
            }

            Emit(g);
        }

        Emit(257);
        if (bits > 0) codes.Add((byte)buffer);
        for (int i = 0; i < codes.Count; i += 255)
        {
            int n = Math.Min(255, codes.Count - i);
            gif.Add((byte)n);
            gif.AddRange(codes.GetRange(i, n));
        }

        gif.AddRange([0, 0x3B]);
        return gif.ToArray();
    }
}
