using System.Globalization;
using System.Text;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Tests.EndToEnd;

namespace PdfPecker.Ocr.Tests;

/// <summary>
/// Golden baselines: each <c>Golden/*.html</c> page is drawn by PdfPecker, rendered at 300 dpi (and for some pages
/// turned, coloured or degraded as a scan would be), read with the real models, and compared with the recognised text
/// in <c>*.expected.txt</c> and the lines and words with their boxes in <c>*.expected.boxes</c>. Every page is read
/// twice, with one thread under ar-SA and with four under fa-IR, whose digits, minus and plus signs differ from ASCII,
/// and both must give the baseline exactly. The expected files record what the models read today, recognition errors
/// included; they are not a statement of what the page says.
/// </summary>
[Collection("EndToEnd")]
public sealed class GoldenTests
{
    private const float Dpi = 300;

    /// <summary>Cultures whose number formatting differs from the invariant one in every way that matters here.</summary>
    private static readonly string[] Cultures = ["ar-SA", "fa-IR", "de-DE"];

    public static TheoryData<string> Pages()
    {
        var data = new TheoryData<string>();
        foreach (string name in GoldenFiles.PageNames()) data.Add(name);
        return data;
    }

    /// <summary>The pages whose searchable PDFs are baselines too: a table, Danish letters and an upside-down scan.</summary>
    public static TheoryData<string> SearchablePages() => new() { "invoice", "letter-da", "rotated-180" };

    [Theory]
    [MemberData(nameof(Pages))]
    public void PageMatchesGolden(string name)
    {
        GoldenPage page = GoldenPage.Get(name);
        OcrPage one = InCulture("ar-SA", () => page.Read(new OcrOptions { MaxDegreeOfParallelism = 1 }));
        OcrPage four = InCulture("fa-IR", () => page.Read(new OcrOptions { MaxDegreeOfParallelism = 4 }));

        string text = one.Text + "\n";
        string boxes = Boxes(one);
        GoldenFiles.Check(name + ".expected.txt", text);
        GoldenFiles.Check(name + ".expected.boxes", boxes);

        // Four threads must give the same lines, positions and confidences, bit for bit, not just the same rounding.
        Assert.Equal(text, four.Text + "\n");
        Assert.Equal(boxes, Boxes(four));
        Assert.Equal(Exact(one), Exact(four));

        // What the package writes from the result must not depend on the culture of the thread that asks.
        string hocr = InCulture("en-US", one.ToHocr);
        foreach (string culture in Cultures)
        {
            Assert.Equal(hocr, InCulture(culture, one.ToHocr));
            Assert.Equal(one.Text, InCulture(culture, () => one.Text));
        }
    }

    [Theory]
    [MemberData(nameof(SearchablePages))]
    public void SearchablePdfMatchesGolden(string name)
    {
        GoldenPage page = GoldenPage.Get(name);
        byte[] scan = page.ScannedPdf();
        byte[] searchable, again;
        // The stamper writes a random file identifier unless SOURCE_DATE_EPOCH (or a date the OCR options cannot pass
        // on) asks for a reproducible one.
        string? epoch = Environment.GetEnvironmentVariable("SOURCE_DATE_EPOCH");
        try
        {
            Environment.SetEnvironmentVariable("SOURCE_DATE_EPOCH", "1793880000");
            searchable = InCulture("ar-SA", () => PdfOcr.MakeSearchable(scan, new OcrOptions { Dpi = Dpi, MaxDegreeOfParallelism = 4 }));
            again = InCulture("de-DE", () => PdfOcr.MakeSearchable(scan, new OcrOptions { Dpi = Dpi, MaxDegreeOfParallelism = 1 }));
        }
        finally
        {
            Environment.SetEnvironmentVariable("SOURCE_DATE_EPOCH", epoch);
        }

        string normalised = PdfNormalizer.Normalize(searchable) + ExtractedText(searchable);
        GoldenFiles.Check(name + ".expected-pdf.txt", normalised);
        Assert.Equal(normalised, PdfNormalizer.Normalize(again) + ExtractedText(again));
        // With a fixed date the identifier is a hash of the result, so one runtime writes the same bytes every time.
        Assert.Equal(searchable, again);
    }

    /// <summary>
    /// What PdfPecker's text extraction reads from the searchable PDF: each line and word with its box in points to one
    /// place, which is how a viewer will select and search the layer.
    /// </summary>
    private static string ExtractedText(byte[] pdf)
    {
        var sb = new StringBuilder("extracted text\n");
        foreach (PdfTextPage page in PdfText.ExtractPages(pdf))
        {
            sb.Append(CultureInfo.InvariantCulture, $"page {page.PageNumber} {page.Width:0.#}x{page.Height:0.#}\n");
            foreach (PdfTextLine line in page.Lines)
            {
                sb.Append(CultureInfo.InvariantCulture, $"line {line.X:0.0},{line.Y:0.0} {line.Width:0.0}x{line.Height:0.0} angle {line.Angle:0} | {line.Text}\n");
                foreach (PdfTextWord word in line.Words)
                    sb.Append(CultureInfo.InvariantCulture, $"  word {word.X:0.0},{word.Y:0.0} {word.Width:0.0}x{word.Height:0.0} | {word.Text}\n");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// The lines and words of a page with their boxes rounded to whole pixels, their angle in whole degrees and their
    /// confidence to two places: precise enough that a change in detection or word splitting shows up, and coarse
    /// enough that the last bits of a float, which differ between SIMD paths, do not.
    /// </summary>
    internal static string Boxes(OcrPage page)
    {
        var sb = new StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"page {page.Width}x{page.Height} dpi {page.Dpi:0.##} lines {page.Lines.Count} confidence {page.Confidence:0.00}\n");
        foreach (OcrLine line in page.Lines)
        {
            sb.Append("line ").Append(Box(line.Box))
                .Append(CultureInfo.InvariantCulture, $" angle {Math.Round(OcrGeometry.AngleDegrees(line.Quad), MidpointRounding.AwayFromZero):0}")
                .Append(CultureInfo.InvariantCulture, $" conf {line.Confidence:0.00} | {line.Text}\n");
            foreach (OcrWord word in line.Words)
            {
                sb.Append("  word ").Append(Box(word.Box)).Append(CultureInfo.InvariantCulture, $" conf {word.Confidence:0.00} | {word.Text}\n");
            }
        }

        return sb.ToString();
    }

    private static string Box(OcrBox box)
    {
        static int R(float v) => (int)MathF.Round(v, MidpointRounding.AwayFromZero);
        return FormattableString.Invariant($"{R(box.X)},{R(box.Y)} {R(box.Width)}x{R(box.Height)}");
    }

    /// <summary>Every number of the result at full precision, for comparing two runs on one machine.</summary>
    private static string Exact(OcrPage page)
    {
        var sb = new StringBuilder();
        foreach (OcrLine line in page.Lines)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{line.Quad} {line.Confidence:R} {line.Text}\n");
            foreach (OcrWord word in line.Words) sb.Append(CultureInfo.InvariantCulture, $"  {word.Quad} {word.Confidence:R} {word.Text}\n");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Runs <paramref name="work"/> with <paramref name="culture"/> as the culture of this thread and of new threads,
    /// after emptying the engine pool, so that the engine's worker threads start in that culture too.
    /// </summary>
    internal static T InCulture<T>(string culture, Func<T> work)
    {
        CultureInfo current = CultureInfo.CurrentCulture, currentUi = CultureInfo.CurrentUICulture;
        CultureInfo? defaultCulture = CultureInfo.DefaultThreadCurrentCulture, defaultUi = CultureInfo.DefaultThreadCurrentUICulture;
        var info = CultureInfo.GetCultureInfo(culture);
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.CurrentUICulture = info;
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.DefaultThreadCurrentUICulture = info;
            EnginePool.Clear();
            return work();
        }
        finally
        {
            CultureInfo.CurrentCulture = current;
            CultureInfo.CurrentUICulture = currentUi;
            CultureInfo.DefaultThreadCurrentCulture = defaultCulture;
            CultureInfo.DefaultThreadCurrentUICulture = defaultUi;
            EnginePool.Clear();
        }
    }
}

/// <summary>
/// A golden page: <c>Golden/&lt;name&gt;.html</c> drawn by PdfPecker and rendered at 300 dpi, then changed the way the
/// page's name says (turned upside down, or degraded like a poor copy), as the pixels a scanner would deliver.
/// </summary>
internal sealed class GoldenPage
{
    private static readonly Dictionary<string, GoldenPage> Cache = new(StringComparer.Ordinal);
    private static readonly DateTimeOffset FixedDate = new(2026, 11, 8, 12, 0, 0, TimeSpan.Zero);

    private GoldenPage(byte[] rgba, int width, int height)
    {
        Rgba = rgba;
        Width = width;
        Height = height;
    }

    public byte[] Rgba { get; }

    public int Width { get; }

    public int Height { get; }

    public static GoldenPage Get(string name)
    {
        lock (Cache)
        {
            if (!Cache.TryGetValue(name, out GoldenPage? page))
            {
                page = Create(name);
                Cache[name] = page;
            }

            return page;
        }
    }

    public OcrPage Read(OcrOptions options) => ImageOcr.RecognizeRgba(Rgba, Width, Height, options);

    /// <summary>
    /// A scanned PDF of the page: its pixels as an image at 300 dpi on a page of the image's size, with a fixed date,
    /// so the file and what <see cref="PdfOcr.MakeSearchable(byte[], OcrOptions?, CancellationToken)"/> makes of it are
    /// reproducible. The image is a BMP, which needs no compressor of its own and so is the same on every runtime.
    /// </summary>
    public byte[] ScannedPdf()
    {
        var rgb = new byte[Width * Height * 3];
        for (int i = 0; i < Width * Height; i++) Rgba.AsSpan(i * 4, 3).CopyTo(rgb.AsSpan(i * 3));
        byte[] bmp = TestImages.Bmp(rgb, Width, Height, 11811);
        return ImageToPdf.Convert([bmp], new PdfOptions { CreationDate = FixedDate }, new ImageToPdfOptions { FitPageToImage = true, Dpi = 300 });
    }

    private static GoldenPage Create(string name)
    {
        RenderedPage rendered = TestPages.Render(GoldenFiles.Html(name), 300);
        byte[] rgba = rendered.Rgba;
        if (name.StartsWith("rotated-180", StringComparison.Ordinal)) rgba = rendered.UpsideDown();
        else if (name.StartsWith("degraded", StringComparison.Ordinal)) rgba = Degrade(rgba, rendered.Width, rendered.Height);
        return new GoldenPage(rgba, rendered.Width, rendered.Height);
    }

    /// <summary>
    /// A poor copy of the page: blurred twice with a 3 by 3 box, its contrast lowered to the range 60 to 215, and noise
    /// of up to 45 levels either way added from a fixed xorshift sequence, so the result is the same on every machine
    /// and runtime. The result is gray, in all three colour channels.
    /// </summary>
    private static byte[] Degrade(byte[] rgba, int width, int height)
    {
        var gray = new int[width * height];
        for (int i = 0; i < gray.Length; i++) gray[i] = rgba[(i * 4) + 1];
        gray = BoxBlur(BoxBlur(gray, width, height), width, height);

        var result = new byte[rgba.Length];
        uint state = 0x9E3779B9;
        for (int i = 0; i < gray.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            int noise = (int)(state % 91) - 45;
            int v = 60 + (gray[i] * 155 / 255) + noise;
            byte b = (byte)Math.Clamp(v, 0, 255);
            result[i * 4] = result[(i * 4) + 1] = result[(i * 4) + 2] = b;
            result[(i * 4) + 3] = 255;
        }

        return result;
    }

    private static int[] BoxBlur(int[] src, int width, int height)
    {
        var dst = new int[src.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int sum = 0, n = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= height) continue;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = x + dx;
                        if (xx < 0 || xx >= width) continue;
                        sum += src[(yy * width) + xx];
                        n++;
                    }
                }

                dst[(y * width) + x] = sum / n;
            }
        }

        return dst;
    }
}
