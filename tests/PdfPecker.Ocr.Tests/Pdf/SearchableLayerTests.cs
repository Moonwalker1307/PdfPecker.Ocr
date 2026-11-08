using System.Diagnostics;
using PdfPecker.Ocr.Pdf;
using Xunit.Abstractions;

namespace PdfPecker.Ocr.Tests.Pdf;

public sealed class SearchableLayerTests
{
    private readonly ITestOutputHelper _output;

    public SearchableLayerTests(ITestOutputHelper output) => _output = output;

    private static readonly double[] Letter = { 0, 0, 612, 792 };

    private static OcrOptions Options(float dpi = 150) => new() { Dpi = dpi, PageMode = OcrPageMode.Force };

    private static byte[] Scan(int pages = 1) => RawPdf.Build(Enumerable.Range(0, pages).Select(_ => new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792))).ToArray());

    private static readonly WordSpec[][] TwoLines =
    {
        new[] { new WordSpec("Invoice", 72, 72, 80, 20), new WordSpec("2026-11", 160, 72, 70, 20) },
        new[] { new WordSpec("Total:", 72, 120, 50, 12), new WordSpec("1,234.50", 130, 120, 60, 12), new WordSpec("EUR", 196, 120, 30, 12) },
    };

    [Fact]
    public void ExtractedTextHasTheWordsInOrderAndInPlace()
    {
        var result = SearchablePdf.MakeSearchable(Scan(), Options(), new FakeRecognizer(TwoLines), default);
        var page = PdfText.ExtractPages(result)[0];
        foreach (var line in page.Lines)
            foreach (var w in line.Words) _output.WriteLine($"{w.Text} {w.X:0.00} {w.Y:0.00} {w.Width:0.00} {w.Height:0.00} {w.FontSize:0.00}");
        Assert.Equal(new[] { "Invoice 2026-11", "Total: 1,234.50 EUR" }, page.Lines.Select(l => l.Text));
        var words = page.Lines.SelectMany(l => l.Words).ToList();
        var expected = TwoLines.SelectMany(l => l).ToList();
        Assert.Equal(expected.Select(w => w.Text), words.Select(w => w.Text));
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.InRange(words[i].X, expected[i].X - 1, expected[i].X + 1);
            Assert.InRange(words[i].Y, expected[i].Y - 1, expected[i].Y + 1);
            Assert.InRange(words[i].Width, expected[i].Width - 1, expected[i].Width + 1);
            Assert.InRange(words[i].Height, expected[i].Height - 1, expected[i].Height + 1);
        }
    }

    [Fact]
    public void LayerIsInvisible()
    {
        var scan = Scan();
        var result = SearchablePdf.MakeSearchable(scan, Options(), new FakeRecognizer(TwoLines), default);
        Assert.NotEqual(scan, result);
        Assert.Equal(Pixels.Render(scan, 100)[0], Pixels.Render(result, 100)[0]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(90)]
    [InlineData(180)]
    [InlineData(270)]
    public void WordsLandWhereTheyAreOnRotatedAndCroppedPages(int rotate)
    {
        // A crop box away from the origin and a rotation: the words are given as the page is displayed, and the text
        // layer must put them there.
        var page = new RawPdf.Page(new double[] { -50, 20, 700, 900 }, new double[] { 30, 60, 530, 760 }, rotate, RawPdf.Shapes(700, 900));
        var scan = RawPdf.Build(page);
        var (w, h) = PdfRenderer.Open(scan).GetPageSize(0);
        Assert.Equal(rotate is 90 or 270 ? (700.0, 500.0) : (500.0, 700.0), (w, h));
        var words = new[] { new[] { new WordSpec("Corner", 20, 30, 60, 14), new WordSpec("text", 90, 30, 40, 14) }, new[] { new WordSpec("Middle", w / 2, h / 2, 70, 16) } };
        var result = SearchablePdf.MakeSearchable(scan, Options(), new FakeRecognizer(words), default);

        var text = PdfText.ExtractPages(result)[0];
        Assert.Equal((w, h), (text.Width, text.Height));
        var found = text.Lines.SelectMany(l => l.Words).ToList();
        foreach (var x in found) _output.WriteLine($"{rotate}: {x.Text} {x.X:0.00} {x.Y:0.00} {x.Width:0.00} {x.Height:0.00}");
        var expected = words.SelectMany(l => l).ToList();
        Assert.Equal(expected.Select(e => e.Text), found.Select(f => f.Text));
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.InRange(found[i].X, expected[i].X - 1, expected[i].X + 1);
            Assert.InRange(found[i].Y, expected[i].Y - 1, expected[i].Y + 1);
        }
        Assert.Equal(Pixels.Render(scan, 72)[0], Pixels.Render(result, 72)[0]);
    }

    [Fact]
    public void RotatedWordsKeepTheirText()
    {
        var words = new[]
        {
            new[] { new WordSpec("Upwards", 100, 600, 90, 18, 90) },
            new[] { new WordSpec("Slanted", 200, 300, 90, 18, 30) },
            new[] { new WordSpec("Upside", 500, 500, 80, 18, 180), new WordSpec("down", 410, 500, 60, 18, 180) },
        };
        var result = SearchablePdf.MakeSearchable(Scan(), Options(), new FakeRecognizer(words), default);
        var page = PdfText.ExtractPages(result)[0];
        foreach (var l in page.Lines) _output.WriteLine($"{l.Text} angle {l.Angle:0.0} at {l.X:0.0},{l.Y:0.0} {l.Width:0.0}x{l.Height:0.0}");
        foreach (var word in new[] { "Upwards", "Slanted", "Upside", "down" }) Assert.Contains(word, page.Text);
        var slanted = page.Lines.Single(l => l.Text.Contains("Slanted"));
        Assert.InRange(slanted.Angle, 25, 35);
    }

    [Fact]
    public void UnicodeComesBackFromTheLayer()
    {
        string[] samples = { "Blåbærsyltetøy", "Straße", "naïve café", "Ærøskøbing", "Ελληνικά", "Кириллица", "中文文本", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "😀🎉", "áccent" };
        var lines = samples.Select((s, i) => s.Split(' ').Select((w, k) => new WordSpec(w, 72 + 120 * k, 60 + 30 * i, 100, 16)).ToArray()).ToArray();
        var result = SearchablePdf.MakeSearchable(Scan(), Options(), new FakeRecognizer(lines), default);
        var text = PdfText.ExtractPages(result)[0].Lines.Select(l => l.Text).ToArray();
        _output.WriteLine(string.Join("\n", text));
        Assert.Equal(samples, text);
    }

    [Fact]
    public void EmptyPageGetsNoText()
    {
        var scan = Scan(2);
        var recognizer = new FakeRecognizer(i => i == 0 ? Array.Empty<IReadOnlyList<WordSpec>>() : TwoLines);
        var result = SearchablePdf.MakeSearchable(scan, Options(), recognizer, default);
        var pages = PdfText.ExtractPages(result);
        Assert.Equal("", pages[0].Text.Trim());
        Assert.Contains("Invoice", pages[1].Text);
        Assert.Equal(Pixels.Render(scan, 50), Pixels.Render(result, 50));
    }

    [Fact]
    public void PagesWithTextAreSkippedUnlessForced()
    {
        var text = new RawPdf.Page(Letter, Content: "BT /F1 12 Tf 72 700 Td (This page already has plenty of text.) Tj ET", Helvetica: true);
        var short_ = new RawPdf.Page(Letter, Content: "BT /F1 12 Tf 72 40 Td (Page 3) Tj ET 0.5 g 72 72 300 300 re f", Helvetica: true);
        var scan = new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792));
        var pdf = RawPdf.Build(text, scan, short_);

        var skip = new FakeRecognizer(TwoLines);
        var result = SearchablePdf.MakeSearchable(pdf, new OcrOptions { Dpi = 72 }, skip, default);
        Assert.Equal(new[] { 1, 2 }, skip.Calls.Select(c => c.PageIndex));
        var pages = PdfText.ExtractPages(result);
        Assert.DoesNotContain("Invoice", pages[0].Text);
        Assert.Contains("Invoice", pages[1].Text);
        Assert.Contains("Invoice", pages[2].Text);

        var force = new FakeRecognizer(TwoLines);
        SearchablePdf.MakeSearchable(pdf, new OcrOptions { Dpi = 72, PageMode = OcrPageMode.Force }, force, default);
        Assert.Equal(new[] { 0, 1, 2 }, force.Calls.Select(c => c.PageIndex));

        // A searchable result has text on every page, so a second run reads nothing and returns a copy.
        var again = new FakeRecognizer(TwoLines);
        var second = SearchablePdf.MakeSearchable(result, new OcrOptions { Dpi = 72 }, again, default);
        Assert.Empty(again.Calls);
        Assert.Equal(result, second);
        Assert.NotSame(result, second);
    }

    [Fact]
    public void PageSelectionAndPassword()
    {
        var html = string.Join("", Enumerable.Range(1, 4).Select(i => $"<div style='break-after:page;height:100px;background:#888'></div>"));
        var plain = HtmlToPdf.Convert(html);
        var pdf = PdfProtector.Encrypt(plain, new PdfSecurity { UserPassword = "secret", OwnerPassword = "owner" });
        var recognizer = new FakeRecognizer(TwoLines);
        var options = new OcrOptions { Dpi = 72, Password = "secret", Pages = PdfPageRange.Parse("4,2,2") };
        var result = SearchablePdf.MakeSearchable(pdf, options, recognizer, default);
        Assert.Equal(new[] { 1, 3 }, recognizer.Calls.Select(c => c.PageIndex));
        var pages = PdfText.ExtractPages(result);
        Assert.Equal(new[] { false, true, false, true }, pages.Select(p => p.Text.Contains("Invoice")));

        Assert.Throws<PdfPasswordException>(() => SearchablePdf.MakeSearchable(pdf, new OcrOptions(), recognizer, default));
        Assert.Throws<ArgumentOutOfRangeException>(() => SearchablePdf.MakeSearchable(plain, new OcrOptions { Pages = PdfPageRange.Parse("9") }, recognizer, default));
    }

    [Fact]
    public void RendersAtTheDpiWithinMaxPixels()
    {
        var recognizer = new FakeRecognizer(TwoLines);
        SearchablePdf.MakeSearchable(Scan(), new OcrOptions { Dpi = 300, PageMode = OcrPageMode.Force }, recognizer, default);
        Assert.Equal((2550, 3300, 300f), (recognizer.Calls[0].Width, recognizer.Calls[0].Height, recognizer.Calls[0].Dpi));

        recognizer = new FakeRecognizer(TwoLines);
        var result = SearchablePdf.MakeSearchable(Scan(), new OcrOptions { Dpi = 300, MaxPixels = 1_000_000, PageMode = OcrPageMode.Force }, recognizer, default);
        var call = recognizer.Calls[0];
        Assert.True((long)call.Width * call.Height <= 1_000_000);
        Assert.True(call.Width * call.Height > 900_000);
        // A letter page is 93.5 square inches, so a million pixels allow about 103 dpi.
        Assert.InRange(call.Dpi, 100, 104);
        // The words still land in place, since the layer scales by the page size.
        var first = PdfText.ExtractPages(result)[0].Lines[0].Words[0];
        Assert.InRange(first.X, 71, 73);
    }

    [Fact]
    public void ImagesBecomeASearchablePdf()
    {
        var page = PdfRenderer.Open(Scan(3)).RenderAllToTiff(new ImageOptions { Dpi = 100 });
        var png = PdfRenderer.Open(Scan()).RenderPage(0, new ImageOptions { Dpi = 200 }).ToPng();
        var recognizer = new FakeRecognizer(TwoLines);
        // The PNG records no resolution, so it gets the one of the options.
        var pdf = SearchablePdf.FromImages(new[] { page, png }, new OcrOptions { Dpi = 200 }, recognizer, default);
        Assert.Equal(new[] { 0, 1, 2, 3 }, recognizer.Calls.Select(c => c.PageIndex));
        Assert.Equal(new[] { 850, 850, 850, 1700 }, recognizer.Calls.Select(c => c.Width));
        var info = PdfDocumentProperties.Read(pdf);
        Assert.Equal(4, info.PageCount);
        // Each page is the size of its image at the resolution the image records, so a letter page comes back as one.
        foreach (var p in info.Pages)
        {
            Assert.InRange(p.Width, 611, 613);
            Assert.InRange(p.Height, 791, 793);
        }
        var text = PdfText.ExtractPages(pdf);
        Assert.All(text, t => Assert.Equal(new[] { "Invoice 2026-11", "Total: 1,234.50 EUR" }, t.Lines.Select(l => l.Text)));
        Assert.InRange(text[3].Lines[0].Words[0].X, 71, 73);
    }

    [Fact]
    public void ThousandsOfWordsAreQuick()
    {
        var lines = Enumerable.Range(0, 100).Select(l => Enumerable.Range(0, 50)
            .Select(k => new WordSpec("w" + (l * 50 + k).ToString(System.Globalization.CultureInfo.InvariantCulture) + "æ", 10 + 11.8 * k, 10 + 7.6 * l, 10, 6)).ToArray()).ToArray();
        var page = new FakeRecognizer(lines).Recognize(Input.PagePixels.FromGray(new byte[2550 * 3300], 2550, 3300), 0, 300, new OcrOptions(), default);
        var layer = new[] { new LayerPage(page, 612, 792) };
        SearchableLayer.Write(layer);

        long before = GC.GetAllocatedBytesForCurrentThread();
        var sw = Stopwatch.StartNew();
        const int runs = 10;
        byte[] stamp = Array.Empty<byte>();
        for (int i = 0; i < runs; i++) stamp = SearchableLayer.Write(layer);
        sw.Stop();
        long allocated = (GC.GetAllocatedBytesForCurrentThread() - before) / runs;
        _output.WriteLine($"5,000 words: {sw.Elapsed.TotalMilliseconds / runs:0.00} ms and {allocated / 1024.0:0} KB per layer, {stamp.Length} bytes");
        Assert.True(sw.Elapsed.TotalMilliseconds / runs < 500, "Writing 5,000 words takes too long.");
        Assert.True(allocated < 8 * 1024 * 1024, "Writing 5,000 words allocates too much.");

        var scan = Scan();
        sw.Restart();
        var result = SearchablePdf.MakeSearchable(scan, new OcrOptions { Dpi = 72, PageMode = OcrPageMode.Force }, new FakeRecognizer(lines), default);
        _output.WriteLine($"MakeSearchable with 5,000 words: {sw.ElapsedMilliseconds} ms, {result.Length} bytes");
        var words = PdfText.ExtractPages(result)[0].Lines.SelectMany(l => l.Words).Select(w => w.Text).ToList();
        Assert.Equal(lines.SelectMany(l => l).Select(w => w.Text), words);
    }
}
