using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>The end-to-end tests read full pages with the bundled models; they run one at a time so their threads do not compete.</summary>
[CollectionDefinition("EndToEnd", DisableParallelization = true)]
public sealed class EndToEndCollection;

/// <summary>
/// Pages drawn by PdfPecker and rendered at 300 dpi, read through the public API with the real models, and the text
/// compared exactly: clean Latin text with Nordic and German letters, numbers, a table, two columns, an upside-down
/// page, 8 point text, image files, scanned PDFs and the searchable text layer.
/// </summary>
[Collection("EndToEnd")]
public sealed class EndToEndTests
{
    [Fact]
    public void LatinTextWithNordicAndGermanLetters()
    {
        RenderedPage page = TestPages.Latin;
        OcrPage result = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        Assert.Equal(TestPages.LatinText, result.Text);
        Assert.Equal((0, page.Width, page.Height, 300f), (result.PageIndex, result.Width, result.Height, result.Dpi));
        Assert.InRange(result.Confidence, 0.95f, 1f);
        Assert.Equal(["Rødgrød", "med", "fløde,", "blåbærsyltetøj", "og", "æblekage."], result.Lines[1].Words.Select(w => w.Text));
        foreach (OcrLine line in result.Lines)
        {
            Assert.Equal(line.Text, string.Join(' ', line.Words.Select(w => w.Text)));
            Assert.InRange(line.Confidence, 0.9f, 1f);
        }
    }

    [Fact]
    public void Numbers()
    {
        RenderedPage page = TestPages.Numbers;
        OcrPage result = ImageOcr.RecognizeGray(page.Gray(), page.Width, page.Height);
        Assert.Equal(TestPages.NumbersText, result.Text);
    }

    [Fact]
    public void TableReadsRowByRow()
    {
        RenderedPage page = TestPages.Table;
        OcrPage result = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        Assert.Equal(TestPages.TableText, result.Text);

        // The cells of a row share a baseline, and the columns line up from row to row.
        Assert.All(result.Lines.Chunk(4), row => Assert.InRange(row.Max(l => l.Box.Y) - row.Min(l => l.Box.Y), 0, 8));
        for (int column = 0; column < 4; column++)
        {
            float[] lefts = [.. result.Lines.Where((_, i) => i % 4 == column).Select(l => l.Box.X)];
            Assert.InRange(lefts.Max() - lefts.Min(), 0, 8);
        }
    }

    [Fact]
    public void LongReceiptIsReadToTheEnd()
    {
        // A receipt of 72 by 900 mm at 200 dpi is 567 x 7,087 pixels. Scaled until its longer side fitted the detection
        // input, its 9 point text was about 5 pixels high and most lines were lost; each item number must be found.
        string items = string.Concat(Enumerable.Range(1, 120).Select(i => $"<p>Item {i:000} beans EUR {i * 3 % 97}.{i % 10}0</p>"));
        byte[] pdf = HtmlToPdf.Convert(TestPages.Document("72mm 900mm", "@page{margin:4mm} p{font:9pt/1.3 monospace;margin:0}", items));
        OcrDocument result = PdfOcr.Recognize(pdf, new OcrOptions { Dpi = 200, PageMode = OcrPageMode.Force });
        Assert.Single(result.Pages);
        var found = System.Text.RegularExpressions.Regex.Matches(result.Text, @"(?<![\d.])\d{3}(?![\d.])").Select(m => m.Value).ToHashSet();
        Assert.Equal(Enumerable.Range(1, 120).Select(i => i.ToString("000", System.Globalization.CultureInfo.InvariantCulture)).Where(n => !found.Contains(n)), []);
    }

    [Fact]
    public void LoneNarrowCharactersAreReadUpright()
    {
        // A table cell holding only I is a box taller than wide, which used to be read turned a quarter: as the CJK
        // character for one. On a page of horizontal text it is read upright.
        const string Rows = "<tr><td>Bolts</td><td>I</td><td>7</td></tr><tr><td>Nuts</td><td>l</td><td>4</td></tr>"
            + "<tr><td>Washers</td><td>!</td><td>1</td></tr><tr><td>Screws</td><td>1</td><td>I</td></tr>";
        byte[] pdf = HtmlToPdf.Convert(TestPages.Document("A4", "td{font:14pt sans-serif;padding:4mm 8mm}", $"<table>{Rows}</table><p>Totals are given in pieces.</p>"));
        OcrPage page = PdfOcr.Recognize(pdf, new OcrOptions { Dpi = 300, PageMode = OcrPageMode.Force }).Pages[0];
        Assert.DoesNotContain(page.Lines, l => l.Text.Contains('\u4E00', StringComparison.Ordinal));
        Assert.True(page.Lines.Any(l => l.Text == "I"), string.Join(" | ", page.Lines.Select(l => $"{l.Text} {l.Confidence:0.00} {l.Box.Width}x{l.Box.Height}")));
        Assert.Contains(page.Lines, l => l.Text == "Washers");
    }

    [Fact]
    public void TwoColumnsReadColumnByColumn()
    {
        RenderedPage page = TestPages.Columns;
        OcrPage result = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        Assert.Equal(TestPages.ColumnsText, result.Text);
        Assert.True(result.Lines[4].Box.X > result.Lines[3].Box.X + (result.Width / 3));
    }

    [Fact]
    public void UpsideDownPageIsTurnedAndReadFromItsTop()
    {
        RenderedPage page = TestPages.Latin;
        OcrPage upright = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        OcrPage turned = ImageOcr.RecognizeRgba(page.UpsideDown(), page.Width, page.Height);
        Assert.Equal(TestPages.LatinText, turned.Text);

        // The quads follow the text, which runs right to left on the turned page, and lie where the upright ones lie
        // after the turn.
        for (int i = 0; i < turned.Lines.Count; i++)
        {
            OcrQuad q = turned.Lines[i].Quad;
            Assert.True(q.TopRight.X < q.TopLeft.X);
            Assert.True(q.BottomLeft.Y < q.TopLeft.Y);
            OcrQuad u = upright.Lines[i].Quad;
            Assert.InRange(q.TopLeft.X, page.Width - u.TopLeft.X - 4, page.Width - u.TopLeft.X + 4);
            Assert.InRange(q.TopLeft.Y, page.Height - u.TopLeft.Y - 4, page.Height - u.TopLeft.Y + 4);
        }

        // Without the orientation step the turned page is not read as text.
        OcrPage unturned = ImageOcr.RecognizeRgba(page.UpsideDown(), page.Width, page.Height, new OcrOptions { DetectRotation = false, MinConfidence = 0 });
        Assert.DoesNotContain("quick brown fox", unturned.Text, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(90)]
    [InlineData(270)]
    public void PageTurnedByAQuarterIsRead(int degrees)
    {
        // Each line of a page on its side is a tall box, which is cropped upright and then checked for being upside
        // down, so the text reads as on the upright page, and the reading order follows the text's direction.
        RenderedPage page = TestPages.Latin;
        OcrPage result = ImageOcr.RecognizeGray(page.TurnedGray(degrees), page.Height, page.Width);
        Assert.Equal(TestPages.LatinText, result.Text);
        Assert.All(result.Lines, l => Assert.True(degrees == 90 ? l.Quad.TopRight.Y > l.Quad.TopLeft.Y + 500 : l.Quad.TopRight.Y < l.Quad.TopLeft.Y - 500));
    }

    [Fact]
    public void EightPointText()
    {
        RenderedPage page = TestPages.Small;
        OcrPage result = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        Assert.Equal(TestPages.SmallText, result.Text);

        // 8 points at 300 dpi: the lines are about 33 pixels high from ascender to descender.
        Assert.All(result.Lines, l => Assert.InRange(l.Box.Height, 22, 50));
    }

    [Fact]
    public void ImageFiles()
    {
        RenderedPage page = TestPages.Numbers;
        OcrPage fromBytes = ImageOcr.Recognize(page.Png);
        Assert.Equal(TestPages.NumbersText, fromBytes.Text);
        Assert.Equal((page.Width, page.Height), (fromBytes.Width, fromBytes.Height));

        using var stream = new MemoryStream(page.Png);
        Assert.Equal(TestPages.NumbersText, ImageOcr.Recognize(stream).Text);

        string path = Path.Combine(Path.GetTempPath(), "pdfpecker-ocr-" + Guid.NewGuid().ToString("N") + ".png");
        File.WriteAllBytes(path, page.Png);
        try
        {
            Assert.Equal(TestPages.NumbersText, ImageOcr.RecognizeFile(path).Text);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ScannedPdf()
    {
        RenderedPage page = TestPages.Latin;
        byte[] scan = PdfMerger.Merge([page.ScannedPdf(), TestPages.Numbers.ScannedPdf()]);
        OcrDocument document = PdfOcr.Recognize(scan);
        Assert.Equal([0, 1], document.Pages.Select(p => p.PageIndex));
        Assert.Equal(TestPages.LatinText + "\f" + TestPages.NumbersText, document.Text);
        Assert.Equal(300f, document.Pages[0].Dpi);
        Assert.InRange(document.Pages[0].Width, page.Width - 2, page.Width + 2);

        using var stream = new MemoryStream(scan);
        OcrDocument second = PdfOcr.Recognize(stream, new OcrOptions { Pages = PdfPageRange.Parse("2") });
        Assert.Equal(1, second.Pages.Single().PageIndex);
        Assert.Equal(TestPages.NumbersText, second.Text);

        string path = Path.Combine(Path.GetTempPath(), "pdfpecker-ocr-" + Guid.NewGuid().ToString("N") + ".pdf");
        File.WriteAllBytes(path, scan);
        try
        {
            Assert.Equal(TestPages.LatinText, PdfOcr.RecognizeFile(path, new OcrOptions { Pages = PdfPageRange.Parse("1") }).Text);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void BornDigitalPdfIsReadWhenForced()
    {
        // The HTML's own PDF has text, so by default it is extracted rather than read; Force reads the rendering.
        RenderedPage page = TestPages.Numbers;
        OcrDocument extracted = PdfOcr.Recognize(page.Pdf);
        Assert.All(extracted.Pages[0].Lines, l => Assert.Equal(1f, l.Confidence));
        OcrDocument forced = PdfOcr.Recognize(page.Pdf, new OcrOptions { PageMode = OcrPageMode.Force });
        Assert.Equal(TestPages.NumbersText, forced.Text);
        Assert.All(forced.Pages[0].Lines, l => Assert.True(l.Confidence < 1f));
    }

    [Fact]
    public void MakeSearchableRoundTrip()
    {
        byte[] scan = TestPages.Latin.ScannedPdf();
        Assert.Equal("", PdfText.Extract(scan).Trim());
        byte[] searchable = PdfOcr.MakeSearchable(scan);
        string extracted = PdfText.Extract(searchable);
        Assert.Equal(TestPages.LatinText, Normalize(extracted));

        // The pages look as they did: the layer is invisible.
        var before = PdfRenderer.Open(scan).RenderPage(0, new ImageOptions { Dpi = 50 });
        var after = PdfRenderer.Open(searchable).RenderPage(0, new ImageOptions { Dpi = 50 });
        Assert.Equal(before.Rgba, after.Rgba);

        // A second pass finds the text and leaves the PDF as it is.
        Assert.Equal(searchable, PdfOcr.MakeSearchable(searchable));
    }

    [Fact]
    public void ImagesToSearchablePdf()
    {
        byte[] pdf = ImageOcr.ToSearchablePdf([TestPages.Latin.Png, TestPages.Numbers.Png]);
        var pages = PdfText.ExtractPages(pdf);
        Assert.Equal(2, pages.Count);
        Assert.Equal(TestPages.LatinText, Normalize(pages[0].Text));
        Assert.Equal(TestPages.NumbersText, Normalize(pages[1].Text));
    }

    [Fact]
    public void ThreadCountDoesNotChangeTheResult()
    {
        RenderedPage page = TestPages.Invoice;
        byte[] gray = page.Gray();
        var options = new OcrOptions();
        OcrPage one = OcrEngine.Recognize(new PixelView(gray, page.Width, page.Height, 1), 0, 300, options, threads: 1, default);
        OcrPage three = OcrEngine.Recognize(new PixelView(gray, page.Width, page.Height, 1), 0, 300, options, threads: 3, default);
        OcrPage four = OcrEngine.Recognize(new PixelView(gray, page.Width, page.Height, 1), 0, 300, options, threads: 4, default);
        Assert.Contains("Total NOK 17,589.45", one.Text, StringComparison.Ordinal);
        AssertIdentical(one, three);
        AssertIdentical(one, four);

        // And through the public option.
        AssertIdentical(one, ImageOcr.RecognizeGray(gray, page.Width, page.Height, new OcrOptions { MaxDegreeOfParallelism = 4 }));
    }

    [Fact]
    public void InvoicePage()
    {
        RenderedPage page = TestPages.Invoice;
        OcrPage result = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        Assert.Equal(
            "Invoice 2026-0412\nExample Customer Ltd.\nStorgata 12, 0155 Oslo\nNorway\nDate: 8 November 2026, due 8 December 2026\n" +
            "Item\nQty\nPrice\nAmount\nConsulting, October\n12\n950.00\n11,400.00\nTravel expenses\n1\n1,234.56\n1,234.56\n" +
            "Software licence\n3\n199.00\n597.00\nSupport hours\n7\n120.00\n840.00\n" +
            "Subtotal 14,071.56\nVAT 25% 3,517.89\nTotal NOK 17,589.45\n" +
            "Payment within 30 days to account 1234.56.78901.\nPlease quote the invoice number with your payment.\n" +
            "Thank you for your order. Questions: invoices@example.com",
            result.Text);
    }

    [Fact]
    public void CancelledBeforeTheStart()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        RenderedPage page = TestPages.Small;
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, null, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.Recognize(page.Png, null, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => PdfOcr.Recognize(page.Pdf, new OcrOptions { PageMode = OcrPageMode.Force }, cts.Token));
    }

    [Fact]
    public void CancelledWhileReading()
    {
        // The public entry point with a pooled worker, its context and its token, cancelled from inside the work rather
        // than after a fixed delay, which a fast engine can beat: at the fourth step of the recognition network's second
        // batch of lines, when the page is detected, turned and partly read, and the network is half way through a run.
        RenderedPage page = TestPages.Invoice;
        byte[] gray = page.Gray();
        var options = new OcrOptions { MaxDegreeOfParallelism = 2 };
        using var probe = new CancelProbe(s => s.Model == ModelStore.RecognitionArgMax && s.Run == 1 && s.Index == 3);
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.RecognizeGray(gray, page.Width, page.Height, options, probe.Token));

        // The network itself stopped: the worker hands the token to its context, which is checked between the
        // operators of a run, so not one more operator ran.
        probe.AssertStoppedAtOnce();

        // The engine is still in order afterwards.
        RenderedPage small = TestPages.Small;
        Assert.Equal(TestPages.SmallText, ImageOcr.RecognizeRgba(small.Rgba, small.Width, small.Height, options).Text);
    }

    [Fact]
    public void MinConfidenceDropsLines()
    {
        RenderedPage page = TestPages.Latin;
        OcrPage all = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { MinConfidence = 0 });
        float lowest = all.Lines.Min(l => l.Confidence);
        OcrPage some = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { MinConfidence = MathF.BitIncrement(lowest) });
        Assert.Equal(all.Lines.Count - all.Lines.Count(l => l.Confidence == lowest), some.Lines.Count);
        Assert.DoesNotContain(ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { MinConfidence = 1 }).Lines, l => l.Confidence < 1);
    }

    [Fact]
    public void BlankPageHasNoLines()
    {
        var white = new byte[400 * 300];
        white.AsSpan().Fill(255);
        OcrPage result = ImageOcr.RecognizeGray(white, 400, 300);
        Assert.Empty(result.Lines);
        Assert.Equal("", result.Text);
        Assert.Equal(0f, result.Confidence);
    }

    private static void AssertIdentical(OcrPage expected, OcrPage actual)
    {
        Assert.Equal(expected.Text, actual.Text);
        Assert.Equal(expected.Lines.Count, actual.Lines.Count);
        for (int i = 0; i < expected.Lines.Count; i++)
        {
            OcrLine a = expected.Lines[i], b = actual.Lines[i];
            Assert.Equal(a.Quad, b.Quad);
            Assert.Equal(a.Confidence, b.Confidence);
            Assert.Equal(a.Words.Select(w => (w.Text, w.Quad, w.Confidence)), b.Words.Select(w => (w.Text, w.Quad, w.Confidence)));
        }

        Assert.Equal(expected.Confidence, actual.Confidence);
    }

    /// <summary>Extracted text with its lines trimmed, single spaces and no empty lines.</summary>
    private static string Normalize(string text) => string.Join('\n', text.Split('\n')
        .Select(l => string.Join(' ', l.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)))
        .Where(l => l.Length > 0));
}
