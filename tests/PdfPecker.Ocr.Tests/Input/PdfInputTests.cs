using PdfPecker.Ocr.Input;

namespace PdfPecker.Ocr.Tests.Input;

public sealed class PdfInputTests
{
    private static readonly double[] Letter = { 0, 0, 612, 792 };

    private static readonly WordSpec[][] OneLine = { new[] { new WordSpec("scanned", 72, 72, 100, 20) } };

    [Theory]
    [InlineData("", 1, false)]
    [InlineData("Page 3", 10, false)]
    [InlineData("  . , ; - 12345 ", 5, true)]
    [InlineData("Hello world", 10, true)]
    [InlineData("𝔘𝔫𝔦𝔠𝔬𝔡𝔢 text", 10, true)]
    [InlineData("中文", 2, true)]
    public void PagesHaveTextFromTheLetterAndDigitCount(string text, int min, bool expected)
        => Assert.Equal(expected, PdfPageSource.HasText(text, min));

    [Fact]
    public void RecognitionReturnsTheTextOfPagesThatHaveIt()
    {
        var text = new RawPdf.Page(Letter, Content: "BT /F1 12 Tf 72 700 Td (This page already has text.) Tj ET", Helvetica: true);
        var scan = new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792));
        var pdf = RawPdf.Build(text, scan);
        var recognizer = new FakeRecognizer(OneLine);
        var doc = DocumentReader.ReadPdf(pdf, new OcrOptions { Dpi = 144 }, recognizer, default);

        Assert.Equal(new[] { 1 }, recognizer.Calls.Select(c => c.PageIndex));
        Assert.Equal(new[] { 0, 1 }, doc.Pages.Select(p => p.PageIndex));
        Assert.Equal("This page already has text.\fscanned", doc.Text);
        var existing = doc.Pages[0];
        Assert.Equal((1224, 1584, 144f), (existing.Width, existing.Height, existing.Dpi));
        var first = existing.Lines[0].Words[0];
        Assert.Equal("This", first.Text);
        Assert.Equal(1f, first.Confidence);
        // The word's box in the pixels of 144 dpi: twice its position in points.
        Assert.InRange(first.Box.X, 143, 145);
        Assert.InRange(first.Box.Y + first.Box.Height, 2 * (792 - 700) - 8, 2 * (792 - 700) + 8);

        var forced = DocumentReader.ReadPdf(pdf, new OcrOptions { Dpi = 72, PageMode = OcrPageMode.Force }, new FakeRecognizer(OneLine), default);
        Assert.Equal("scanned\fscanned", forced.Text);
    }

    [Fact]
    public void RecognitionReadsTheChosenPagesOnce()
    {
        var pdf = RawPdf.Build(Enumerable.Repeat(new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792)), 5).ToArray());
        var recognizer = new FakeRecognizer(OneLine);
        var doc = DocumentReader.ReadPdf(pdf, new OcrOptions { Dpi = 30, Pages = PdfPageRange.Parse("5,1-2,2,last") }, recognizer, default);
        Assert.Equal(new[] { 0, 1, 4 }, doc.Pages.Select(p => p.PageIndex));
        Assert.All(recognizer.Calls, c => Assert.Equal((255, 330, PixelFormat.Gray8), (c.Width, c.Height, c.Format)));
    }

    [Fact]
    public void ImageRecognitionUsesTheImageResolution()
    {
        var tiff = PdfRenderer.Open(RawPdf.Build(new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792)))).RenderAllToTiff(new ImageOptions { Dpi = 40 });
        var recognizer = new FakeRecognizer(OneLine);
        var page = DocumentReader.ReadImage(tiff, null, recognizer, default);
        Assert.Equal((0, 340, 440, 40f), (page.PageIndex, page.Width, page.Height, page.Dpi));
    }

    [Fact]
    public void BadOptionsFailBeforeAnyWork()
    {
        var pdf = RawPdf.Build(new RawPdf.Page(Letter));
        var recognizer = new FakeRecognizer(OneLine);
        foreach (var options in new[]
        {
            new OcrOptions { Dpi = 0 }, new OcrOptions { Dpi = float.NaN }, new OcrOptions { Dpi = 5000 },
            new OcrOptions { MinConfidence = -0.1f }, new OcrOptions { MinConfidence = float.NaN },
            new OcrOptions { PageMode = (OcrPageMode)7 }, new OcrOptions { MaxDegreeOfParallelism = 0 },
            new OcrOptions { MaxPixels = 9_999 }, new OcrOptions { MinPageTextCharacters = 0 },
        })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => DocumentReader.ReadPdf(pdf, options, recognizer, default));
            Assert.Throws<ArgumentOutOfRangeException>(() => Ocr.Pdf.SearchablePdf.MakeSearchable(pdf, options, recognizer, default));
        }
        Assert.Empty(recognizer.Calls);
    }

    [Fact]
    public void BadPdfsFailWithDocumentedExceptions()
    {
        var recognizer = new FakeRecognizer(OneLine);
        Assert.Throws<ArgumentNullException>(() => DocumentReader.ReadPdf(null!, null, recognizer, default));
        Assert.Throws<InvalidDataException>(() => DocumentReader.ReadPdf("not a pdf"u8.ToArray(), null, recognizer, default));
        Assert.Throws<InvalidDataException>(() => Ocr.Pdf.SearchablePdf.MakeSearchable(Array.Empty<byte>(), null, recognizer, default));
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var pdf = RawPdf.Build(new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792)));
        Assert.ThrowsAny<OperationCanceledException>(() => Ocr.Pdf.SearchablePdf.MakeSearchable(pdf, null, recognizer, cts.Token));
    }

    [Fact]
    public void PageRasterKeepsWithinThePixelBudget()
    {
        Assert.Equal(300, PageRaster.FitDpi(612, 792, 300, 100_000_000));
        foreach (long max in new long[] { 10_000, 123_457, 1_000_000, 8_000_000 })
        {
            float dpi = PageRaster.FitDpi(612, 792, 600, max);
            Assert.True(PageRaster.PixelCount(612, 792, dpi) <= max);
            Assert.True(PageRaster.PixelCount(612, 792, dpi * 1.01f) > max * 0.99);
        }
        // A long, narrow page: the budget, not the side, is the limit.
        float narrow = PageRaster.FitDpi(14_400, 10, 300, 100_000_000);
        Assert.Equal(300, narrow);
        Assert.True(PageRaster.PixelCount(1e9, 1e9, PageRaster.FitDpi(1e9, 1e9, 300, 10_000)) <= 10_000);
    }
}
