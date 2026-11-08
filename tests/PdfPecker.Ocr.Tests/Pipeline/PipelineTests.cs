using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>Whole pages through the pipeline, with networks that replay rapidocr's recorded outputs.</summary>
public sealed class PipelineTests
{
    private static readonly CharacterSet Characters = new(PipelineFixture.Characters[1..^1]);

    [Fact]
    public void GapPageReadsWithTheMissingSpace()
    {
        using PipelineFixture fx = PipelineFixture.Load("gap");
        PageResult page = Run(fx, orientation: null);
        Assert.Equal(8, page.Lines.Count);
        Assert.Equal("EUR 1,234.56 VAT 25%", page.Lines[0].Text);
        Assert.All(page.Lines.Take(6), l => Assert.Equal("EUR 1,234.56 VAT 25%", l.Text));
        Assert.Equal(["EUR 1,234.56", "VAT 25%"], page.Lines.Skip(6).Select(l => l.Text).ToArray());
        Assert.All(page.Lines.Take(6), l => Assert.Equal(["EUR", "1,234.56", "VAT", "25%"], l.Words.Select(w => w.Text).ToArray()));
    }

    [Fact]
    public void UpsideDownPageIsTurnedAndReadFromItsTop()
    {
        using PipelineFixture fx = PipelineFixture.Load("upside_down");
        PageResult page = Run(fx, ReplayNetworks.Orientation(upsideDown: true));
        Assert.Equal("Dear Sir or Madam,\nplease find enclosed the signed contract.\nKind regards, Anna Jensen", page.Text);
        Assert.All(page.Lines, l => Assert.True(l.UpsideDown));

        // Line quads run along the text, right to left on the page.
        Assert.All(page.Lines, l => Assert.True(l.Quad.P1.X < l.Quad.P0.X));
    }

    [Fact]
    public void ColourPageReadsAsRapidocr()
    {
        using PipelineFixture fx = PipelineFixture.Load("nordic");
        PageResult page = Run(fx, ReplayNetworks.Orientation(upsideDown: false));
        Assert.Equal(
            "Rødgrød med fløde og æblekage på AErø\nBlåbærsyltetøj, smørrebrød og øl i Århus\nGrößere Straßen führen über die Brücke\nSchöne Grüße aus München, Köln und Düsseldorf",
            page.Text);
        Assert.InRange(page.Confidence, 0.99f, 1f);
    }

    [Fact]
    public void BatchesFollowRapidocrsSizes()
    {
        // The table's 20 cells give four batches, sorted by aspect ratio, of the widths rapidocr used.
        using PipelineFixture fx = PipelineFixture.Load("table");
        FunctionNetwork rec = ReplayNetworks.Recognition(fx, Characters.Count);
        var pipeline = new OcrPipeline(ReplayNetworks.Detection(fx), rec, null, Characters, DetectionOptions.Rapidocr, new RecognitionOptions { DetectRotation = false });
        pipeline.Run(new PixelView(fx.Image(), fx.Width, fx.Height, fx.Channels), 150);
        // Each recorded line carries its batch's shape; a shape of n lines appears n times per batch.
        string[] expected = [.. RecordedLine.Read(fx).GroupBy(l => string.Join(',', l.InputShape))
            .SelectMany(g => Enumerable.Repeat(g.Key, g.Count() / g.First().InputShape[0])).Order(StringComparer.Ordinal)];
        string[] actual = [.. rec.InputShapes.Select(s => string.Join(',', s)).Order(StringComparer.Ordinal)];
        Assert.Equal(expected, actual);
    }

    private static PageResult Run(PipelineFixture fx, INetwork? orientation)
    {
        var pipeline = new OcrPipeline(
            ReplayNetworks.Detection(fx),
            ReplayNetworks.Recognition(fx, Characters.Count),
            orientation,
            Characters,
            DetectionOptions.Rapidocr,
            new RecognitionOptions { DetectRotation = orientation is not null });
        return pipeline.Run(new PixelView(fx.Image(), fx.Width, fx.Height, fx.Channels), 150);
    }
}
