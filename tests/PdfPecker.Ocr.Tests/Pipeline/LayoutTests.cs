using System.Numerics;
using System.Text.Json;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>Reading order on rapidocr's recorded lines and on synthetic pages.</summary>
public sealed class LayoutTests
{
    [Fact]
    public void ColumnsAreReadOneAfterTheOther()
    {
        List<RecognizedLine> lines = FinalLines("columns");
        string[] order = Ordered(Shuffle(lines));
        Assert.Equal(
            [
                "Quarterly Newsletter",
                "Our new office opens in", "March. The move will take", "two weeks, and phones stay", "on throughout the move.", "Parking is free for guests.",
                "The annual meeting is held", "on 14 May at 10:00. Please", "register by 30 April so we", "can plan the catering for", "everyone who attends.",
                "Contact: office@example.org, phone +45 12 34 56 78",
            ],
            order);
    }

    [Fact]
    public void TablesAreReadRowByRow()
    {
        string[] order = Ordered(Shuffle(FinalLines("table")));
        Assert.Equal(["Region", "Units", "Revenue", "Share", "North", "1,204", "48,160", "31%", "South", "987", "39,480", "25%"], order[..12]);
        Assert.Equal(["West", "201", "8,040", "5%"], order[^4..]);
    }

    [Fact]
    public void InvoiceRowsStayTogether()
    {
        string[] order = Ordered(Shuffle(FinalLines("invoice")));
        Assert.Equal("Example Supplies Ltd.", order[0]);
        AssertFollows(order, "Item", "Qty", "Price", "Amount");
        AssertFollows(order, "Printer paper A4, 80 g", "10", "4.95", "49.50");
        AssertFollows(order, "Toner cartridge, black", "2", "89.00", "178.00");
        AssertFollows(order, "Total EUR 1,234.56", "VAT 25%");
        Assert.Equal("Page 1 of 1", order[^1]);
    }

    [Fact]
    public void UpsideDownPagesAreReadFromTheirTop()
    {
        // The classifier turned all three lines, so their quads run right to left on the page.
        List<RecognizedLine> lines = FinalLines("upside_down", turned: true);
        Assert.Equal(["Dear Sir or Madam,", "please find enclosed the signed contract.", "Kind regards, Anna Jensen"], Ordered(lines));
    }

    [Fact]
    public void PagesTurnedAQuarterAreReadAlongTheText()
    {
        // A paragraph turned 90 degrees counter-clockwise: lines read bottom to top, the first line is leftmost.
        var lines = new List<RecognizedLine>();
        for (int i = 0; i < 3; i++)
        {
            float x = 100 + (i * 40);
            var quad = new Quad(new(x, 900), new(x, 100), new(x + 30, 100), new(x + 30, 900));
            lines.Add(Line($"line {i}", quad));
        }

        Assert.Equal(1, ReadingOrder.DominantQuarterTurn(lines));
        Assert.Equal(["line 0", "line 1", "line 2"], Ordered(Shuffle(lines)));
    }

    [Fact]
    public void ShortLinesBesideEachOtherAreOneRow()
    {
        // Two labels on one row and a paragraph below them: the row is read left to right first.
        var lines = new List<RecognizedLine>
        {
            Line("Name:", Box(50, 50, 150, 80)),
            Line("Jane Doe", Box(400, 52, 560, 82)),
            Line("A paragraph of running text below the row.", Box(50, 120, 900, 150)),
        };
        Assert.Equal(["Name:", "Jane Doe", "A paragraph of running text below the row."], Ordered(Shuffle(lines)));
    }

    [Fact]
    public void PageTextAndConfidence()
    {
        var lines = new List<RecognizedLine>
        {
            Line("second", Box(10, 100, 200, 130), 0.5f),
            Line("first", Box(10, 10, 200, 40), 1f),
        };
        PageResult page = PageAssembler.Assemble(300, 300, 300, lines);
        Assert.Equal("first\nsecond", page.Text);
        Assert.Equal((5 * 1f + 6 * 0.5f) / 11, page.Confidence, 0.0001f);
        Assert.Equal(0f, PageAssembler.Assemble(10, 10, 300, []).Confidence);
        Assert.Equal(string.Empty, PageAssembler.Assemble(10, 10, 300, []).Text);
    }

    [Fact]
    public void ManyOverlappingLinesAreOrderedInBoundedTime()
    {
        // 1,000 lines piled on each other, with no gap to cut at.
        var lines = new List<RecognizedLine>();
        var random = new Random(5);
        for (int i = 0; i < 1000; i++)
        {
            float x = random.Next(0, 1000), y = random.Next(0, 1000);
            lines.Add(Line(i.ToString(System.Globalization.CultureInfo.InvariantCulture), Box(x, y, x + random.Next(5, 800), y + random.Next(5, 300))));
        }

        var watch = System.Diagnostics.Stopwatch.StartNew();
        List<List<RecognizedLine>> blocks = ReadingOrder.Order(lines);
        Assert.Equal(1000, blocks.Sum(b => b.Count));
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5));
    }

    internal static RecognizedLine Line(string text, Quad quad, float confidence = 1) => new(text, confidence, quad, [], 0, false);

    internal static Quad Box(float x0, float y0, float x1, float y1) => new(new(x0, y0), new(x1, y0), new(x1, y1), new(x0, y1));

    private static string[] Ordered(IReadOnlyList<RecognizedLine> lines) =>
        [.. ReadingOrder.Order(lines).SelectMany(b => b).Select(l => l.Text)];

    private static List<RecognizedLine> Shuffle(List<RecognizedLine> lines)
    {
        var copy = new List<RecognizedLine>(lines);
        var random = new Random(42);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }

        return copy;
    }

    private static void AssertFollows(string[] order, params string[] sequence)
    {
        int start = Array.IndexOf(order, sequence[0]);
        Assert.True(start >= 0, $"'{sequence[0]}' is missing");
        for (int i = 1; i < sequence.Length; i++)
        {
            Assert.True(start + i < order.Length && order[start + i] == sequence[i], $"'{sequence[i]}' does not follow '{sequence[i - 1]}': {string.Join(" | ", order)}");
        }
    }

    /// <summary>rapidocr's final lines of a fixture as recognised lines, optionally with quads turned 180 degrees.</summary>
    private static List<RecognizedLine> FinalLines(string name, bool turned = false)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        var lines = new List<RecognizedLine>();
        foreach (JsonElement f in fx.Root.GetProperty("final").EnumerateArray())
        {
            var p = f.GetProperty("box").EnumerateArray().Select(e => new Vector2(e[0].GetSingle(), e[1].GetSingle())).ToArray();
            Quad quad = turned ? new Quad(p[2], p[3], p[0], p[1]) : new Quad(p[0], p[1], p[2], p[3]);
            lines.Add(Line(f.GetProperty("text").GetString()!, quad));
        }

        return lines;
    }
}
