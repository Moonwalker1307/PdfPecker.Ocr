using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>
/// Runs of dots, which the recogniser cannot count and the detector does not see as text: dot leaders between a title
/// and its page number, and runs of dots in a line, counted on the page (see <see cref="DotRuns"/>).
/// </summary>
[Collection("EndToEnd")]
public sealed class DotRunTests
{
    private static readonly string[] Contents =
    [
        "1 Indledning ........................................ 3",
        "Bilag .......... 41",
        "Se side 4 ... og 5 .... og 6 ......",
        "Version 1.2.3, kl. 9.30 og 10.0.0.1 ... osv.",
        "Navn: Jens i Aalborg? Ja; og så!",
    ];

    [Theory]
    [InlineData("serif", 12)]
    [InlineData("sans-serif", 11)]
    [InlineData("monospace", 10)]
    public void DotLeadersAndRunsAreCounted(string family, int points)
    {
        string html = TestPages.Document("170mm 60mm", $"p{{font:{points}pt/1.6 {family};margin:0;white-space:pre}}", TestPages.Paragraphs(string.Join('\n', Contents)));
        RenderedPage page = TestPages.Get("dots-" + family, html);
        OcrPage result = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);
        Assert.Equal(string.Join('\n', Contents), result.Text);
        foreach (OcrLine line in result.Lines)
        {
            Assert.Equal(line.Text, string.Join(' ', line.Words.Select(w => w.Text)));
        }

        // The leader is a word of its own, between the title and the page number.
        OcrLine first = result.Lines[0];
        OcrWord leader = first.Words[2];
        Assert.Equal(new string('.', 40), leader.Text);
        Assert.True(leader.Box.X > first.Words[1].Box.X + first.Words[1].Box.Width - 1 && leader.Box.X + leader.Box.Width < first.Words[3].Box.X + 1);
    }

    [Theory]
    [InlineData("Indledning .... . 3", new[] { 11, 17 })]
    [InlineData("lidt... og…", new[] { 4, 7, 10, 11 })]
    [InlineData("1.2.3 kl. 9.30", new int[0])]
    [InlineData(". . . .", new[] { 0, 7 })]
    [InlineData("a. b", new int[0])]
    public void TextRunsAreTwoOrMoreDotsOrAnEllipsis(string text, int[] bounds)
    {
        Assert.Equal(bounds, DotRuns.TextRuns(text).SelectMany(r => new[] { r.Start, r.End }));
    }
}
