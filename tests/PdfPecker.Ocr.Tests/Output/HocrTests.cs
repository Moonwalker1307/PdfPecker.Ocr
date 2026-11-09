using System.Globalization;
using System.Xml;
using System.Xml.Linq;

namespace PdfPecker.Ocr.Tests.Output;

public sealed class HocrTests
{
    private static readonly XNamespace Xhtml = "http://www.w3.org/1999/xhtml";

    private static OcrWord Word(string text, float x, float y, float w, float h, float confidence = 0.9f)
        => new(text, OcrGeometry.FromBox(x, y, w, h), confidence);

    private static OcrLine Line(params OcrWord[] words)
    {
        var quad = new OcrQuad(words[0].Quad.TopLeft, words[^1].Quad.TopRight, words[^1].Quad.BottomRight, words[0].Quad.BottomLeft);
        return new OcrLine(string.Join(' ', words.Select(w => w.Text)), quad, words.Average(w => w.Confidence), words);
    }

    private static OcrPage SamplePage(int index = 0) => new(index, 2480, 3508, 300, new[]
    {
        Line(Word("Hello", 100.4f, 200.6f, 150.2f, 40, 0.97f), Word("World", 270, 200, 160, 40, 0.88f)),
        Line(Word("Price:", 100, 300, 120, 30), Word("<5", 230, 300, 50, 30), Word("&", 290, 300, 20, 30), Word("\"q\"", 320, 300, 40, 30)),
    });

    private static XDocument Parse(string hocr)
    {
        // The DTD is named but not fetched: hOCR readers parse it as XML without loading the DTD.
        var settings = new XmlReaderSettings { DtdProcessing = DtdProcessing.Ignore, XmlResolver = null };
        using var reader = XmlReader.Create(new StringReader(hocr), settings);
        return XDocument.Load(reader, LoadOptions.PreserveWhitespace);
    }

    private static IEnumerable<XElement> ByClass(XDocument doc, string cls)
        => doc.Descendants().Where(e => (string?)e.Attribute("class") == cls);

    [Fact]
    public void PageHasLinesAndWordsWithBoxesAndConfidences()
    {
        var doc = Parse(SamplePage(4).ToHocr());
        Assert.Equal(Xhtml + "html", doc.Root!.Name);
        var page = Assert.Single(ByClass(doc, "ocr_page"));
        Assert.Equal("bbox 0 0 2480 3508; ppageno 4; scan_res 300 300", (string?)page.Attribute("title"));
        var lines = ByClass(doc, "ocr_line").ToList();
        Assert.Equal(2, lines.Count);
        Assert.Equal("bbox 100 200 430 241; x_wconf 93", (string?)lines[0].Attribute("title"));
        var words = ByClass(doc, "ocrx_word").ToList();
        Assert.Equal(new[] { "Hello", "World", "Price:", "<5", "&", "\"q\"" }, words.Select(w => w.Value));
        // Boxes cover the word with whole pixels: down for the top-left corner, up for the bottom-right.
        Assert.Equal("bbox 100 200 251 241; x_wconf 97", (string?)words[0].Attribute("title"));
        Assert.Equal("bbox 270 200 430 240; x_wconf 88", (string?)words[1].Attribute("title"));
        Assert.Equal(new[] { "word_1_1_1", "word_1_1_2", "word_1_2_1" }, words.Take(3).Select(w => (string?)w.Attribute("id")));
        Assert.All(words, w => Assert.Equal("ocr_line", (string?)w.Parent!.Attribute("class")));
        var meta = doc.Descendants(Xhtml + "meta").ToDictionary(m => (string?)m.Attribute("name") ?? (string)m.Attribute("http-equiv")!, m => (string?)m.Attribute("content"));
        Assert.Equal("ocr_page ocr_line ocrx_word ocrp_wconf", meta["ocr-capabilities"]);
        Assert.StartsWith("PdfPecker.Ocr ", meta["ocr-system"]);
    }

    [Fact]
    public void DocumentHasOnePageElementPerPage()
    {
        var document = new OcrDocument(new[] { SamplePage(0), new OcrPage(1, 100, 50, 72, Array.Empty<OcrLine>()), SamplePage(2) });
        var doc = Parse(document.ToHocr());
        var pages = ByClass(doc, "ocr_page").ToList();
        Assert.Equal(new[] { "page_1", "page_2", "page_3" }, pages.Select(p => (string?)p.Attribute("id")));
        Assert.Empty(pages[1].Elements());
        Assert.Equal("3", doc.Descendants(Xhtml + "meta").Single(m => (string?)m.Attribute("name") == "ocr-number-of-pages").Attribute("content")!.Value);
        Assert.Equal(new[] { "word_3_1_1" }, ByClass(doc, "ocrx_word").Where(w => w.Value == "Hello").Skip(1).Select(w => (string?)w.Attribute("id")));
    }

    [Fact]
    public void RotatedLineHasTextAngle()
    {
        // A line that reads upwards, its quad turned 90 degrees counter-clockwise.
        var quad = new OcrQuad(new OcrPoint(100, 500), new OcrPoint(100, 300), new OcrPoint(140, 300), new OcrPoint(140, 500));
        var page = new OcrPage(0, 1000, 1000, 300, new[] { new OcrLine("up", quad, 1, new[] { new OcrWord("up", quad, 1) }) });
        var line = Assert.Single(ByClass(Parse(page.ToHocr()), "ocr_line"));
        Assert.Equal("bbox 100 300 140 500; textangle 90; x_wconf 100", (string?)line.Attribute("title"));
    }

    [Fact]
    public void HostileTextAndBoxesStayWellFormed()
    {
        var nan = new OcrQuad(new OcrPoint(float.NaN, 1), new OcrPoint(float.PositiveInfinity, 2), new OcrPoint(3, float.NegativeInfinity), new OcrPoint(4, 5));
        var huge = OcrGeometry.FromBox(-1e30f, -5, 3e38f, 1e20f);
        var words = new[]
        {
            new OcrWord("a\0b\u0001c\u001F", nan, float.NaN),
            new OcrWord("\uD800lone\uDFFF", huge, -3),
            new OcrWord("￾￿", default, 7),
            new OcrWord("", default, 0.5f),
            new OcrWord("😀]]>", huge, 0.5f),
        };
        var page = new OcrPage(-1, -10, 20, float.NaN, new[] { new OcrLine("x", nan, float.NaN, words), new OcrLine("only text <b>", huge, 1, Array.Empty<OcrWord>()) });
        var doc = Parse(page.ToHocr());
        var texts = ByClass(doc, "ocrx_word").Select(w => w.Value).ToList();
        Assert.Equal(new[] { "a b c ", "�lone�", "  ", "", "😀]]>", "only text <b>" }, texts);
        foreach (var e in ByClass(doc, "ocrx_word").Concat(ByClass(doc, "ocr_line")))
        {
            var box = ((string)e.Attribute("title")!).Split(';')[0].Split(' ').Skip(1).Select(int.Parse).ToArray();
            Assert.All(box, v => Assert.InRange(v, 0, 20));
            Assert.True(box[2] >= box[0] && box[3] >= box[1]);
        }
        Assert.Equal("bbox 0 0 0 20; ppageno 0", (string?)ByClass(doc, "ocr_page").Single().Attribute("title"));
    }

    [Theory]
    [InlineData("ar-SA")]
    [InlineData("fa-IR")]
    [InlineData("de-DE")]
    public void OutputDoesNotDependOnCulture(string culture)
    {
        var expected = SamplePage().ToHocr();
        var saved = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Assert.Equal(expected, SamplePage().ToHocr());
            Assert.Equal("Page 0: 2480 x 3508 px, 2 lines", SamplePage().ToString());
        }
        finally
        {
            CultureInfo.CurrentCulture = saved;
        }
    }

    [Fact]
    public void TextJoinsLinesAndPages()
    {
        var page = SamplePage();
        Assert.Equal("Hello World\nPrice: <5 & \"q\"", page.Text);
        Assert.Same(page.Text, page.Text);
        var empty = new OcrPage(1, 10, 10, 300, Array.Empty<OcrLine>());
        Assert.Equal("", empty.Text);
        Assert.Equal("Hello World\nPrice: <5 & \"q\"\f\fHello World\nPrice: <5 & \"q\"", new OcrDocument(new[] { page, empty, page }).Text);
        Assert.Equal("", new OcrDocument(Array.Empty<OcrPage>()).Text);
    }

    [Fact]
    public void ConfidenceIsWeightedByCharacters()
    {
        var a = new OcrLine("abcdefghij", default, 1f, Array.Empty<OcrWord>());
        var b = new OcrLine("xy", default, 0.4f, Array.Empty<OcrWord>());
        var broken = new OcrLine("zzz", default, float.NaN, Array.Empty<OcrWord>());
        Assert.Equal(0.9f, new OcrPage(0, 1, 1, 300, new[] { a, b, broken }).Confidence, 5);
        Assert.Equal(0f, new OcrPage(0, 1, 1, 300, Array.Empty<OcrLine>()).Confidence);
    }

    [Fact]
    public void ResultsCannotBeChangedThroughTheirLists()
    {
        var words = new List<OcrWord> { Word("a", 0, 0, 1, 1) };
        var line = new OcrLine("a", default, 1, words);
        words.Clear();
        Assert.Single(line.Words);
        Assert.False(line.Words is OcrWord[]);
        Assert.Throws<NotSupportedException>(() => ((IList<OcrWord>)line.Words).Add(Word("b", 0, 0, 1, 1)));
    }

    [Fact]
    public void BoxIsTheBoundsOfTheQuad()
    {
        var quad = new OcrQuad(new OcrPoint(10, 20), new OcrPoint(50, 5), new OcrPoint(55, 25), new OcrPoint(15, 40));
        Assert.Equal(new OcrBox(10, 5, 45, 35), new OcrWord("w", quad, 1).Box);
        Assert.Equal(new OcrBox(4, 5, 0, 0), OcrGeometry.Bounds(new OcrQuad(new(float.NaN, 1), new(4, float.NaN), new(4, 5), new(float.NaN, float.NaN))));
        Assert.Equal(default, OcrGeometry.Bounds(new OcrQuad(new(float.NaN, 1), new(1, float.NaN), default, default) with { BottomRight = new(float.NaN, 0), BottomLeft = new(0, float.NaN) }));
    }
}
