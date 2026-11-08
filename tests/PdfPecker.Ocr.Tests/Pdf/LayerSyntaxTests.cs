using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Pdf;

namespace PdfPecker.Ocr.Tests.Pdf;

public sealed class LayerSyntaxTests
{
    private static readonly double[] Letter = { 0, 0, 612, 792 };

    [Fact]
    public void FontTablesAreConsistent()
    {
        var font = GlyphlessFont.Bytes.ToArray();
        Assert.Equal(0x00010000u, BinaryPrimitives.ReadUInt32BigEndian(font));
        int count = BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(4));
        var tables = new Dictionary<string, (int Offset, int Length)>();
        for (int i = 0; i < count; i++)
        {
            var entry = font.AsSpan(12 + 16 * i);
            string tag = Encoding.ASCII.GetString(entry[..4]);
            uint sum = BinaryPrimitives.ReadUInt32BigEndian(entry[4..]);
            int offset = (int)BinaryPrimitives.ReadUInt32BigEndian(entry[8..]), length = (int)BinaryPrimitives.ReadUInt32BigEndian(entry[12..]);
            Assert.Equal(0, offset % 4);
            var data = font.AsSpan(offset, (length + 3) & ~3).ToArray();
            if (tag == "head") BinaryPrimitives.WriteUInt32BigEndian(data.AsSpan(8), 0);
            Assert.Equal(sum, Checksum(data));
            tables.Add(tag, (offset, length));
        }
        Assert.Equal(new[] { "OS/2", "cmap", "glyf", "head", "hhea", "hmtx", "loca", "maxp", "name", "post" }, tables.Keys);
        Assert.Equal(0xB1B0AFBAu, Checksum(font));
        int glyphs = BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(tables["maxp"].Offset + 4));
        int metrics = BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(tables["hhea"].Offset + 34));
        Assert.Equal(3, glyphs);
        Assert.Equal(3, metrics);
        Assert.Equal(4 * glyphs, tables["hmtx"].Length);
        Assert.Equal(2 * (glyphs + 1), tables["loca"].Length);
        int hmtx = tables["hmtx"].Offset;
        // The widths PDF/A compares with the font dictionary's /DW 500 and the /W of zero-width codes.
        Assert.Equal(new[] { 500, 500, 0 }, Enumerable.Range(0, 3).Select(g => (int)BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(hmtx + 4 * g))));
        Assert.Equal(1000, BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(tables["head"].Offset + 18)));
    }

    private static uint Checksum(ReadOnlySpan<byte> data)
    {
        uint sum = 0;
        for (int i = 0; i + 4 <= data.Length; i += 4) sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(data[i..]));
        return sum;
    }

    [Fact]
    public void LayerUsesAnEmbeddedType0FontWithToUnicode()
    {
        var page = new OcrPage(0, 612, 792, 72, new[] { Line("Blåbær 𝔘 á", 72, 72) });
        var pdf = Inflated(SearchableLayer.Write(new[] { new LayerPage(page, 612, 792) }));
        Assert.Contains("/Subtype /Type0", pdf);
        Assert.Contains("/Encoding /Identity-H", pdf);
        Assert.Contains("/Subtype /CIDFontType2", pdf);
        Assert.Contains("/FontFile2 6 0 R", pdf);
        Assert.Contains("/ToUnicode 7 0 R", pdf);
        Assert.Contains("/DW 500", pdf);
        Assert.Contains("3 Tr", pdf);
        // The combining acute accent (code 10, after B l å b æ r, the space and 𝔘 a) is the one zero-width code.
        Assert.Contains("/W [ 10 10 0 ]", pdf);
        Assert.Contains("<0008> <D835DD18>", pdf);
        Assert.Contains("<0003> <00E5>", pdf);
    }

    [Theory]
    [InlineData("ar-SA")]
    [InlineData("fa-IR")]
    [InlineData("de-DE")]
    public void LayerBytesDoNotDependOnCulture(string culture)
    {
        var layer = new[] { new LayerPage(new OcrPage(0, 1000, 1300, 120, new[] { Line("-1.5 négatif", -3.25f, 9.125f), Line("x", 1e-7f, 999.5f) }), 600.5, 780.25) };
        var expected = SearchableLayer.Write(layer);
        var saved = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Assert.Equal(expected, SearchableLayer.Write(layer));
        }
        finally
        {
            CultureInfo.CurrentCulture = saved;
        }
        Assert.DoesNotContain("E-", Inflated(expected));
    }

    [Fact]
    public void HostileWordsNeverBreakThePdf()
    {
        var nan = new OcrQuad(new(float.NaN, 1), new(2, 3), new(4, 5), new(6, 7));
        var lines = new[]
        {
            new OcrLine("bad", nan, 1, new[] { new OcrWord("nan", nan, 1), new OcrWord("ok", OcrGeometry.FromBox(10, 10, 50, 20), 1) }),
            new OcrLine("huge", default, 1, new[] { new OcrWord("huge", OcrGeometry.FromBox(-3e38f, -1e30f, float.MaxValue, 1e25f), 1) }),
            new OcrLine("flat", default, 1, new[] { new OcrWord("flat", OcrGeometry.FromBox(10, 10, 50, 0), 1), new OcrWord("point", OcrGeometry.FromBox(10, 10, 0, 0), 1) }),
            new OcrLine("chars", default, 1, new[]
            {
                new OcrWord("", OcrGeometry.FromBox(10, 40, 50, 20), 1),
                new OcrWord("\0\t\n\r\u0001\u007F", OcrGeometry.FromBox(70, 40, 50, 20), 1),
                new OcrWord("\uD800x\uDC00", OcrGeometry.FromBox(130, 40, 50, 20), 1),
                new OcrWord("(paren)\\ <hex>", OcrGeometry.FromBox(190, 40, 50, 20), 1),
                new OcrWord("́‍", OcrGeometry.FromBox(250, 40, 50, 20), 1),
            }),
            new OcrLine("no words <&>", OcrGeometry.FromBox(10, 80, 200, 20), 1, Array.Empty<OcrWord>()),
            new OcrLine("", OcrGeometry.FromBox(10, 120, 200, 20), 1, Array.Empty<OcrWord>()),
        };
        var pages = new[]
        {
            new OcrPage(0, 612, 792, 72, lines),
            new OcrPage(1, 0, -5, float.NaN, lines),
            new OcrPage(2, int.MaxValue, 1, float.PositiveInfinity, lines),
        };
        var scan = RawPdf.Build(Enumerable.Repeat(new RawPdf.Page(Letter, Content: RawPdf.Shapes(612, 792)), 3).ToArray());
        int call = 0;
        var result = Ocr.Pdf.SearchablePdf.MakeSearchable(scan, new OcrOptions { Dpi = 72, PageMode = OcrPageMode.Force },
            new DelegateRecognizer(() => pages[call++]), default);

        var text = PdfText.ExtractPages(result);
        Assert.Equal(3, text.Count);
        var first = text[0].Text;
        Assert.Contains("ok", first);
        Assert.Contains("�x�", first);
        Assert.Contains("(paren)\\ <hex>", first);
        Assert.Contains("no words <&>", first);
        Assert.DoesNotContain("nan", first);
        Assert.DoesNotContain("flat", first);
        Assert.DoesNotContain('\0', first);
        Assert.Equal(Pixels.Render(scan, 36), Pixels.Render(result, 36));
    }

    [Fact]
    public void ManyDistinctCharactersShareTheReplacementCodeWhenCodesRunOut()
    {
        var map = new SearchableLayer.CidMap();
        var codes = new List<int>();
        var text = new StringBuilder();
        // 70,000 distinct characters that are neither controls nor surrogates.
        for (int c = 0x100; text.Length < 140_000; c++)
        {
            if (c is >= 0xD800 and <= 0xDFFF or 0xFFFD) continue;
            text.Append(char.ConvertFromUtf32(c));
            if (c > 0x10000 && text.Length >= 140_000) break;
        }
        map.Encode(text.ToString(), codes);
        Assert.Equal(SearchableLayer.CidMap.MaxCodes, map.Count);
        Assert.All(codes, c => Assert.InRange(c, 1, 0xFFFF));
        int replacement = codes[^1];
        Assert.True(codes.Count(c => c == replacement) > 1000);
        Assert.NotEmpty(map.ToUnicode());
        Assert.NotEmpty(map.CidToGidMap());
    }

    [Fact]
    public void PixelsMapToDisplayedPoints()
    {
        var page = new OcrPage(0, 1000, 1300, 120, Array.Empty<OcrLine>());
        var g = new PageGeometry(600, 780, page);
        Assert.Equal((0.0, 780.0), g.ToPoints(new OcrPoint(0, 0)));
        Assert.Equal((600.0, 0.0), g.ToPoints(new OcrPoint(1000, 1300)));
        Assert.Equal((300.0, 390.0), g.ToPoints(new OcrPoint(500, 650)));
        // By resolution from the top-left corner, for an image smaller than its page: 120 dpi is 0.6 points per pixel.
        var byDpi = new PageGeometry(600, 900, page, byDpi: true);
        Assert.Equal((60.0, 840.0), byDpi.ToPoints(new OcrPoint(100, 100)));
        // A stamp page drawn half size takes half the points.
        Assert.Equal((150.0, 195.0), new PageGeometry(300, 390, page, shrink: 2).ToPoints(new OcrPoint(500, 650)));
        Assert.Equal((30.0, 420.0), new PageGeometry(300, 450, page, byDpi: true, shrink: 2).ToPoints(new OcrPoint(100, 100)));
    }

    [Fact]
    public void NumbersAreWrittenWithoutExponentsAndWithinRange()
    {
        var b = new PdfBuffer();
        foreach (var v in new[] { 0.0, -0.0, 1e-9, -1e-9, 1.23456789, -7.5, 1e300, -1e300, double.NaN, double.PositiveInfinity, 12345.6789 })
            b.Real(v).Space();
        Assert.Equal("0 0 0 0 1.2346 -7.5 32767 -32767 0 32767 12345.6789 ", Encoding.ASCII.GetString(b.Written));
    }

    private static OcrLine Line(string text, float x, float y)
    {
        var words = text.Split(' ').Select((w, i) => new OcrWord(w, OcrGeometry.FromBox(x + 60 * i, y, 50, 12), 1)).ToArray();
        return new OcrLine(text, OcrGeometry.FromBox(x, y, 60 * words.Length, 12), 1, words);
    }

    /// <summary>The PDF as Latin-1 text with its Flate streams inflated, for looking at its syntax.</summary>
    private static string Inflated(byte[] pdf)
    {
        var text = Encoding.Latin1.GetString(pdf);
        var sb = new StringBuilder();
        int pos = 0;
        while (true)
        {
            int start = text.IndexOf(">>\nstream\n", pos, StringComparison.Ordinal);
            if (start < 0) break;
            int end = text.IndexOf("\nendstream", start, StringComparison.Ordinal);
            sb.Append(text, pos, start + 10 - pos);
            var data = pdf.AsSpan(start + 10, end - start - 10).ToArray();
            try
            {
                using var z = new ZLibStream(new MemoryStream(data), CompressionMode.Decompress);
                var output = new MemoryStream();
                z.CopyTo(output);
                sb.Append(Encoding.Latin1.GetString(output.ToArray()));
            }
            catch (InvalidDataException)
            {
                sb.Append("[binary]");
            }
            pos = end;
        }
        return sb.Append(text, pos, text.Length - pos).ToString();
    }

    private sealed class DelegateRecognizer : IPageRecognizer
    {
        private readonly Func<OcrPage> _next;

        public DelegateRecognizer(Func<OcrPage> next) => _next = next;

        public OcrPage Recognize(PagePixels pixels, int pageIndex, float dpi, OcrOptions options, CancellationToken cancellationToken) => _next();
    }
}
