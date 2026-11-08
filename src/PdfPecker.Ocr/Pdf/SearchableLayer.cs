using System.Text;

namespace PdfPecker.Ocr.Pdf;

/// <summary>One page of the text layer: what OCR read on it, if anything, and its displayed size in points.</summary>
/// <param name="Page">The OCR result, or null for a page that keeps no text layer (an empty stamp page).</param>
/// <param name="Width">The page's displayed width in points.</param>
/// <param name="Height">The page's displayed height in points.</param>
/// <param name="ScaleByDpi">
/// Place words by the page's resolution from the top-left corner instead of scaling the pixels to the page size: for
/// pages that may be larger than the image they show.
/// </param>
internal readonly record struct LayerPage(OcrPage? Page, double Width, double Height, bool ScaleByDpi = false);

/// <summary>
/// Writes the invisible text layer as a PDF of its own, one page per page of the scanned document, to be put on the
/// scans with <see cref="PdfStamper"/>.
/// </summary>
/// <remarks>
/// <para>
/// Each word is drawn in text render mode 3 (neither filled nor stroked), so nothing shows, but viewers select, copy
/// and search it, and text extraction reads it. The font is <see cref="GlyphlessFont"/> as a Type0 font with
/// Identity-H encoding: each character gets a 16-bit code (a CID) the first time it appears, the ToUnicode CMap maps the
/// code back to the character (as a surrogate pair beyond the Basic Multilingual Plane), and CIDToGIDMap sends every
/// code to an empty glyph: one 500 units wide, or one of no width for combining marks and format characters (such as
/// the joiners inside emoji sequences), so extraction keeps a mark on its letter. Numbering characters in order of appearance rather than using their UTF-16 code
/// units keeps every character one code and one glyph, also beyond the BMP, and keeps the maps small.
/// </para>
/// <para>
/// A word is placed with a text matrix at its quad's bottom-left corner, turned to the quad's direction, with the font
/// size the quad's height and horizontal scaling (<c>Tz</c>) that makes its glyphs exactly as wide as the quad. The
/// baseline sits a fifth of the height above the bottom edge, where the font's descent puts it, so a selection covers
/// the word's box. A space follows every word but the last of a line, so extracted text has its word breaks.
/// </para>
/// <para>
/// The layer is meant to keep a PDF/A file valid: the font is embedded with a ToUnicode map and glyph widths that match
/// its <c>/DW</c>, and there is no transparency.
/// </para>
/// </remarks>
internal static class SearchableLayer
{
    private const string FontResource = "OCR";

    /// <summary>Writes the layer for the pages.</summary>
    /// <exception cref="ArgumentException">There are no pages.</exception>
    public static byte[] Write(IReadOnlyList<LayerPage> pages, CancellationToken cancellationToken = default)
    {
        if (pages.Count == 0) throw new ArgumentException("The text layer needs at least one page.", nameof(pages));
        var cids = new CidMap();
        var contents = new byte[]?[pages.Count];
        for (int i = 0; i < pages.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (pages[i].Page is { } page) contents[i] = PageContent(page, Size(pages[i]), pages[i].ScaleByDpi, cids);
        }

        var pdf = new PdfBuffer(4096);
        var offsets = new List<int> { 0 };
        int fontId = cids.Count > 0 ? 3 : 0;
        int firstPageId = fontId != 0 ? 9 : 3;
        // Each page is two objects, the page and its content stream (written even when empty, for simplicity).
        int objectCount = firstPageId + 2 * pages.Count;

        pdf.Ascii("%PDF-1.7\n%").Bytes([0xE2, 0xE3, 0xCF, 0xD3]).Ascii("\n");
        Begin(pdf, offsets, 1).Ascii("<< /Type /Catalog /Pages 2 0 R >>");
        End(pdf);
        Begin(pdf, offsets, 2).Ascii("<< /Type /Pages /Count ").Int(pages.Count).Ascii(" /Kids [");
        for (int i = 0; i < pages.Count; i++) pdf.Space().Int(firstPageId + 2 * i).Ascii(" 0 R");
        pdf.Ascii(" ] >>");
        End(pdf);

        if (fontId != 0) WriteFont(pdf, offsets, cids);

        for (int i = 0; i < pages.Count; i++)
        {
            int pageId = firstPageId + 2 * i, contentId = pageId + 1;
            var (w, h, _) = Size(pages[i]);
            Begin(pdf, offsets, pageId).Ascii("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ").Real(w).Space().Real(h).Ascii("] /Resources <<");
            if (contents[i] != null && fontId != 0) pdf.Ascii(" /Font << /").Ascii(FontResource).Ascii(" 3 0 R >>");
            pdf.Ascii(" >> /Contents ").Int(contentId).Ascii(" 0 R >>");
            End(pdf);
            Stream(pdf, offsets, contentId, "", contents[i] ?? Array.Empty<byte>());
        }

        int xref = pdf.Length;
        pdf.Ascii("xref\n0 ").Int(objectCount).Ascii("\n0000000000 65535 f \n");
        for (int id = 1; id < objectCount; id++)
        {
            string offset = offsets[id].ToString("D10", System.Globalization.CultureInfo.InvariantCulture);
            pdf.Ascii(offset).Ascii(" 00000 n \n");
        }
        pdf.Ascii("trailer\n<< /Size ").Int(objectCount).Ascii(" /Root 1 0 R >>\nstartxref\n").Int(xref).Ascii("\n%%EOF\n");
        return pdf.ToArray();
    }

    /// <summary>The largest page side written, PDF's own limit.</summary>
    private const double MaxSide = 14_400;

    /// <summary>
    /// The stamp page's size, and the factor it was shrunk by: a page larger than PDF allows (a page with a user unit,
    /// for example) gets a stamp of the same shape within the limit, which <see cref="PdfStamper"/> scales up to fit it.
    /// A size that is not a usable number gives a letter page.
    /// </summary>
    private static (double W, double H, double Shrink) Size(LayerPage page)
    {
        if (!(double.IsFinite(page.Width) && double.IsFinite(page.Height) && page.Width > 0.01 && page.Height > 0.01)) return (612, 792, 1);
        double shrink = Math.Max(1, Math.Max(page.Width, page.Height) / MaxSide);
        return (Math.Max(0.02, page.Width / shrink), Math.Max(0.02, page.Height / shrink), shrink);
    }

    /// <summary>The content stream of one page, deflated, or null when no word could be placed.</summary>
    private static byte[]? PageContent(OcrPage page, (double W, double H, double Shrink) size, bool scaleByDpi, CidMap cids)
    {
        var geometry = new PageGeometry(size.W, size.H, page, scaleByDpi, size.Shrink);
        var content = new PdfBuffer(64 + 96 * page.Lines.Count * 4);
        var codes = new List<int>(32);
        bool any = false;
        content.Ascii("q\n3 Tr\n");
        foreach (var line in page.Lines)
        {
            bool open = false;
            if (line.Words.Count == 0)
            {
                // A line without words is placed as one word, its spaces included.
                any |= Word(content, ref open, geometry, line.Quad, line.Text, last: true, cids, codes);
            }
            else
            {
                for (int w = 0; w < line.Words.Count; w++)
                {
                    var word = line.Words[w];
                    any |= Word(content, ref open, geometry, word.Quad, word.Text, w == line.Words.Count - 1, cids, codes);
                }
            }
            if (open) content.Ascii("ET\n");
        }
        content.Ascii("Q\n");
        return any ? content.Deflate() : null;
    }

    /// <summary>Writes one word, opening the line's text object first if needed; false when the word cannot be placed.</summary>
    private static bool Word(PdfBuffer content, ref bool open, in PageGeometry geometry, in OcrQuad quad, string text, bool last,
        CidMap cids, List<int> codes)
    {
        codes.Clear();
        cids.Encode(text, codes);
        if (codes.Count == 0 || !OcrGeometry.IsFinite(quad)) return false;

        var (tlx, tly) = geometry.ToPoints(quad.TopLeft);
        var (trx, try_) = geometry.ToPoints(quad.TopRight);
        var (brx, bry) = geometry.ToPoints(quad.BottomRight);
        var (blx, bly) = geometry.ToPoints(quad.BottomLeft);

        // The direction is the mean of the top and bottom edges, the up vector the mean of the left and right edges.
        double dx = (trx - tlx + brx - blx) / 2, dy = (try_ - tly + bry - bly) / 2;
        double width = Math.Sqrt(dx * dx + dy * dy);
        if (!(width >= 0.01) || !double.IsFinite(width)) return false;
        double cos = dx / width, sin = dy / width;
        double ux = (tlx - blx + trx - brx) / 2, uy = (tly - bly + try_ - bry) / 2;
        // The height is measured across the direction, so a slanted quad gets the height of its text.
        double height = Math.Abs(cos * uy - sin * ux);
        if (!(height >= 0.01) || !double.IsFinite(height)) return false;

        double descent = -(double)GlyphlessFont.Descent / (GlyphlessFont.Ascent - GlyphlessFont.Descent);
        // The baseline starts above the bottom-left corner by the descent, measured along the up direction.
        double ox = blx - sin * height * descent, oy = bly + cos * height * descent;
        int wide = 0;
        foreach (int code in codes)
            if (!cids.IsZeroWidth(code)) wide++;
        double glyphs = wide * (GlyphlessFont.Advance / (double)GlyphlessFont.UnitsPerEm);
        // A word of marks alone has no width to stretch.
        double scale = wide > 0 ? 100 * width / (glyphs * height) : 100;

        if (!open)
        {
            content.Ascii("BT\n");
            open = true;
        }
        content.Byte((byte)'/').Ascii(FontResource).Space().Real(height).Ascii(" Tf ").Real(scale).Ascii(" Tz ")
            .Real(cos).Space().Real(sin).Space().Real(-sin).Space().Real(cos).Space().Real(ox).Space().Real(oy).Ascii(" Tm <");
        foreach (int code in codes) content.Hex4(code);
        if (!last) content.Hex4(cids.Space);
        content.Ascii("> Tj\n");
        return true;
    }

    private static void WriteFont(PdfBuffer pdf, List<int> offsets, CidMap cids)
    {
        // 3: the Type0 font, 4: its CIDFont, 5: the descriptor, 6: the font file, 7: ToUnicode, 8: CIDToGIDMap.
        Begin(pdf, offsets, 3).Ascii("<< /Type /Font /Subtype /Type0 /BaseFont /").Ascii(GlyphlessFont.Name)
            .Ascii(" /Encoding /Identity-H /DescendantFonts [4 0 R] /ToUnicode 7 0 R >>");
        End(pdf);
        Begin(pdf, offsets, 4).Ascii("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /").Ascii(GlyphlessFont.Name)
            .Ascii(" /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor 5 0 R /DW ")
            .Int(GlyphlessFont.Advance);
        cids.WriteZeroWidths(pdf);
        pdf.Ascii(" /CIDToGIDMap 8 0 R >>");
        End(pdf);
        Begin(pdf, offsets, 5).Ascii("<< /Type /FontDescriptor /FontName /").Ascii(GlyphlessFont.Name)
            .Ascii(" /Flags 5 /FontBBox [0 ").Int(GlyphlessFont.Descent).Space().Int(GlyphlessFont.Advance).Space().Int(GlyphlessFont.Ascent)
            .Ascii("] /ItalicAngle 0 /Ascent ").Int(GlyphlessFont.Ascent).Ascii(" /Descent ").Int(GlyphlessFont.Descent)
            .Ascii(" /CapHeight ").Int(GlyphlessFont.Ascent).Ascii(" /StemV 80 /FontFile2 6 0 R >>");
        End(pdf);
        var font = GlyphlessFont.Bytes;
        Stream(pdf, offsets, 6, " /Length1 " + font.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), PdfBuffer.Deflate(font));
        Stream(pdf, offsets, 7, "", cids.ToUnicode());
        Stream(pdf, offsets, 8, "", cids.CidToGidMap());
    }

    private static PdfBuffer Begin(PdfBuffer pdf, List<int> offsets, int id)
    {
        while (offsets.Count <= id) offsets.Add(0);
        offsets[id] = pdf.Length;
        return pdf.Int(id).Ascii(" 0 obj\n");
    }

    private static void End(PdfBuffer pdf) => pdf.Ascii("\nendobj\n");

    /// <summary>Writes a stream object whose data is already deflated (or empty).</summary>
    private static void Stream(PdfBuffer pdf, List<int> offsets, int id, string extra, byte[] deflated)
    {
        Begin(pdf, offsets, id).Ascii("<< /Length ").Int(deflated.Length);
        if (deflated.Length > 0) pdf.Ascii(" /Filter /FlateDecode");
        pdf.Ascii(extra).Ascii(" >>\nstream\n").Bytes(deflated).Ascii("\nendstream");
        End(pdf);
    }

    /// <summary>
    /// The codes of the layer's characters: each character gets the next free code when it first appears, from 1 (code
    /// 0 is the font's .notdef). Control characters are left out and lone surrogates become U+FFFD, so whatever a
    /// recognizer returns, the layer holds only characters a ToUnicode map can name.
    /// </summary>
    internal sealed class CidMap
    {
        /// <summary>The most codes: 16 bits, without code 0.</summary>
        public const int MaxCodes = 0xFFFF;

        private readonly Dictionary<int, int> _codes = new();
        private readonly List<int> _chars = new() { 0 };
        private readonly List<int> _zeroWidth = new();

        /// <summary>The number of codes given out.</summary>
        public int Count => _chars.Count - 1;

        /// <summary>The code of the space between words.</summary>
        public int Space => Code(' ');

        /// <summary>Appends the codes of the text's characters.</summary>
        public void Encode(string text, List<int> codes)
        {
            var span = text.AsSpan();
            while (!span.IsEmpty)
            {
                Rune.DecodeFromUtf16(span, out var rune, out int used);
                span = span[used..];
                // A lone surrogate decodes as U+FFFD; control characters (tabs and line breaks too) have no place in a word.
                if (Rune.IsControl(rune)) continue;
                codes.Add(Code(rune.Value));
            }
        }

        private int Code(int c)
        {
            if (_codes.TryGetValue(c, out int code)) return code;
            if (Count >= MaxCodes - 1 && c != 0xFFFD)
            {
                // Past 65,534 distinct characters the rest share the code of U+FFFD, which keeps one code free for it.
                return Code(0xFFFD);
            }
            code = _chars.Count;
            _chars.Add(c);
            _codes.Add(c, code);
            if (TakesNoRoom(c)) _zeroWidth.Add(code);
            return code;
        }

        /// <summary>Whether a code's character is drawn with the zero-width glyph.</summary>
        public bool IsZeroWidth(int code) => _zeroWidth.Count > 0 && _zeroWidth.BinarySearch(code) >= 0;

        /// <summary>Marks and format characters sit on the character before them and take no room of their own.</summary>
        private static bool TakesNoRoom(int c) => Rune.GetUnicodeCategory(new Rune(c)) is
            System.Globalization.UnicodeCategory.NonSpacingMark or System.Globalization.UnicodeCategory.EnclosingMark or System.Globalization.UnicodeCategory.Format;

        /// <summary>Writes the <c>/W</c> entry that gives the zero-width codes their width of 0, when there are any.</summary>
        public void WriteZeroWidths(PdfBuffer pdf)
        {
            if (_zeroWidth.Count == 0) return;
            pdf.Ascii(" /W [");
            // Runs of consecutive codes are written as one range: first last width.
            for (int i = 0; i < _zeroWidth.Count;)
            {
                int j = i;
                while (j + 1 < _zeroWidth.Count && _zeroWidth[j + 1] == _zeroWidth[j] + 1) j++;
                pdf.Space().Int(_zeroWidth[i]).Space().Int(_zeroWidth[j]).Ascii(" 0");
                i = j + 1;
            }
            pdf.Ascii(" ]");
        }

        /// <summary>The ToUnicode CMap, deflated.</summary>
        public byte[] ToUnicode()
        {
            var cmap = new PdfBuffer(256 + 24 * Count);
            cmap.Ascii("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n")
                .Ascii("/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n")
                .Ascii("/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n")
                .Ascii("1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n");
            for (int start = 1; start < _chars.Count; start += 100)
            {
                int end = Math.Min(start + 100, _chars.Count);
                cmap.Int(end - start).Ascii(" beginbfchar\n");
                for (int code = start; code < end; code++)
                {
                    cmap.Byte((byte)'<').Hex4(code).Ascii("> <");
                    int c = _chars[code];
                    if (c > 0xFFFF)
                    {
                        cmap.Hex4(0xD800 + ((c - 0x10000) >> 10)).Hex4(0xDC00 + ((c - 0x10000) & 0x3FF));
                    }
                    else cmap.Hex4(c);
                    cmap.Ascii(">\n");
                }
                cmap.Ascii("endbfchar\n");
            }
            cmap.Ascii("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n");
            return cmap.Deflate();
        }

        /// <summary>The CIDToGIDMap: glyph 0 for code 0 and one of the two empty glyphs for every other code, deflated.</summary>
        public byte[] CidToGidMap()
        {
            var map = new byte[2 * _chars.Count];
            for (int code = 1; code < _chars.Count; code++) map[2 * code + 1] = GlyphlessFont.WideGlyph;
            foreach (int code in _zeroWidth) map[2 * code + 1] = GlyphlessFont.ZeroWidthGlyph;
            return PdfBuffer.Deflate(map);
        }
    }
}
