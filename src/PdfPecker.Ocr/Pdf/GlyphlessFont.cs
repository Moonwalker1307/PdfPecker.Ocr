using System.Buffers.Binary;
using System.Text;

namespace PdfPecker.Ocr.Pdf;

/// <summary>
/// A TrueType font with no visible glyphs, for the invisible text layer: every character of the layer is drawn with
/// glyph 1, which is empty and 500 units wide, so a word can be stretched to its box with horizontal scaling alone.
/// Combining marks and other characters that take no room of their own use glyph 2, which is empty and has no width,
/// so text extraction keeps them on the letter before them.
/// </summary>
/// <remarks>
/// The font is built in code rather than shipped as a file, so there is nothing to license and nothing to keep in
/// step. It has the tables PDF readers and PDF/A validators look for (head, hhea, maxp, OS/2, hmtx, cmap, loca, glyf,
/// name and post), with widths that match the <c>/DW 500</c> of the font dictionary, which PDF/A requires. The bytes
/// never change, so they are built once.
/// </remarks>
internal static class GlyphlessFont
{
    /// <summary>The font's name in the PDF and in its name table.</summary>
    public const string Name = "OcrGlyphless";

    /// <summary>Font units per em: with 1000, glyph widths read the same as PDF widths.</summary>
    public const int UnitsPerEm = 1000;

    /// <summary>The width of the glyph for characters that take room, in font units.</summary>
    public const int Advance = 500;

    /// <summary>The glyph for characters that take room.</summary>
    public const int WideGlyph = 1;

    /// <summary>The glyph, of no width, for combining marks and format characters.</summary>
    public const int ZeroWidthGlyph = 2;

    /// <summary>The ascent in font units: the text layer puts the baseline this share of the box above its bottom.</summary>
    public const int Ascent = 800;

    /// <summary>The descent in font units, negative below the baseline.</summary>
    public const int Descent = -200;

    private static readonly Lazy<byte[]> s_bytes = new(Build);

    /// <summary>The font file.</summary>
    public static ReadOnlySpan<byte> Bytes => s_bytes.Value;

    private static byte[] Build()
    {
        var tables = new (string Tag, byte[] Data)[]
        {
            ("OS/2", Os2()),
            ("cmap", Cmap()),
            ("glyf", new byte[4]),
            ("head", Head()),
            ("hhea", Hhea()),
            ("hmtx", Hmtx()),
            ("loca", new byte[8]),
            ("maxp", Maxp()),
            ("name", NameTable()),
            ("post", Post()),
        };

        int count = tables.Length;
        int offset = 12 + 16 * count;
        int total = offset;
        foreach (var (_, data) in tables) total += Pad(data.Length);
        var font = new byte[total];
        var span = font.AsSpan();

        // The offset table: 8 is the largest power of two not above the 10 tables, which sets the search fields.
        BinaryPrimitives.WriteUInt32BigEndian(span, 0x00010000);
        BinaryPrimitives.WriteUInt16BigEndian(span[4..], (ushort)count);
        BinaryPrimitives.WriteUInt16BigEndian(span[6..], 8 * 16);
        BinaryPrimitives.WriteUInt16BigEndian(span[8..], 3);
        BinaryPrimitives.WriteUInt16BigEndian(span[10..], (ushort)(count * 16 - 8 * 16));

        int headOffset = 0;
        for (int i = 0; i < count; i++)
        {
            var (tag, data) = tables[i];
            var entry = span.Slice(12 + 16 * i, 16);
            Encoding.ASCII.GetBytes(tag, entry);
            BinaryPrimitives.WriteUInt32BigEndian(entry[4..], Checksum(data));
            BinaryPrimitives.WriteUInt32BigEndian(entry[8..], (uint)offset);
            BinaryPrimitives.WriteUInt32BigEndian(entry[12..], (uint)data.Length);
            data.CopyTo(span[offset..]);
            if (tag == "head") headOffset = offset;
            offset += Pad(data.Length);
        }

        // The whole font's checksum, with the adjustment field still 0, sets the adjustment.
        BinaryPrimitives.WriteUInt32BigEndian(span[(headOffset + 8)..], unchecked(0xB1B0AFBA - Checksum(font)));
        return font;
    }

    private static int Pad(int length) => (length + 3) & ~3;

    private static uint Checksum(ReadOnlySpan<byte> data)
    {
        uint sum = 0;
        int i = 0;
        for (; i + 4 <= data.Length; i += 4) sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(data[i..]));
        if (i < data.Length)
        {
            Span<byte> last = stackalloc byte[4];
            last.Clear();
            data[i..].CopyTo(last);
            sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(last));
        }
        return sum;
    }

    private static byte[] Head()
    {
        var t = new byte[54];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt32BigEndian(s, 0x00010000);          // version
        BinaryPrimitives.WriteUInt32BigEndian(s[4..], 0x00010000);     // font revision
        BinaryPrimitives.WriteUInt32BigEndian(s[12..], 0x5F0F3CF5);    // magic number
        BinaryPrimitives.WriteUInt16BigEndian(s[16..], 0x000B);        // baseline and left sidebearing at 0, integer scaling
        BinaryPrimitives.WriteUInt16BigEndian(s[18..], UnitsPerEm);
        // The created and modified dates stay 0, so the font's bytes are the same on every build.
        BinaryPrimitives.WriteInt16BigEndian(s[36..], 0);              // xMin
        BinaryPrimitives.WriteInt16BigEndian(s[38..], Descent);        // yMin
        BinaryPrimitives.WriteInt16BigEndian(s[40..], Advance);        // xMax
        BinaryPrimitives.WriteInt16BigEndian(s[42..], Ascent);         // yMax
        BinaryPrimitives.WriteUInt16BigEndian(s[46..], 3);             // lowest readable size in pixels
        BinaryPrimitives.WriteInt16BigEndian(s[48..], 2);              // font direction hint
        BinaryPrimitives.WriteInt16BigEndian(s[50..], 0);              // short loca offsets
        return t;
    }

    private static byte[] Hhea()
    {
        var t = new byte[36];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt32BigEndian(s, 0x00010000);
        BinaryPrimitives.WriteInt16BigEndian(s[4..], Ascent);
        BinaryPrimitives.WriteInt16BigEndian(s[6..], Descent);
        BinaryPrimitives.WriteUInt16BigEndian(s[10..], Advance);       // largest advance
        BinaryPrimitives.WriteInt16BigEndian(s[18..], 1);              // caret slope rise: upright
        BinaryPrimitives.WriteUInt16BigEndian(s[34..], 3);             // number of horizontal metrics
        return t;
    }

    private static byte[] Maxp()
    {
        var t = new byte[32];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt32BigEndian(s, 0x00010000);
        BinaryPrimitives.WriteUInt16BigEndian(s[4..], 3);              // glyphs: .notdef, the empty glyph and the zero-width one
        BinaryPrimitives.WriteUInt16BigEndian(s[14..], 2);             // zones
        return t;
    }

    private static byte[] Hmtx()
    {
        // Advance and left side bearing of .notdef, the empty glyph and the zero-width glyph.
        var t = new byte[12];
        BinaryPrimitives.WriteUInt16BigEndian(t, Advance);
        BinaryPrimitives.WriteUInt16BigEndian(t.AsSpan(4), Advance);
        return t;
    }

    private static byte[] Cmap()
    {
        // One Windows Unicode subtable (format 4) with only the closing segment: the PDF maps codes to glyphs itself,
        // through CIDToGIDMap, and this keeps readers that expect a cmap content.
        var t = new byte[4 + 8 + 24];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt16BigEndian(s[2..], 1);              // one subtable
        BinaryPrimitives.WriteUInt16BigEndian(s[4..], 3);              // Windows
        BinaryPrimitives.WriteUInt16BigEndian(s[6..], 1);              // Unicode BMP
        BinaryPrimitives.WriteUInt32BigEndian(s[8..], 12);
        var f = s[12..];
        BinaryPrimitives.WriteUInt16BigEndian(f, 4);                   // format
        BinaryPrimitives.WriteUInt16BigEndian(f[2..], 24);             // length
        BinaryPrimitives.WriteUInt16BigEndian(f[6..], 2);              // segment count × 2
        BinaryPrimitives.WriteUInt16BigEndian(f[8..], 2);              // search range
        BinaryPrimitives.WriteUInt16BigEndian(f[10..], 0);             // entry selector
        BinaryPrimitives.WriteUInt16BigEndian(f[12..], 0);             // range shift
        BinaryPrimitives.WriteUInt16BigEndian(f[14..], 0xFFFF);        // end code
        BinaryPrimitives.WriteUInt16BigEndian(f[18..], 0xFFFF);        // start code (after the reserved pad)
        BinaryPrimitives.WriteUInt16BigEndian(f[20..], 1);             // delta: 0xFFFF maps to glyph 0
        return t;
    }

    private static byte[] Os2()
    {
        var t = new byte[96];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt16BigEndian(s, 4);                   // version
        BinaryPrimitives.WriteInt16BigEndian(s[2..], Advance);         // average width
        BinaryPrimitives.WriteUInt16BigEndian(s[4..], 400);            // regular weight
        BinaryPrimitives.WriteUInt16BigEndian(s[6..], 5);              // normal width
        // fsType 0: the font may be embedded and installed without restriction.
        "PDFP"u8.CopyTo(s[58..]);                                      // vendor
        BinaryPrimitives.WriteUInt16BigEndian(s[62..], 0x0040);        // regular
        BinaryPrimitives.WriteUInt16BigEndian(s[64..], 0x0020);        // first character
        BinaryPrimitives.WriteUInt16BigEndian(s[66..], 0xFFFF);        // last character
        BinaryPrimitives.WriteInt16BigEndian(s[68..], Ascent);         // typographic ascender
        BinaryPrimitives.WriteInt16BigEndian(s[70..], Descent);        // typographic descender
        BinaryPrimitives.WriteUInt16BigEndian(s[74..], Ascent);        // Windows ascent
        BinaryPrimitives.WriteUInt16BigEndian(s[76..], -Descent);      // Windows descent
        BinaryPrimitives.WriteInt16BigEndian(s[86..], 500);            // x-height
        BinaryPrimitives.WriteInt16BigEndian(s[88..], Ascent);         // cap height
        BinaryPrimitives.WriteUInt16BigEndian(s[92..], 0x0020);        // break character
        return t;
    }

    private static byte[] NameTable()
    {
        var names = new (int Id, string Text)[] { (1, Name), (2, "Regular"), (3, Name), (4, Name), (5, "Version 1.0"), (6, Name) };
        var strings = new List<byte[]>();
        foreach (var (_, text) in names) strings.Add(Encoding.BigEndianUnicode.GetBytes(text));
        int header = 6 + 12 * names.Length;
        int total = header;
        foreach (var b in strings) total += b.Length;
        var t = new byte[total];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt16BigEndian(s[2..], (ushort)names.Length);
        BinaryPrimitives.WriteUInt16BigEndian(s[4..], (ushort)header);
        int at = 0;
        for (int i = 0; i < names.Length; i++)
        {
            var r = s.Slice(6 + 12 * i, 12);
            BinaryPrimitives.WriteUInt16BigEndian(r, 3);               // Windows
            BinaryPrimitives.WriteUInt16BigEndian(r[2..], 1);          // Unicode BMP
            BinaryPrimitives.WriteUInt16BigEndian(r[4..], 0x0409);     // English (United States)
            BinaryPrimitives.WriteUInt16BigEndian(r[6..], (ushort)names[i].Id);
            BinaryPrimitives.WriteUInt16BigEndian(r[8..], (ushort)strings[i].Length);
            BinaryPrimitives.WriteUInt16BigEndian(r[10..], (ushort)at);
            strings[i].CopyTo(s[(header + at)..]);
            at += strings[i].Length;
        }
        return t;
    }

    private static byte[] Post()
    {
        var t = new byte[32];
        var s = t.AsSpan();
        BinaryPrimitives.WriteUInt32BigEndian(s, 0x00030000);          // version 3: no glyph names
        BinaryPrimitives.WriteInt16BigEndian(s[8..], -100);            // underline position
        BinaryPrimitives.WriteInt16BigEndian(s[10..], 50);             // underline thickness
        BinaryPrimitives.WriteUInt32BigEndian(s[12..], 1);             // fixed pitch
        return t;
    }
}
