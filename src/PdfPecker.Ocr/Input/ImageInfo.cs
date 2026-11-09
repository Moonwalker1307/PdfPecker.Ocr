using System.Buffers.Binary;

namespace PdfPecker.Ocr.Input;

/// <summary>The kinds of image file PdfPecker reads.</summary>
internal enum ImageFormat
{
    /// <summary>Not an image file PdfPecker reads.</summary>
    Unknown,
    /// <summary>PNG.</summary>
    Png,
    /// <summary>JPEG.</summary>
    Jpeg,
    /// <summary>GIF.</summary>
    Gif,
    /// <summary>WebP.</summary>
    WebP,
    /// <summary>BMP.</summary>
    Bmp,
    /// <summary>TIFF, which can hold several pages.</summary>
    Tiff,
}

/// <summary>
/// What an image file's header says: its format, the resolution it records, and for TIFF where each page starts.
/// PdfPecker's public API decodes images but does not report their resolution or read past a TIFF's first page, so
/// this reads just those fields. It never decodes pixels.
/// </summary>
/// <remarks>
/// The input is untrusted: every offset is checked against the data, loops are bounded, and anything unreadable gives
/// an unknown resolution rather than an exception. PdfPecker itself decides later whether the image can be decoded.
/// </remarks>
internal sealed class ImageInfo
{
    /// <summary>The most TIFF pages read, as many as <see cref="ImageToPdf"/> converts in one call.</summary>
    public const int MaxFrames = ImageToPdf.MaxImages;

    private const int MaxChunks = 10_000;
    private static readonly int[] NoFrames = Array.Empty<int>();

    private ImageInfo(ImageFormat format, float dpi, int[] frames, int width = 0, int height = 0)
    {
        Format = format;
        Dpi = dpi;
        TiffFrames = frames;
        Width = Math.Max(0, width);
        Height = Math.Max(0, height);
    }

    /// <summary>The file's format.</summary>
    public ImageFormat Format { get; }

    /// <summary>The horizontal resolution the file records in pixels per inch, or 0 when it records none that is plausible.</summary>
    public float Dpi { get; }

    /// <summary>
    /// The width in pixels of the image as it is displayed (after a JPEG's EXIF orientation), or 0 when the header
    /// does not say. For TIFF it is the first page's.
    /// </summary>
    public int Width { get; }

    /// <summary>The height in pixels of the image as it is displayed, or 0 when the header does not say.</summary>
    public int Height { get; }

    /// <summary>For TIFF, the offset of each page's directory (IFD) in the file; empty for other formats.</summary>
    public IReadOnlyList<int> TiffFrames { get; }

    /// <summary>The number of pages: the TIFF directories found, otherwise 1.</summary>
    public int FrameCount => Format == ImageFormat.Tiff && TiffFrames.Count > 0 ? TiffFrames.Count : 1;

    /// <summary>Reads the header of an image file.</summary>
    public static ImageInfo Read(ReadOnlySpan<byte> data)
    {
        var format = Sniff(data);
        try
        {
            return format switch
            {
                ImageFormat.Png => new ImageInfo(format, PngDpi(data), NoFrames,
                    data.Length >= 24 ? Size(BinaryPrimitives.ReadUInt32BigEndian(data[16..])) : 0,
                    data.Length >= 24 ? Size(BinaryPrimitives.ReadUInt32BigEndian(data[20..])) : 0),
                ImageFormat.Jpeg => ReadJpeg(data),
                ImageFormat.Bmp => new ImageInfo(format, BmpDpi(data), NoFrames,
                    data.Length >= 26 ? BinaryPrimitives.ReadInt32LittleEndian(data[18..]) : 0,
                    data.Length >= 26 ? Size((uint)Math.Abs((long)BinaryPrimitives.ReadInt32LittleEndian(data[22..]))) : 0),
                ImageFormat.Gif => new ImageInfo(format, 0, NoFrames,
                    data.Length >= 10 ? BinaryPrimitives.ReadUInt16LittleEndian(data[6..]) : 0,
                    data.Length >= 10 ? BinaryPrimitives.ReadUInt16LittleEndian(data[8..]) : 0),
                ImageFormat.WebP => ReadWebP(data),
                ImageFormat.Tiff => ReadTiff(data),
                _ => new ImageInfo(format, 0, NoFrames),
            };
        }
        catch (ArgumentOutOfRangeException)
        {
            // A truncated field: the header is damaged, so the resolution is unknown. PdfPecker reports the damage
            // when it decodes the image.
            return new ImageInfo(format, 0, NoFrames);
        }
    }

    /// <summary>The format from the file's first bytes, the way PdfPecker recognises them.</summary>
    public static ImageFormat Sniff(ReadOnlySpan<byte> data) => data switch
    {
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, ..] => ImageFormat.Png,
        [0xFF, 0xD8, ..] => ImageFormat.Jpeg,
        [0x47, 0x49, 0x46, 0x38, ..] => ImageFormat.Gif,
        [0x52, 0x49, 0x46, 0x46, _, _, _, _, 0x57, 0x45, 0x42, 0x50, ..] => ImageFormat.WebP,
        [0x42, 0x4D, ..] => ImageFormat.Bmp,
        [0x49, 0x49, 0x2A, 0x00, ..] or [0x4D, 0x4D, 0x00, 0x2A, ..] => ImageFormat.Tiff,
        _ => ImageFormat.Unknown,
    };

    /// <summary>
    /// Page <paramref name="frame"/> of a TIFF file as a file of its own, so a reader of first pages, such as
    /// <see cref="ImageToPdf"/>, reads that page. A file of one page is returned as it is; a page of a multi-page file
    /// gets a header, a copy of its directory and its values, and only the strips or tiles it refers to, with their
    /// offsets rewritten (see <see cref="TiffPage"/>).
    /// </summary>
    /// <param name="tiff">The TIFF file.</param>
    /// <param name="info">What <see cref="Read"/> found in it.</param>
    /// <param name="frame">The page, from 0.</param>
    /// <param name="budget">The most bytes the page may take; a page that needs more is refused before it is made.</param>
    /// <exception cref="PdfLimitException">The page would take more than <paramref name="budget"/> bytes.</exception>
    public static byte[] TiffFrame(byte[] tiff, ImageInfo info, int frame, long budget = long.MaxValue)
        => info.FrameCount == 1 ? tiff : TiffPage(tiff, info.TiffFrames[frame], budget);

    /// <summary>
    /// The bytes the pages of a multi-page TIFF may take together once each is a file of its own: twice the file and a
    /// megabyte. Each page's own data adds up to the file, but pages may share strips, and a file of thousands of pages
    /// that all refer to one large strip would otherwise be copied and decoded once per page.
    /// </summary>
    public static long TiffFramesBudget(int fileLength) => 2L * fileLength + (1 << 20);

    // The size in bytes of each TIFF field type, by type number: BYTE, ASCII, SHORT, LONG, RATIONAL, SBYTE, UNDEFINED,
    // SSHORT, SLONG, SRATIONAL, FLOAT, DOUBLE and IFD.
    private static ReadOnlySpan<byte> TiffTypeSizes => [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4];

    /// <summary>
    /// Tags that point at other parts of the file which a page's pixels do not need (free space, sub-directories, the
    /// EXIF, GPS and interoperability directories, and the old-style JPEG data PdfPecker does not read); a copied
    /// directory leaves them out rather than point at the wrong bytes.
    /// </summary>
    private static bool TiffPointerTag(int tag) => tag is 288 or 289 or 330 or 513 or 514 or 519 or 520 or 521 or 34665 or 34853 or 40965;

    private readonly record struct TiffEntry(int Tag, int Type, uint Count, int At, long Size);

    /// <summary>
    /// The page whose directory is at <paramref name="ifd"/> as a TIFF file of its own, in the file's byte order.
    /// </summary>
    /// <remarks>
    /// Values outside the file are left out, as are tags of unknown types; strips and tiles are clipped to the file, and
    /// strips that share bytes share them in the copy too. The copy is padded to the size of the whole file at most,
    /// so that PdfPecker's check that a few bytes do not claim a huge image (its limit on how much a file may expand)
    /// decides for a well compressed page as it did with the whole file.
    /// </remarks>
    private static byte[] TiffPage(byte[] tiff, int ifd, long budget)
    {
        bool little = tiff[0] == 0x49;
        var entries = new List<TiffEntry>();
        int count = ifd >= 8 && ifd + 2 <= tiff.Length ? U16(tiff, ifd, little) : 0;
        for (int i = 0; i < count; i++)
        {
            int at = ifd + 2 + i * 12;
            if (at + 12 > tiff.Length) break;
            int tag = U16(tiff, at, little), type = U16(tiff, at + 2, little);
            uint n = U32(tiff, at + 4, little);
            if (type < 1 || type >= TiffTypeSizes.Length || TiffPointerTag(tag)) continue;
            long size = (long)TiffTypeSizes[type] * n;
            if (size > 4 && U32(tiff, at + 8, little) + size > tiff.Length) continue;
            entries.Add(new TiffEntry(tag, type, n, at, size));
        }

        bool tiled = entries.Exists(e => e.Tag == 324);
        int offsetsAt = entries.FindIndex(e => e.Tag == (tiled ? 324 : 273));
        int countsAt = entries.FindIndex(e => e.Tag == (tiled ? 325 : 279));
        long[] offsets = offsetsAt >= 0 ? TiffValues(tiff, entries[offsetsAt], little) : [];
        long[] counts = countsAt >= 0 ? TiffValues(tiff, entries[countsAt], little) : [];
        if (offsets.Length > 0 && counts.Length == 0)
        {
            // Without byte counts nothing says where the data ends: the page keeps the whole file, as it once did.
            if (tiff.Length > budget) throw TooLarge(tiff.Length, budget);
            var whole = (byte[])tiff.Clone();
            W32(whole, 4, (uint)ifd, little);
            return whole;
        }

        // The layout: the header, the directory, the values that do not fit in their entries, then the data.
        long pos = 8 + 2 + entries.Count * 12L + 4;
        var valueAt = new long[entries.Count];
        for (int j = 0; j < entries.Count; j++)
        {
            // The strip or tile offsets and byte counts are written as LONGs, since the new offsets may need 32 bits.
            long size = j == offsetsAt || j == countsAt ? 4L * entries[j].Count : entries[j].Size;
            if (size <= 4) continue;
            pos += pos & 1;
            valueAt[j] = pos;
            pos += size;
        }

        var newOffsets = new long[offsets.Length];
        var newCounts = new long[counts.Length];
        var placed = new Dictionary<(long Start, long Length), long>();
        var ranges = new List<(long From, long To, long Length)>();
        for (int i = 0; i < offsets.Length; i++)
        {
            long start = offsets[i];
            long length = i < counts.Length && start < tiff.Length ? Math.Min(counts[i], tiff.Length - start) : 0;
            if (i < counts.Length) newCounts[i] = length;
            if (length <= 0) continue;
            if (!placed.TryGetValue((start, length), out long to))
            {
                to = pos;
                placed.Add((start, length), to);
                ranges.Add((start, to, length));
                pos += length;
                if (pos > budget) throw TooLarge(pos, budget);
            }
            newOffsets[i] = to;
        }

        pos = Math.Max(pos, Math.Min(tiff.Length, TiffRawBytes(tiff, entries, little) / 1000 + 1));
        if (pos > budget) throw TooLarge(pos, budget);
        if (pos > Array.MaxLength) throw new PdfLimitException("A page of the TIFF file is larger than 2 GB.");

        var page = new byte[pos];
        tiff.AsSpan(0, 4).CopyTo(page);
        W32(page, 4, 8, little);
        W16(page, 8, entries.Count, little);
        for (int j = 0; j < entries.Count; j++)
        {
            var e = entries[j];
            int at = 10 + j * 12;
            bool rewritten = j == offsetsAt || j == countsAt;
            W16(page, at, e.Tag, little);
            W16(page, at + 2, rewritten ? 4 : e.Type, little);
            W32(page, at + 4, e.Count, little);
            if (rewritten)
            {
                long[] values = j == offsetsAt ? newOffsets : newCounts;
                int into = valueAt[j] != 0 ? (int)valueAt[j] : at + 8;
                if (valueAt[j] != 0) W32(page, at + 8, (uint)valueAt[j], little);
                for (int k = 0; k < e.Count; k++) W32(page, into + k * 4, k < values.Length ? (uint)values[k] : 0, little);
            }
            else if (e.Size <= 4)
            {
                tiff.AsSpan(e.At + 8, 4).CopyTo(page.AsSpan(at + 8));
            }
            else
            {
                W32(page, at + 8, (uint)valueAt[j], little);
                tiff.AsSpan((int)U32(tiff, e.At + 8, little), (int)e.Size).CopyTo(page.AsSpan((int)valueAt[j]));
            }
        }

        foreach (var (from, to, length) in ranges) tiff.AsSpan((int)from, (int)length).CopyTo(page.AsSpan((int)to));
        return page;
    }

    private static PdfLimitException TooLarge(long size, long budget) => new(FormattableString.Invariant(
        $"The pages of the TIFF file refer to more data than the file holds: a page needs {size} bytes, and {budget} are left."));

    /// <summary>The SHORT or LONG values of an entry, or none for another type.</summary>
    private static long[] TiffValues(ReadOnlySpan<byte> tiff, TiffEntry e, bool little)
    {
        if (e.Type is not (3 or 4)) return [];
        int at = e.Size <= 4 ? e.At + 8 : (int)U32(tiff, e.At + 8, little);
        var values = new long[e.Count];
        for (int i = 0; i < values.Length; i++) values[i] = e.Type == 3 ? U16(tiff, at + i * 2, little) : U32(tiff, at + i * 4, little);
        return values;
    }

    /// <summary>The bytes of a page's pixels as stored, uncompressed: rows of samples of the first BitsPerSample.</summary>
    private static long TiffRawBytes(ReadOnlySpan<byte> tiff, List<TiffEntry> entries, bool little)
    {
        double raw = Math.Ceiling(TiffFirst(tiff, entries, 256, 0, little) * (double)TiffFirst(tiff, entries, 277, 1, little)
            * TiffFirst(tiff, entries, 258, 1, little) / 8) * TiffFirst(tiff, entries, 257, 0, little);
        return raw >= long.MaxValue / 2 ? long.MaxValue / 2 : (long)raw;
    }

    /// <summary>The first SHORT or LONG value of a tag among the entries, or <paramref name="fallback"/>.</summary>
    private static long TiffFirst(ReadOnlySpan<byte> tiff, List<TiffEntry> entries, int tag, long fallback, bool little)
    {
        foreach (var e in entries)
        {
            if (e.Tag != tag) continue;
            if (e.Count == 0) return fallback;
            int at = e.Size <= 4 ? e.At + 8 : (int)U32(tiff, e.At + 8, little);
            return e.Type == 3 ? U16(tiff, at, little) : e.Type == 4 ? U32(tiff, at, little) : fallback;
        }
        return fallback;
    }

    private static void W16(byte[] d, int at, int value, bool little)
    {
        if (little) BinaryPrimitives.WriteUInt16LittleEndian(d.AsSpan(at), (ushort)value);
        else BinaryPrimitives.WriteUInt16BigEndian(d.AsSpan(at), (ushort)value);
    }

    private static void W32(byte[] d, int at, uint value, bool little)
    {
        if (little) BinaryPrimitives.WriteUInt32LittleEndian(d.AsSpan(at), value);
        else BinaryPrimitives.WriteUInt32BigEndian(d.AsSpan(at), value);
    }

    /// <summary>A resolution, or 0 when it is outside what scanners and screens use.</summary>
    internal static float Plausible(double dpi) => dpi >= 10 && dpi <= 10_000 ? (float)dpi : 0;

    private static float PngDpi(ReadOnlySpan<byte> data)
    {
        int pos = 8;
        for (int n = 0; n < MaxChunks && pos + 8 <= data.Length; n++)
        {
            uint length = BinaryPrimitives.ReadUInt32BigEndian(data[pos..]);
            var type = data.Slice(pos + 4, 4);
            if (type.SequenceEqual("IDAT"u8) || type.SequenceEqual("IEND"u8)) break;
            if (type.SequenceEqual("pHYs"u8) && length >= 9 && pos + 17 <= data.Length)
            {
                uint perUnit = BinaryPrimitives.ReadUInt32BigEndian(data[(pos + 8)..]);
                // Unit 1 is the metre; unit 0 gives only the aspect ratio.
                return data[pos + 16] == 1 ? Plausible(perUnit * 0.0254) : 0;
            }
            if (length > int.MaxValue - 12 || pos + 12L + length > data.Length) break;
            pos += 12 + (int)length;
        }
        return 0;
    }

    private static int Size(uint value) => value > int.MaxValue ? 0 : (int)value;

    private static ImageInfo ReadWebP(ReadOnlySpan<byte> data)
    {
        if (data.Length < 30) return new ImageInfo(ImageFormat.WebP, 0, NoFrames);
        var chunk = data.Slice(12, 4);
        var payload = data[20..];
        int w = 0, h = 0;
        if (chunk.SequenceEqual("VP8X"u8))
        {
            // The canvas size minus one, in 24 bits each, after four bytes of flags.
            w = (payload[4] | payload[5] << 8 | payload[6] << 16) + 1;
            h = (payload[7] | payload[8] << 8 | payload[9] << 16) + 1;
        }
        else if (chunk.SequenceEqual("VP8L"u8) && payload[0] == 0x2F)
        {
            // Lossless: a signature byte, then the size minus one in two 14-bit fields.
            uint bits = BinaryPrimitives.ReadUInt32LittleEndian(payload[1..]);
            w = (int)(bits & 0x3FFF) + 1;
            h = (int)((bits >> 14) & 0x3FFF) + 1;
        }
        else if (chunk.SequenceEqual("VP8 "u8) && payload[3] == 0x9D && payload[4] == 0x01 && payload[5] == 0x2A)
        {
            // Lossy: a three-byte frame tag and start code, then the size in 14 bits each.
            w = BinaryPrimitives.ReadUInt16LittleEndian(payload[6..]) & 0x3FFF;
            h = BinaryPrimitives.ReadUInt16LittleEndian(payload[8..]) & 0x3FFF;
        }
        return new ImageInfo(ImageFormat.WebP, 0, NoFrames, w, h);
    }

    private static ImageInfo ReadJpeg(ReadOnlySpan<byte> data)
    {
        int pos = 2;
        float exif = 0, jfif = 0;
        bool haveJfif = false, turned = false;
        int w = 0, h = 0;
        for (int n = 0; n < MaxChunks && pos + 4 <= data.Length; n++)
        {
            if (data[pos] != 0xFF) break;
            byte marker = data[pos + 1];
            if (marker == 0xFF) { pos++; continue; }
            if (marker is 0xDA or 0xD9) break;
            int length = BinaryPrimitives.ReadUInt16BigEndian(data[(pos + 2)..]);
            if (length < 2 || pos + 2 + length > data.Length) break;
            var segment = data.Slice(pos + 4, length - 2);
            if (marker is >= 0xC0 and <= 0xCF and not 0xC4 and not 0xC8 and not 0xCC)
            {
                // A start of frame: precision, then the height and width. The headers with a resolution come before it.
                if (segment.Length >= 5)
                {
                    h = BinaryPrimitives.ReadUInt16BigEndian(segment[1..]);
                    w = BinaryPrimitives.ReadUInt16BigEndian(segment[3..]);
                }
                break;
            }
            if (marker == 0xE0 && !haveJfif && segment.Length >= 12 && segment[..5].SequenceEqual("JFIF\0"u8))
            {
                int units = segment[7], x = BinaryPrimitives.ReadUInt16BigEndian(segment[8..]);
                // Units 1 are dots per inch, 2 dots per centimetre; 0 gives only the aspect ratio.
                haveJfif = units is 1 or 2;
                jfif = units == 1 ? Plausible(x) : units == 2 ? Plausible(x * 2.54) : 0;
            }
            else if (marker == 0xE1 && segment.Length > 14 && segment[..6].SequenceEqual("Exif\0\0"u8))
            {
                var tiff = segment[6..];
                if (Sniff(tiff) == ImageFormat.Tiff)
                {
                    exif = TiffDpi(tiff, FirstIfd(tiff));
                    // Orientations 5 to 8 turn the image a quarter, which PdfPecker applies, so width and height swap.
                    turned = TiffShort(tiff, FirstIfd(tiff), 0x0112) is >= 5 and <= 8;
                }
            }
            pos += 2 + length;
        }
        return new ImageInfo(ImageFormat.Jpeg, haveJfif ? jfif : exif, NoFrames, turned ? h : w, turned ? w : h);
    }

    private static float BmpDpi(ReadOnlySpan<byte> data)
    {
        // BITMAPINFOHEADER and its successors keep the horizontal pixels per metre 24 bytes into the header.
        if (data.Length < 14 + 28) return 0;
        uint headerSize = BinaryPrimitives.ReadUInt32LittleEndian(data[14..]);
        if (headerSize < 40) return 0;
        int perMetre = BinaryPrimitives.ReadInt32LittleEndian(data[(14 + 24)..]);
        return Plausible(perMetre * 0.0254);
    }

    private static ImageInfo ReadTiff(ReadOnlySpan<byte> data)
    {
        var frames = new List<int>();
        var seen = new HashSet<int>();
        int ifd = FirstIfd(data);
        bool little = data[0] == 0x49;
        while (ifd >= 8 && ifd + 2 <= data.Length && frames.Count < MaxFrames && seen.Add(ifd))
        {
            frames.Add(ifd);
            int count = U16(data, ifd, little);
            long next = ifd + 2L + count * 12L;
            if (next + 4 > data.Length) break;
            ifd = (int)Math.Min(U32(data, (int)next, little), int.MaxValue);
        }
        if (frames.Count == 0) return new ImageInfo(ImageFormat.Tiff, 0, NoFrames);
        return new ImageInfo(ImageFormat.Tiff, TiffDpi(data, frames[0]), frames.ToArray(),
            TiffShort(data, frames[0], 256), TiffShort(data, frames[0], 257));
    }

    private static int FirstIfd(ReadOnlySpan<byte> tiff)
        => tiff.Length < 8 ? 0 : (int)Math.Min(U32(tiff, 4, tiff[0] == 0x49), int.MaxValue);

    /// <summary>The XResolution of one TIFF directory, in pixels per inch; TIFF's default unit is the inch.</summary>
    internal static float TiffDpi(ReadOnlySpan<byte> tiff, int ifd)
    {
        bool little = tiff[0] == 0x49;
        if (ifd < 8 || ifd + 2 > tiff.Length) return 0;
        int count = U16(tiff, ifd, little);
        double resolution = 0;
        int unit = 2;
        for (int i = 0; i < count; i++)
        {
            int entry = ifd + 2 + i * 12;
            if (entry + 12 > tiff.Length) break;
            int tag = U16(tiff, entry, little), type = U16(tiff, entry + 2, little);
            if (tag == 282 && type == 5)
            {
                uint at = U32(tiff, entry + 8, little);
                if (at > int.MaxValue - 8 || at + 8 > tiff.Length) continue;
                uint num = U32(tiff, (int)at, little), den = U32(tiff, (int)at + 4, little);
                if (den != 0) resolution = (double)num / den;
            }
            else if (tag == 296 && type == 3) unit = U16(tiff, entry + 8, little);
        }
        // Unit 1 means the image has no absolute size; 3 is the centimetre.
        return unit == 2 ? Plausible(resolution) : unit == 3 ? Plausible(resolution * 2.54) : 0;
    }

    /// <summary>The value of a SHORT or LONG tag of a TIFF directory, or 0 when it is missing.</summary>
    private static int TiffShort(ReadOnlySpan<byte> tiff, int ifd, int wanted)
    {
        bool little = tiff[0] == 0x49;
        if (ifd < 8 || ifd + 2 > tiff.Length) return 0;
        int count = U16(tiff, ifd, little);
        for (int i = 0; i < count; i++)
        {
            int entry = ifd + 2 + i * 12;
            if (entry + 12 > tiff.Length) break;
            if (U16(tiff, entry, little) != wanted) continue;
            int type = U16(tiff, entry + 2, little);
            return type == 3 ? U16(tiff, entry + 8, little) : type == 4 ? Size(U32(tiff, entry + 8, little)) : 0;
        }
        return 0;
    }

    private static int U16(ReadOnlySpan<byte> d, int at, bool little)
        => little ? BinaryPrimitives.ReadUInt16LittleEndian(d[at..]) : BinaryPrimitives.ReadUInt16BigEndian(d[at..]);

    private static uint U32(ReadOnlySpan<byte> d, int at, bool little)
        => little ? BinaryPrimitives.ReadUInt32LittleEndian(d[at..]) : BinaryPrimitives.ReadUInt32BigEndian(d[at..]);
}
