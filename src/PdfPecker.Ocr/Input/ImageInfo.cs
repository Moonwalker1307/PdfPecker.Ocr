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

    private ImageInfo(ImageFormat format, float dpi, int[] frames)
    {
        Format = format;
        Dpi = dpi;
        TiffFrames = frames;
    }

    /// <summary>The file's format.</summary>
    public ImageFormat Format { get; }

    /// <summary>The horizontal resolution the file records in pixels per inch, or 0 when it records none that is plausible.</summary>
    public float Dpi { get; }

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
                ImageFormat.Png => new ImageInfo(format, PngDpi(data), NoFrames),
                ImageFormat.Jpeg => new ImageInfo(format, JpegDpi(data), NoFrames),
                ImageFormat.Bmp => new ImageInfo(format, BmpDpi(data), NoFrames),
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
    /// A copy of a TIFF file whose header points at page <paramref name="frame"/>, so a reader of first pages, such as
    /// <see cref="ImageToPdf"/>, reads that page. The rest of the file is unchanged, which keeps every offset valid.
    /// </summary>
    public static byte[] TiffFrame(byte[] tiff, ImageInfo info, int frame)
    {
        if (frame == 0) return tiff;
        var copy = (byte[])tiff.Clone();
        int offset = info.TiffFrames[frame];
        if (tiff[0] == 0x49) BinaryPrimitives.WriteInt32LittleEndian(copy.AsSpan(4), offset);
        else BinaryPrimitives.WriteInt32BigEndian(copy.AsSpan(4), offset);
        return copy;
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

    private static float JpegDpi(ReadOnlySpan<byte> data)
    {
        int pos = 2;
        float exif = 0;
        for (int n = 0; n < MaxChunks && pos + 4 <= data.Length; n++)
        {
            if (data[pos] != 0xFF) break;
            byte marker = data[pos + 1];
            if (marker == 0xFF) { pos++; continue; }
            // Start of scan or of a frame: the headers that carry a resolution come before both.
            if (marker is 0xDA or 0xD9 || marker is >= 0xC0 and <= 0xCF and not 0xC4 and not 0xC8 and not 0xCC) break;
            int length = BinaryPrimitives.ReadUInt16BigEndian(data[(pos + 2)..]);
            if (length < 2 || pos + 2 + length > data.Length) break;
            var segment = data.Slice(pos + 4, length - 2);
            if (marker == 0xE0 && segment.Length >= 12 && segment[..5].SequenceEqual("JFIF\0"u8))
            {
                int units = segment[7], x = BinaryPrimitives.ReadUInt16BigEndian(segment[8..]);
                // Units 1 are dots per inch, 2 dots per centimetre; 0 gives only the aspect ratio.
                if (units == 1) return Plausible(x);
                if (units == 2) return Plausible(x * 2.54);
            }
            else if (marker == 0xE1 && exif == 0 && segment.Length > 14 && segment[..6].SequenceEqual("Exif\0\0"u8))
            {
                var tiff = segment[6..];
                if (Sniff(tiff) == ImageFormat.Tiff) exif = TiffDpi(tiff, FirstIfd(tiff));
            }
            pos += 2 + length;
        }
        return exif;
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
        float dpi = frames.Count > 0 ? TiffDpi(data, frames[0]) : 0;
        return new ImageInfo(ImageFormat.Tiff, dpi, frames.ToArray());
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

    private static int U16(ReadOnlySpan<byte> d, int at, bool little)
        => little ? BinaryPrimitives.ReadUInt16LittleEndian(d[at..]) : BinaryPrimitives.ReadUInt16BigEndian(d[at..]);

    private static uint U32(ReadOnlySpan<byte> d, int at, bool little)
        => little ? BinaryPrimitives.ReadUInt32LittleEndian(d[at..]) : BinaryPrimitives.ReadUInt32BigEndian(d[at..]);
}
