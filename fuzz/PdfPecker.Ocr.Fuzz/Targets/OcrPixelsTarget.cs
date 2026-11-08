using System.Buffers.Binary;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Raw pixel buffers through <see cref="ImageOcr.RecognizeRgba"/> and <see cref="ImageOcr.RecognizeGray"/>. The first
/// bytes of the input choose the channel count, the options and the stated width and height, and the rest are the
/// pixels, so the buffer length and the stated size disagree in every way: zero, negative, huge and products that
/// pass the range of an int.
/// </summary>
/// <remarks>
/// The layout is: byte 0 holds flags (bit 0 RGBA rather than gray, bit 1 sizes as full 32-bit numbers rather than 16-bit
/// ones, bit 2 no rotation check, bit 3 two threads, bit 4 the height taken from the buffer's length instead of the
/// input), byte 1 the minimum confidence in 255ths, then the width and the height (two or four bytes each,
/// little-endian), then the pixels. Bit 4 keeps mutated buffers, which rarely keep their stated length, reaching the
/// models rather than all ending in the length check.
/// </remarks>
internal static class OcrPixelsTarget
{
    private const long MaxPixels = 250_000;

    public static Target Target { get; } = new()
    {
        Name = "ocr-pixels",
        Extension = ".pix",
        MaxSize = 1024 * 1024,
        MinTimeout = TimeSpan.FromSeconds(20),
        // A buffer shorter than its stated size, or a size below 1, is the documented ArgumentException family.
        ExtraExpected = ex => ex is ArgumentException,
        Seeds = Generate,
        Run = Run,
    };

    private static void Run(byte[] input, CancellationToken ct)
    {
        if (input.Length < 2) return;
        byte flags = input[0];
        bool wide = (flags & 2) != 0;
        int header = 2 + (wide ? 8 : 4);
        if (input.Length < header) return;
        int width = wide ? BinaryPrimitives.ReadInt32LittleEndian(input.AsSpan(2)) : BinaryPrimitives.ReadUInt16LittleEndian(input.AsSpan(2));
        int height = wide ? BinaryPrimitives.ReadInt32LittleEndian(input.AsSpan(6)) : BinaryPrimitives.ReadUInt16LittleEndian(input.AsSpan(4));
        var options = new OcrOptions
        {
            MaxPixels = MaxPixels,
            MinConfidence = input[1] / 255f,
            DetectRotation = (flags & 4) == 0,
            MaxDegreeOfParallelism = (flags & 8) != 0 ? 2 : 1,
        };
        ReadOnlySpan<byte> pixels = input.AsSpan(header);
        int channels = (flags & 1) != 0 ? 4 : 1;
        if ((flags & 16) != 0 && width > 0) height = pixels.Length / channels / width;
        OcrPage page = channels == 4
            ? ImageOcr.RecognizeRgba(pixels, width, height, options, ct)
            : ImageOcr.RecognizeGray(pixels, width, height, options, ct);
        Check(page, width, height);
    }

    /// <summary>A result must describe the image it came from: its size, and quads and confidences that are numbers.</summary>
    private static void Check(OcrPage page, int width, int height)
    {
        if (page.Width != width || page.Height != height)
            throw new InvalidOperationException($"The page is {page.Width} x {page.Height}, the image {width} x {height}.");
        foreach (var line in page.Lines)
        {
            if (!(line.Confidence >= 0 && line.Confidence <= 1) || !float.IsFinite(line.Box.X) || !float.IsFinite(line.Box.Width))
                throw new InvalidOperationException("A line has a confidence outside 0 to 1 or a box that is not a number.");
        }
    }

    private static IEnumerable<(string Name, byte[] Data)> Generate(SeedContext ctx)
    {
        // Pages of text at a low resolution, as gray and as RGBA, so mutation starts from pixels the models read.
        foreach (var (name, pdf) in ctx.GeneratedPdfs().Take(3))
        {
            var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = 60 });
            var gray = new byte[image.Width * image.Height];
            for (int i = 0; i < gray.Length; i++) gray[i] = image.Rgba[(i * 4) + 1];
            yield return (name + ":gray", Frame(16, image.Width, image.Height, gray));
            yield return (name + ":rgba", Frame(1 | 8, image.Width, image.Height, image.Rgba));
        }

        yield return ("generated:tiny", Frame(0, 1, 1, [0]));
        yield return ("generated:stripes", Frame(4, 64, 48, Enumerable.Range(0, 64 * 48).Select(i => (byte)((i / 64) % 4 < 2 ? 0 : 255)).ToArray()));
        yield return ("generated:wide-sizes", Wide(1, 300, 2, new byte[300 * 2 * 4]));
        yield return ("generated:overflow", Wide(0, 65536, 65536, new byte[1024]));
    }

    private static byte[] Frame(byte flags, int width, int height, byte[] pixels)
    {
        var data = new byte[6 + pixels.Length];
        data[0] = flags;
        data[1] = 128;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(2), (ushort)width);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(4), (ushort)height);
        pixels.CopyTo(data.AsSpan(6));
        return data;
    }

    private static byte[] Wide(byte flags, int width, int height, byte[] pixels)
    {
        var data = new byte[10 + pixels.Length];
        data[0] = (byte)(flags | 2);
        data[1] = 0;
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(2), width);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(6), height);
        pixels.CopyTo(data.AsSpan(10));
        return data;
    }
}
