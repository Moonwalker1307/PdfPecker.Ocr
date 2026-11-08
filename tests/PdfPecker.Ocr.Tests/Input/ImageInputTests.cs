using System.Buffers.Binary;
using PdfPecker.Ocr.Input;

namespace PdfPecker.Ocr.Tests.Input;

public sealed class ImageInputTests
{
    private static readonly Lazy<RenderedImage> s_page = new(() =>
    {
        var html = "<body style='font:11pt serif'>" + string.Concat(Enumerable.Range(1, 30).Select(i => $"<p style='color:#{i * 8:x2}3060'>Line {i}: æøå ß café</p>")) + "</body>";
        return PdfRenderer.Open(HtmlToPdf.Convert(html)).RenderPage(0, new ImageOptions { Dpi = 120 });
    });

    private static RenderedImage Page => s_page.Value;

    private static byte[] Rgb(RenderedImage image) => PixelsOf(image.Rgba, image.Width, image.Height);

    private static byte[] PixelsOf(byte[] rgba, int w, int h)
    {
        using var p = PagePixels.FromRgba(rgba, w, h);
        return p.Pixels.ToArray();
    }

    [Fact]
    public void PngDecodesToItsOwnPixels()
    {
        var png = Page.ToPng();
        var decoded = ImageDecoder.Decode(png, 300, 100_000_000, default);
        using (decoded.Pixels)
        {
            Assert.Equal((Page.Width, Page.Height), (decoded.Pixels.Width, decoded.Pixels.Height));
            Assert.Equal(PixelFormat.Rgb24, decoded.Pixels.Format);
            Assert.Equal(Rgb(Page), decoded.Pixels.Pixels.ToArray());
            // The PNG records no resolution, so it has the fallback.
            Assert.Equal(300, decoded.Dpi);
        }
    }

    [Fact]
    public void EveryPageOfATiffDecodes()
    {
        var pdf = HtmlToPdf.Convert("<p>one</p><p style='break-before:page;font-size:40pt'>two</p><p style='break-before:page;background:#c33'>three</p>");
        var renderer = PdfRenderer.Open(pdf);
        var tiff = renderer.RenderAllToTiff(new ImageOptions { Dpi = 50 });
        var info = ImageInfo.Read(tiff);
        Assert.Equal(ImageFormat.Tiff, info.Format);
        Assert.Equal(3, info.FrameCount);
        Assert.Equal(50, info.Dpi);
        for (int i = 0; i < 3; i++)
        {
            var page = renderer.RenderPage(i, new ImageOptions { Dpi = 50 });
            var decoded = ImageDecoder.Decode(ImageInfo.TiffFrame(tiff, info, i), 300, long.MaxValue, default);
            using (decoded.Pixels)
            {
                Assert.Equal(PixelsOf(page.Rgba, page.Width, page.Height), decoded.Pixels.Pixels.ToArray());
                Assert.Equal(50, decoded.Dpi);
            }
        }
    }

    [Fact]
    public void BmpWithResolutionDecodes()
    {
        const int w = 37, h = 11;
        var rgb = new byte[w * h * 3];
        new Random(5).NextBytes(rgb);
        var bmp = TestImages.Bmp(rgb, w, h, pixelsPerMetre: 11_811);
        Assert.Equal(300, ImageInfo.Read(bmp).Dpi, 0.1);
        var decoded = ImageDecoder.Decode(bmp, 72, long.MaxValue, default);
        using (decoded.Pixels)
        {
            Assert.Equal(rgb, decoded.Pixels.Pixels.ToArray());
            Assert.Equal(300, decoded.Dpi, 0.1);
        }
    }

    [Fact]
    public void LossyFormatsDecodeCloseToTheSource()
    {
        var source = Rgb(Page);
        foreach (var data in new[] { Page.ToJpeg(95), Page.ToWebP(95) })
        {
            var decoded = ImageDecoder.Decode(data, 300, long.MaxValue, default);
            using (decoded.Pixels)
            {
                Assert.Equal((Page.Width, Page.Height), (decoded.Pixels.Width, decoded.Pixels.Height));
                var pixels = decoded.Pixels.Pixels;
                double diff = 0;
                for (int i = 0; i < source.Length; i++) diff += Math.Abs(source[i] - pixels[i]);
                Assert.True(diff / source.Length < 3, $"Mean difference {diff / source.Length:0.00}");
            }
        }
    }

    [Fact]
    public void GifDecodes()
    {
        // A 1 x 1 GIF whose one pixel is red.
        var gif = Convert.FromBase64String("R0lGODlhAQABAPAAAP8AAAAAACwAAAAAAQABAAACAkQBADs=");
        Assert.Equal(ImageFormat.Gif, ImageInfo.Sniff(gif));
        var decoded = ImageDecoder.Decode(gif, 300, long.MaxValue, default);
        using (decoded.Pixels)
            Assert.Equal(new byte[] { 255, 0, 0 }, decoded.Pixels.Pixels.ToArray());
    }

    [Fact]
    public void TransparencyBecomesWhite()
    {
        var pdf = HtmlToPdf.Convert("<div style='margin:20px;width:40px;height:40px;background:#000'></div>");
        var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = 30, Transparent = true });
        var decoded = ImageDecoder.Decode(image.ToPng(), 300, long.MaxValue, default);
        using (decoded.Pixels)
        {
            Assert.Equal(PixelFormat.Gray8, decoded.Pixels.Format);
            Assert.Equal(255, decoded.Pixels.Pixels[0]);
            Assert.Contains((byte)0, decoded.Pixels.Pixels.ToArray());
        }
    }

    [Fact]
    public void LargeImagesAreReadSmallerWithinMaxPixels()
    {
        var png = Page.ToPng();
        var decoded = ImageDecoder.Decode(png, 300, 200_000, default);
        using (decoded.Pixels)
        {
            long count = (long)decoded.Pixels.Width * decoded.Pixels.Height;
            Assert.InRange(count, 180_000, 200_000);
            Assert.InRange(decoded.Dpi / (300.0 * decoded.Pixels.Width / Page.Width), 0.998, 1.002);
        }
    }

    [Fact]
    public void BadImagesFailWithInvalidDataException()
    {
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(new byte[] { 1, 2, 3 }, 300, long.MaxValue, default));
        var png = Page.ToPng();
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(png[..(png.Length / 3)], 300, long.MaxValue, default));
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 300, long.MaxValue, default));
    }

    [Fact]
    public void ResolutionsAreReadFromHeaders()
    {
        var png = WithPhys(Page.ToPng(), 7874); // 200 dpi
        Assert.Equal(200, ImageInfo.Read(png).Dpi, 0.1);
        Assert.Equal(0, ImageInfo.Read(WithPhys(Page.ToPng(), 7874, unit: 0)).Dpi);
        // PdfPecker still decodes the PNG with the extra chunk, so its CRC is right.
        using (ImageDecoder.Decode(png, 300, long.MaxValue, default).Pixels) { }

        var jpeg = Page.ToJpeg(80);
        Assert.Equal(ImageFormat.Jpeg, ImageInfo.Sniff(jpeg));
        var jfif = Jfif(units: 1, density: 150);
        Assert.Equal(150, ImageInfo.Read(jfif).Dpi);
        Assert.Equal(254, ImageInfo.Read(Jfif(units: 2, density: 100)).Dpi, 0.01);
        Assert.Equal(0, ImageInfo.Read(Jfif(units: 0, density: 1)).Dpi);
        Assert.Equal(0, ImageInfo.Read(TestImages.Bmp(new byte[3], 1, 1, pixelsPerMetre: 0)).Dpi);
    }

    [Fact]
    public void DamagedHeadersGiveNoResolutionInsteadOfFailing()
    {
        var files = new[]
        {
            WithPhys(Page.ToPng(), 7874), Page.ToJpeg(80), TestImages.Bmp(new byte[12], 2, 2, 3780),
            PdfRenderer.Open(HtmlToPdf.Convert("<p>a</p><p style='break-before:page'>b</p>")).RenderAllToTiff(new ImageOptions { Dpi = 20 }),
        };
        var rnd = new Random(1);
        foreach (var file in files)
        {
            for (int length = 0; length < Math.Min(file.Length, 400); length++) _ = ImageInfo.Read(file.AsSpan(0, length));
            for (int i = 0; i < 2000; i++)
            {
                var copy = (byte[])file.Clone();
                for (int k = 0; k < 4; k++) copy[rnd.Next(Math.Min(copy.Length, 300))] = (byte)rnd.Next(256);
                var info = ImageInfo.Read(copy);
                Assert.True(info.Dpi == 0 || info.Dpi is >= 10 and <= 10_000);
                Assert.InRange(info.FrameCount, 1, ImageInfo.MaxFrames);
            }
        }
    }

    [Fact]
    public void TiffDirectoryLoopsEnd()
    {
        // A little-endian TIFF whose only directory points back at itself.
        var tiff = new byte[8 + 2 + 12 + 4];
        "II*\0"u8.CopyTo(tiff);
        BinaryPrimitives.WriteInt32LittleEndian(tiff.AsSpan(4), 8);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(8), 1);
        BinaryPrimitives.WriteInt32LittleEndian(tiff.AsSpan(22), 8);
        var info = ImageInfo.Read(tiff);
        Assert.Equal(1, info.FrameCount);
    }

    /// <summary>The PNG with a pHYs chunk after its header chunk.</summary>
    private static byte[] WithPhys(byte[] png, uint perUnit, byte unit = 1)
    {
        var chunk = new byte[21];
        BinaryPrimitives.WriteUInt32BigEndian(chunk, 9);
        "pHYs"u8.CopyTo(chunk.AsSpan(4));
        BinaryPrimitives.WriteUInt32BigEndian(chunk.AsSpan(8), perUnit);
        BinaryPrimitives.WriteUInt32BigEndian(chunk.AsSpan(12), perUnit);
        chunk[16] = unit;
        BinaryPrimitives.WriteUInt32BigEndian(chunk.AsSpan(17), Crc32(chunk.AsSpan(4, 13)));
        int afterHeader = 8 + 25;
        return png[..afterHeader].Concat(chunk).Concat(png[afterHeader..]).ToArray();
    }

    /// <summary>The start of a JPEG file with a JFIF header, enough for the header reader.</summary>
    private static byte[] Jfif(byte units, ushort density)
    {
        var data = new byte[] { 0xFF, 0xD8, 0xFF, 0xE0, 0, 16, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 1, 2, units, 0, 0, 0, 0, 0, 0, 0xFF, 0xD9 };
        BinaryPrimitives.WriteUInt16BigEndian(data.AsSpan(14), density);
        BinaryPrimitives.WriteUInt16BigEndian(data.AsSpan(16), density);
        return data;
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320 : crc >> 1;
        }
        return ~crc;
    }
}
