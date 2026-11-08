using System.Diagnostics;
using PdfPecker.Ocr.Tests.EndToEnd;

namespace PdfPecker.Ocr.Tests;

/// <summary>
/// The public entry points with the real engine on input that is wrong, damaged or built to cost time: each must fail
/// with an exception the API documents, or finish within a time bound with a bounded result.
/// </summary>
[Collection("EndToEnd")]
public sealed class HostileInputTests
{
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(60);

    [Theory]
    [InlineData(0, 5)]
    [InlineData(5, 0)]
    [InlineData(-1, 5)]
    [InlineData(5, int.MinValue)]
    public void RawPixelSizesMustBePositive(int width, int height)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.RecognizeGray(new byte[100], width, height));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.RecognizeRgba(new byte[100], width, height));
    }

    [Theory]
    [InlineData(5, 5, 24)]
    [InlineData(int.MaxValue, int.MaxValue, 100)]
    [InlineData(65536, 65536, 1 << 20)]
    public void RawPixelBuffersMustHoldTheImage(int width, int height, int length)
    {
        Assert.Throws<ArgumentException>(() => ImageOcr.RecognizeGray(new byte[length], width, height));
        Assert.Throws<ArgumentException>(() => ImageOcr.RecognizeRgba(new byte[length], width, height));
    }

    [Fact]
    public void RawPixelsAboveMaxPixelsAreRefused()
    {
        var options = new OcrOptions { MaxPixels = 10_000 };
        var white = new byte[101 * 100 * 4];
        white.AsSpan().Fill(255);
        Assert.Throws<PdfLimitException>(() => ImageOcr.RecognizeGray(white, 101, 100, options));
        Assert.Throws<PdfLimitException>(() => ImageOcr.RecognizeRgba(white, 101, 100, options));
        Assert.Empty(ImageOcr.RecognizeGray(white, 100, 100, options).Lines);
    }

    [Fact]
    public void BadArgumentsFailBeforeAnyWork()
    {
        Assert.Throws<ArgumentNullException>(() => ImageOcr.Recognize((byte[])null!));
        Assert.Throws<ArgumentNullException>(() => ImageOcr.Recognize((Stream)null!));
        Assert.Throws<ArgumentNullException>(() => ImageOcr.RecognizeFile(null!));
        Assert.Throws<ArgumentNullException>(() => PdfOcr.Recognize((byte[])null!));
        Assert.Throws<ArgumentNullException>(() => PdfOcr.Recognize((Stream)null!));
        Assert.Throws<ArgumentNullException>(() => PdfOcr.RecognizeFile(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.RecognizeGray(new byte[100], 10, 10, new OcrOptions { MinConfidence = 2 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.RecognizeRgba(new byte[400], 10, 10, new OcrOptions { MaxDegreeOfParallelism = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.Recognize(TestPages.Small.Png, new OcrOptions { Dpi = 0 }));
        Assert.Throws<FileNotFoundException>(() => ImageOcr.RecognizeFile(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".png")));
    }

    [Fact]
    public void DamagedFilesFailWithInvalidDataException()
    {
        byte[] png = TestPages.Small.Png;
        Assert.Throws<InvalidDataException>(() => ImageOcr.Recognize([1, 2, 3, 4, 5]));
        Assert.Throws<InvalidDataException>(() => ImageOcr.Recognize(png[..(png.Length / 2)]));
        Assert.Throws<InvalidDataException>(() => ImageOcr.Recognize([]));
        Assert.Throws<InvalidDataException>(() => PdfOcr.Recognize("not a PDF at all"u8.ToArray()));
        Assert.Throws<InvalidDataException>(() => PdfOcr.Recognize([]));
    }

    /// <summary>Pixels that cost the most per pixel: noise and stripes give the detector many regions to look at.</summary>
    [Theory]
    [InlineData("noise")]
    [InlineData("stripes")]
    [InlineData("checkerboard")]
    [InlineData("black")]
    public void CostlyPixelsFinishWithinTheBound(string kind)
    {
        const int W = 1200, H = 1600;
        var gray = new byte[W * H];
        var random = new Random(7);
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                gray[(y * W) + x] = kind switch
                {
                    "noise" => (byte)random.Next(256),
                    "stripes" => (byte)((y / 3) % 2 == 0 ? 0 : 255),
                    "checkerboard" => (byte)(((x / 4) + (y / 4)) % 2 == 0 ? 0 : 255),
                    _ => 0,
                };
            }
        }

        var watch = Stopwatch.StartNew();
        OcrPage page = ImageOcr.RecognizeGray(gray, W, H, new OcrOptions { MinConfidence = 0 });
        Assert.True(watch.Elapsed < Bound, $"{kind}: {watch.Elapsed}");
        Assert.True(page.Lines.Count <= 1000);
        Assert.All(page.Lines, l => Assert.True(OcrGeometry.IsFinite(l.Quad)));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 20000)]
    [InlineData(20000, 1)]
    [InlineData(30000, 3)]
    [InlineData(2, 2)]
    public void ExtremeShapesFinish(int width, int height)
    {
        var gray = new byte[width * height];
        for (int i = 0; i < gray.Length; i++)
        {
            gray[i] = (byte)(i % 7 == 0 ? 0 : 255);
        }

        var watch = Stopwatch.StartNew();
        OcrPage page = ImageOcr.RecognizeGray(gray, width, height, new OcrOptions { MinConfidence = 0 });
        Assert.True(watch.Elapsed < Bound, watch.Elapsed.ToString());
        Assert.Equal((width, height), (page.Width, page.Height));
    }
}
