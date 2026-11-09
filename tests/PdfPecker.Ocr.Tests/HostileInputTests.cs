using System.Diagnostics;
using System.Globalization;
using System.Text;
using PdfPecker.Ocr.Tests.EndToEnd;
using Xunit.Abstractions;

namespace PdfPecker.Ocr.Tests;

/// <summary>
/// Hostile input and the end-to-end tests share the processors with nothing else: timed tests run alone, so a busy
/// machine running other tests at the same time cannot fail them.
/// </summary>
[CollectionDefinition(nameof(HostileInputCollection), DisableParallelization = true)]
public sealed class HostileInputCollection;

/// <summary>
/// The public entry points with the real engine on input that is wrong, damaged or built to cost time: each must fail
/// with an exception the API documents, or finish within a time bound with a bounded result.
/// </summary>
[Collection(nameof(HostileInputCollection))]
public sealed class HostileInputTests
{
    private readonly ITestOutputHelper _output;

    public HostileInputTests(ITestOutputHelper output) => _output = output;

    /// <summary>The bound for work that reads a whole page with the models.</summary>
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(60);

    /// <summary>The bound for input that must be refused or read without much work.</summary>
    private static readonly TimeSpan Quick = TimeSpan.FromSeconds(20);

    /// <summary>What a damaged image may throw, as <see cref="ImageOcr.Recognize(byte[], OcrOptions?, CancellationToken)"/> documents it.</summary>
    private static readonly Type[] ImageFailures = [typeof(InvalidDataException), typeof(PdfLimitException)];

    /// <summary>What a damaged PDF may throw, as <see cref="PdfOcr.Recognize(byte[], OcrOptions?, CancellationToken)"/> documents it.</summary>
    private static readonly Type[] PdfFailures = [typeof(InvalidDataException), typeof(PdfLimitException), typeof(PdfPasswordException), typeof(NotSupportedException)];

    /// <summary>
    /// Runs <paramref name="work"/> on a task of its own and fails the test when it is not done within
    /// <paramref name="bound"/>, cancelling the token then so the work stops instead of running on behind the other tests.
    /// The token is only cancelled after the bound has passed, so a slow input fails rather than ending in an
    /// <see cref="OperationCanceledException"/> that looks documented. The work's own exception is thrown as it is.
    /// </summary>
    private static T Bounded<T>(Func<CancellationToken, T> work, TimeSpan? bound = null)
    {
        TimeSpan limit = bound ?? Bound;
        using var cts = new CancellationTokenSource();
        var watch = Stopwatch.StartNew();
        Task<T> task = Task.Run(() => work(cts.Token));
        if (!((IAsyncResult)task).AsyncWaitHandle.WaitOne(limit))
        {
            cts.Cancel();
            Assert.Fail($"Did not finish within {limit.TotalSeconds:0} s.");
        }

        return task.GetAwaiter().GetResult();
    }

    /// <summary>
    /// Runs <paramref name="work"/> within the bound; it may succeed, or throw one of <paramref name="allowed"/> (or a
    /// subclass), and nothing else.
    /// </summary>
    private Exception? Documented(string label, Func<CancellationToken, object?> work, Type[] allowed, TimeSpan? bound = null)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            object? result = Bounded(work, bound);
            _output.WriteLine($"{label}: read in {watch.ElapsedMilliseconds} ms ({result})");
            return null;
        }
        catch (Exception ex) when (allowed.Any(t => t.IsInstanceOfType(ex)))
        {
            _output.WriteLine($"{label}: {ex.GetType().Name} in {watch.ElapsedMilliseconds} ms: {ex.Message}");
            return ex;
        }
        catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
        {
            Assert.Fail($"{label}: {ex.GetType().Name} is not documented: {ex.Message}\n{ex.StackTrace}");
            return ex;
        }
    }

    // ---- Raw pixel buffers ----

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
    [InlineData(10, 10, 99)]
    [InlineData(int.MaxValue, int.MaxValue, 100)]
    [InlineData(65536, 65536, 1 << 20)]
    [InlineData(46341, 46341, 1 << 16)]
    [InlineData(int.MaxValue, 2, 1 << 16)]
    [InlineData(1, int.MaxValue, 1 << 16)]
    public void RawPixelBuffersMustHoldTheImage(int width, int height, int length)
    {
        Assert.Throws<ArgumentException>(() => ImageOcr.RecognizeGray(new byte[length], width, height));
        Assert.Throws<ArgumentException>(() => ImageOcr.RecognizeRgba(new byte[length], width, height));
    }

    [Fact]
    public void RgbaBufferOfGrayLengthIsTooShort()
    {
        // A gray buffer passed as RGBA holds a quarter of the bytes the stated size needs.
        var gray = new byte[200 * 100];
        Assert.Throws<ArgumentException>(() => ImageOcr.RecognizeRgba(gray, 200, 100));
        Assert.Throws<ArgumentException>(() => ImageOcr.RecognizeRgba(new byte[(200 * 100 * 4) - 1], 200, 100));
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
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.RecognizeGray(new byte[100], 10, 10, new OcrOptions { MinConfidence = float.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.RecognizeRgba(new byte[400], 10, 10, new OcrOptions { MaxDegreeOfParallelism = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.Recognize(TestPages.Small.Png, new OcrOptions { Dpi = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImageOcr.Recognize(TestPages.Small.Png, new OcrOptions { Dpi = float.PositiveInfinity }));
        Assert.Throws<ArgumentOutOfRangeException>(() => PdfOcr.Recognize(TestPages.Small.Pdf, new OcrOptions { MaxPixels = 9_999 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => PdfOcr.Recognize(TestPages.Small.Pdf, new OcrOptions { PageMode = (OcrPageMode)17 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => PdfOcr.Recognize(TestPages.Small.Pdf, new OcrOptions { Pages = PdfPageRange.Parse("2") }));
        using (var unreadable = new UnreadableStream())
        {
            Assert.Throws<ArgumentException>(() => ImageOcr.Recognize(unreadable));
            Assert.Throws<ArgumentException>(() => PdfOcr.Recognize(unreadable));
            Assert.Throws<ArgumentException>(() => PdfOcr.MakeSearchable(unreadable, new MemoryStream()));
        }

        Assert.Throws<FileNotFoundException>(() => ImageOcr.RecognizeFile(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".png")));
    }

    // ---- Image files ----

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

    public static TheoryData<string> ImageFormats() => new() { "png", "jpeg", "tiff", "webp", "gif", "bmp" };

    /// <summary>
    /// Each format cut short at the header and in the middle, and changed at fixed places in its header and data: every
    /// variant is read or refused with a documented exception, quickly. The pages are small and MaxPixels low, so a
    /// variant that still decodes costs little to read.
    /// </summary>
    [Theory]
    [MemberData(nameof(ImageFormats))]
    public void TruncatedAndDamagedImagesFailWithDocumentedExceptions(string format)
    {
        byte[] image = SampleImages.Get(format);
        var options = new OcrOptions { MaxPixels = 200_000 };
        Assert.NotEmpty(Bounded(ct => ImageOcr.Recognize(image, options, ct), Quick).Text);

        var variants = new List<(string Label, byte[] Data)>();
        foreach (int length in new[] { 1, 4, 8, 16, 33, 64, image.Length / 4, image.Length / 2, image.Length - 16, image.Length - 1 })
        {
            if (length > 0 && length < image.Length) variants.Add(($"cut at {length}", image[..length]));
        }

        uint state = 0x2545F491 ^ (uint)format.Length;
        for (int i = 0; i < 16; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Half the changes land in the first 64 bytes, where the header's sizes, types and offsets are.
            int at = (int)(i % 2 == 0 ? state % (uint)Math.Min(64, image.Length) : state % (uint)image.Length);
            byte[] changed = (byte[])image.Clone();
            changed[at] = (byte)(i % 4 < 2 ? changed[at] ^ 0xFF : (i % 4 == 2 ? 0 : 0x7F));
            variants.Add(($"byte {at} changed", changed));
        }

        foreach (var (label, data) in variants)
        {
            Documented($"{format}, {label}", ct => ImageOcr.Recognize(data, options, ct), ImageFailures, Quick);
        }
    }

    [Fact]
    public void OnePixelImagesHaveNoText()
    {
        byte[] white = [255, 255, 255];
        foreach (byte[] image in new[] { TestImages.Bmp(white, 1, 1, 11811), TestImages.Gif([0], 1, 1), TestImages.Tiff(1, 1, [128]), TestImages.ZeroPng(1, 1, 8, 1) })
        {
            OcrPage page = Bounded(ct => ImageOcr.Recognize(image, null, ct), Quick);
            Assert.Equal((1, 1), (page.Width, page.Height));
            Assert.Empty(page.Lines);
        }

        Assert.Empty(ImageOcr.RecognizeGray([0], 1, 1).Lines);
        Assert.Empty(ImageOcr.RecognizeRgba([0, 0, 0, 0], 1, 1).Lines);
    }

    [Fact]
    public void ZeroSizeImagesAreRefused()
    {
        byte[] png = TestPages.Small.Png;
        foreach (var (label, image) in new[]
        {
            ("png 0 x 0", TestImages.WithPngSize(png, 0, 0)),
            ("png 0 wide", TestImages.WithPngSize(png, 0, 10)),
            ("tiff 0 x 0", TestImages.Tiff(0, 0, [])),
            ("tiff 0 high", TestImages.Tiff(10, 0, new byte[10])),
            ("gif 0 x 0", TestImages.Gif([], 0, 0)),
            ("bmp 0 x 0", TestImages.Bmp([], 0, 0, 11811)),
        })
        {
            Exception? ex = Documented(label, ct => ImageOcr.Recognize(image, null, ct), ImageFailures, Quick);
            Assert.True(ex != null, $"{label} was read as an image.");
        }
    }

    /// <summary>
    /// Files whose headers state billions of pixels, with little or no data behind them: they must be refused or read
    /// within <see cref="OcrOptions.MaxPixels"/> without decoding or allocating the stated size.
    /// </summary>
    public static TheoryData<string> Bombs() => new()
    {
        "png-header-only", "png-zeros-1bit", "png-zeros-8bit", "jpeg-header", "bmp-header", "gif-screen", "tiff-strip", "webp-header",
    };

    [Theory]
    [MemberData(nameof(Bombs))]
    public void HugeDeclaredSizesAreRefusedQuickly(string kind)
    {
        byte[] image = BombImage(kind);
        long before = GC.GetTotalAllocatedBytes(precise: false);
        var watch = Stopwatch.StartNew();
        Exception? ex = Documented(kind, ct => ImageOcr.Recognize(image, null, ct), ImageFailures, Quick);
        long allocated = GC.GetTotalAllocatedBytes(precise: false) - before;
        // Reading within the default limit of 100 million pixels may take a few hundred megabytes; decoding the stated
        // size would take gigabytes.
        Assert.True(allocated < 1_500_000_000L, $"{kind}: {allocated / 1_000_000} MB allocated in {watch.Elapsed.TotalSeconds:0.0} s ({ex?.GetType().Name ?? "read"}).");
    }

    private static byte[] BombImage(string kind)
    {
        byte[] png = SampleImages.Get("png");
        switch (kind)
        {
            case "png-header-only":
                return TestImages.WithPngSize(png, 100_000, 100_000);
            case "png-zeros-1bit":
                // 50,000 by 50,000 pixels, 2.5 billion, in about 300 kB: every row is there, all zeros.
                return TestImages.ZeroPng(50_000, 50_000, 1, 50_000);
            case "png-zeros-8bit":
                // 30,000 by 30,000 gray pixels whose data ends after 2,000 rows.
                return TestImages.ZeroPng(30_000, 30_000, 8, 2_000);
            case "jpeg-header":
            {
                byte[] jpeg = (byte[])SampleImages.Get("jpeg").Clone();
                for (int i = 2; i + 8 < jpeg.Length; i++)
                {
                    // The first start-of-frame marker holds the height and width as 16-bit numbers.
                    if (jpeg[i] == 0xFF && jpeg[i + 1] is 0xC0 or 0xC1 or 0xC2)
                    {
                        jpeg[i + 5] = jpeg[i + 6] = jpeg[i + 7] = jpeg[i + 8] = 0xFF;
                        return jpeg;
                    }
                }

                throw new InvalidOperationException("The sample JPEG has no start-of-frame marker.");
            }

            case "bmp-header":
            {
                byte[] bmp = (byte[])SampleImages.Get("bmp").Clone();
                BitConverter.TryWriteBytes(bmp.AsSpan(18), 60_000);
                BitConverter.TryWriteBytes(bmp.AsSpan(22), 60_000);
                return bmp;
            }

            case "gif-screen":
            {
                byte[] gif = (byte[])SampleImages.Get("gif").Clone();
                // The logical screen at offset 6 and the image descriptor after the 768-byte palette both go to 65,535.
                gif[6] = gif[7] = gif[8] = gif[9] = 0xFF;
                int descriptor = 13 + 768;
                gif[descriptor + 5] = gif[descriptor + 6] = gif[descriptor + 7] = gif[descriptor + 8] = 0xFF;
                return gif;
            }

            case "tiff-strip":
                return TestImages.Tiff(1_000_000, 1_000_000, new byte[4096]);
            case "webp-header":
            {
                byte[] webp = (byte[])SampleImages.Get("webp").Clone();
                int vp8l = IndexOf(webp, "VP8L"u8);
                if (vp8l >= 0)
                {
                    // The lossless header's width and height minus one are 14 bits each after the signature byte.
                    webp[vp8l + 9] = 0xFF;
                    webp[vp8l + 10] = 0xFF;
                    webp[vp8l + 11] = 0xFF;
                    webp[vp8l + 12] = (byte)(webp[vp8l + 12] | 0x0F);
                    return webp;
                }

                int vp8 = IndexOf(webp, "VP8 "u8);
                Assert.True(vp8 >= 0, "The sample WebP has neither a VP8L nor a VP8 chunk.");
                // The lossy key frame's width and height are 14 bits each after its start code.
                webp[vp8 + 14] = 0xFF;
                webp[vp8 + 15] = 0x3F;
                webp[vp8 + 16] = 0xFF;
                webp[vp8 + 17] = 0x3F;
                return webp;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }

    [Fact]
    public void ImageLargerThanMaxPixelsIsReadSmaller()
    {
        byte[] png = SampleImages.Get("png");
        var options = new OcrOptions { MaxPixels = 10_000 };
        OcrPage page = Bounded(ct => ImageOcr.Recognize(png, options, ct), Quick);
        Assert.True((long)page.Width * page.Height <= options.MaxPixels, page.ToString());
        Assert.True(page.Dpi < options.Dpi);
    }

    /// <summary>
    /// Files whose header states a small image around a large one that PdfPecker decodes at its own size: a BMP that
    /// wraps a PNG, and a WebP whose VP8X canvas is smaller than its frame. The small-image path, which renders at
    /// three times the size, must not be taken on the header's word, and the page read must stay within MaxPixels.
    /// </summary>
    public static TheoryData<string> SmallHeaders() => new() { "bmp-png", "webp-canvas" };

    [Theory]
    [MemberData(nameof(SmallHeaders))]
    public void HeadersSmallerThanTheImageStayWithinMaxPixels(string kind)
    {
        byte[] image = kind switch
        {
            "bmp-png" => TestImages.WrappingBmp(LargeBlank.Value.ToPng(), 5, 1, 1),
            "webp-canvas" => TestImages.WebPWithCanvas(LargeBlank.Value.ToWebP(), 2, 2),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        var options = new OcrOptions { MaxPixels = 10_000, MaxDegreeOfParallelism = 1 };

        long before = GC.GetTotalAllocatedBytes(precise: true);
        Exception? ex = Documented(kind, ct =>
        {
            OcrPage page = ImageOcr.Recognize(image, options, ct);
            Assert.True((long)page.Width * page.Height <= options.MaxPixels, page.ToString());
            return page;
        }, ImageFailures, Quick);
        long recognized = GC.GetTotalAllocatedBytes(precise: true) - before;
        Exception? searchable = Documented(kind + " searchable", ct => ImageOcr.ToSearchablePdf([image], options, ct).Length, ImageFailures, Quick);
        long both = GC.GetTotalAllocatedBytes(precise: true) - before;
        _output.WriteLine($"{kind}: {recognized / 1_000_000} MB to read, {(both - recognized) / 1_000_000} MB for the searchable PDF");

        // Decoding the 3,000 by 3,000 image takes about a hundred megabytes; rendering it at three times the size, as
        // the small-image path does, would take more than a gigabyte.
        Assert.True(recognized < 500_000_000L, $"{kind}: {recognized / 1_000_000} MB allocated to read ({ex?.GetType().Name ?? "read"}).");
        Assert.True(both - recognized < 500_000_000L, $"{kind}: {(both - recognized) / 1_000_000} MB allocated for the searchable PDF ({searchable?.GetType().Name ?? "made"}).");
    }

    /// <summary>A white image of 3,000 by 3,000 pixels, which PNG and WebP store in a few kilobytes.</summary>
    private static readonly Lazy<RenderedImage> LargeBlank = new(() =>
        PdfRenderer.Open(HtmlToPdf.Convert(TestPages.Document("2250pt 2250pt", "", ""))).RenderPage(0, new ImageOptions { Dpi = 96 }));

    [Fact]
    public void SearchablePdfOfDamagedImagesFailsWithDocumentedExceptions()
    {
        byte[] png = SampleImages.Get("png");
        Type[] allowed = [.. ImageFailures, typeof(ArgumentException)];
        Documented("truncated", ct => ImageOcr.ToSearchablePdf([png, png[..(png.Length / 2)]], null, ct), allowed, Quick);
        Documented("bomb", ct => ImageOcr.ToSearchablePdf([TestImages.WithPngSize(png, 100_000, 100_000)], null, ct), allowed, Quick);
        Assert.Throws<ArgumentException>(() => ImageOcr.ToSearchablePdf([]));
        Assert.Throws<ArgumentNullException>(() => ImageOcr.ToSearchablePdf([png, null!]));
    }

    // ---- PDFs ----

    [Fact]
    public void DamagedPdfsFailWithDocumentedExceptions()
    {
        byte[] pdf = TestPages.Small.ScannedPdf();
        var options = new OcrOptions { Dpi = 100 };
        var variants = new List<(string, byte[])>
        {
            ("header only", "%PDF-1.7\n"u8.ToArray()),
            ("no trailer", pdf[..(pdf.Length - 40)]),
            ("half", pdf[..(pdf.Length / 2)]),
            ("first 1000 bytes", pdf[..1000]),
        };
        uint state = 0x1234567;
        for (int i = 0; i < 12; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            byte[] changed = (byte[])pdf.Clone();
            int at = (int)(state % (uint)pdf.Length);
            changed[at] ^= 0x55;
            variants.Add(($"byte {at} changed", changed));
        }

        foreach (var (label, data) in variants)
        {
            Documented(label, ct => PdfOcr.Recognize(data, options, ct), PdfFailures, Quick);
            Documented(label + ", searchable", ct => PdfOcr.MakeSearchable(data, options, ct), [.. PdfFailures, typeof(InvalidOperationException)], Quick);
        }
    }

    [Fact]
    public void EncryptedPdfNeedsItsPassword()
    {
        byte[] pdf = TestPages.Small.ScannedPdf();
        byte[] encrypted = PdfProtector.Encrypt(pdf, new PdfSecurity { UserPassword = "secret", OwnerPassword = "owner" }, null);
        var options = new OcrOptions { Dpi = 150 };
        Assert.Throws<PdfPasswordException>(() => Bounded(ct => PdfOcr.Recognize(encrypted, options, ct), Quick));
        Assert.Throws<PdfPasswordException>(() => Bounded(ct => PdfOcr.MakeSearchable(encrypted, options, ct), Quick));
        Assert.Throws<PdfPasswordException>(() => Bounded(ct => PdfOcr.Recognize(encrypted, new OcrOptions { Dpi = 150, Password = "wrong" }, ct), Quick));
        OcrDocument read = Bounded(ct => PdfOcr.Recognize(encrypted, new OcrOptions { Dpi = 300, Password = "secret" }, ct));
        Assert.Equal(TestPages.SmallText, read.Text);
    }

    /// <summary>
    /// Ten thousand empty pages: the text check and rendering of each is small but not free, so the bound is on the
    /// whole document. Each page used to cost over 100 ms in the detector, which scales small pages up to 736 pixels a
    /// side; blank pages are now skipped (see <see cref="UniformPagesHaveNoLinesAndCostLittle"/>).
    /// </summary>
    [Fact]
    public void TenThousandEmptyPagesFinishWithinTheBound()
    {
        byte[] pdf = ManyPages(10_000, "0 0 100 100", "");
        var watch = Stopwatch.StartNew();
        OcrDocument document = Bounded(ct => PdfOcr.Recognize(pdf, new OcrOptions { Dpi = 72 }, ct));
        Assert.Equal(10_000, document.Pages.Count);
        Assert.All(document.Pages, p => Assert.Empty(p.Lines));
        Assert.Equal(string.Empty, document.Text.Trim('\n', '\f'));
        _output.WriteLine($"10,000 empty pages in {watch.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void ManyPagesStopWhenCancelled()
    {
        byte[] pdf = ManyPages(10_000, "0 0 595 842", "0 g 100 700 300 20 re f");
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        var watch = Stopwatch.StartNew();
        Assert.ThrowsAny<OperationCanceledException>(() => Bounded(_ => PdfOcr.Recognize(pdf, new OcrOptions { Dpi = 150 }, cts.Token), Quick));
        Assert.ThrowsAny<OperationCanceledException>(() => Bounded(_ => PdfOcr.MakeSearchable(pdf, new OcrOptions { Dpi = 150 }, cts.Token), Quick));
        Assert.True(watch.Elapsed < Quick, watch.Elapsed.ToString());
    }

    /// <summary>
    /// Page boxes from huge to nothing: a huge page is read at a resolution that keeps it within MaxPixels, and a page
    /// with no area or a box that is not a number is still read or refused as documented.
    /// </summary>
    [Theory]
    [InlineData("0 0 14400 14400")]
    [InlineData("0 0 1e9 1e9")]
    [InlineData("0 0 14400 1")]
    [InlineData("0 0 1 14400")]
    [InlineData("0 0 0 0")]
    [InlineData("10 10 10 10")]
    [InlineData("0 0 -500 -500")]
    [InlineData("0 0 1e400 5")]
    [InlineData("0 0 nan 5")]
    public void ExtremeMediaBoxesStayWithinMaxPixels(string mediaBox)
    {
        byte[] pdf = ManyPages(1, mediaBox, "0 g 10 10 50 5 re f");
        var options = new OcrOptions { MaxPixels = 4_000_000 };
        OcrDocument? document = null;
        Documented(mediaBox, ct => document = PdfOcr.Recognize(pdf, options, ct), PdfFailures);
        if (document != null)
        {
            OcrPage page = Assert.Single(document.Pages);
            Assert.True((long)page.Width * page.Height <= options.MaxPixels, page.ToString());
        }

        Documented(mediaBox + ", searchable", ct => PdfOcr.MakeSearchable(pdf, new OcrOptions { MaxPixels = 4_000_000, PageMode = OcrPageMode.Force }, ct), [.. PdfFailures, typeof(InvalidOperationException)]);
    }

    /// <summary>
    /// A MaxPixels above what PdfPecker's renderer can hand over (2 GB of RGBA) on a page that needs it: the page must be
    /// refused with PdfLimitException before rendering, not end in an OverflowException inside the renderer.
    /// </summary>
    [Fact]
    public void PagesAboveTheRenderersLimitAreRefused()
    {
        // 200 by 200 inches at 150 dpi: 900 million pixels, brought down to 700 million by MaxPixels.
        byte[] pdf = ManyPages(1, "0 0 14400 14400", "0 g 10 10 50 5 re f");
        var options = new OcrOptions { MaxPixels = 700_000_000, Dpi = 150, PageMode = OcrPageMode.Force };
        Assert.Throws<PdfLimitException>(() => Bounded(ct => PdfOcr.Recognize(pdf, options, ct), Quick));
        Assert.Throws<PdfLimitException>(() => Bounded(ct => PdfOcr.MakeSearchable(pdf, options, ct), Quick));
    }

    [Fact]
    public void HugePageAtTheDefaultLimitFinishes()
    {
        // 200 by 200 inches at 300 dpi would be 3.6 billion pixels; the default limit reads it at about 50 dpi.
        byte[] pdf = ManyPages(1, "0 0 14400 14400", "0 g 1000 13000 9000 300 re f");
        OcrDocument document = Bounded(ct => PdfOcr.Recognize(pdf, null, ct));
        OcrPage page = Assert.Single(document.Pages);
        Assert.True((long)page.Width * page.Height <= 100_000_000, page.ToString());
    }

    /// <summary>Images of one pixel by many on a page, stretched over it, as a damaged scan might hold.</summary>
    [Theory]
    [InlineData(20000, 1)]
    [InlineData(1, 30000)]
    [InlineData(3, 3)]
    [InlineData(100000, 100000)]
    public void PdfImagesOfExtremeShapesFinish(int width, int height)
    {
        byte[] pdf = PdfWithImage(width, height);
        Documented($"{width} x {height}", ct => PdfOcr.Recognize(pdf, new OcrOptions { Dpi = 150 }, ct), PdfFailures);
        Documented($"{width} x {height}, searchable", ct => PdfOcr.MakeSearchable(pdf, new OcrOptions { Dpi = 150 }, ct), [.. PdfFailures, typeof(InvalidOperationException)]);
    }

    // ---- Pixels that cost time ----

    /// <summary>
    /// Pixels that cost the most per pixel at A4 and 300 dpi: noise, thin stripes and a fine checkerboard give the
    /// detector many regions to look at, and an all-black page one region the size of the page.
    /// </summary>
    [Theory]
    [InlineData("noise")]
    [InlineData("stripes")]
    [InlineData("vertical-stripes")]
    [InlineData("checkerboard")]
    [InlineData("black")]
    [InlineData("text-like")]
    public void CostlyPixelsFinishWithinTheBound(string kind)
    {
        const int W = 2480, H = 3508;
        byte[] gray = CostlyPixels(kind, W, H);
        var watch = Stopwatch.StartNew();
        OcrPage page = Bounded(ct => ImageOcr.RecognizeGray(gray, W, H, new OcrOptions { MinConfidence = 0 }, ct));
        Assert.True(page.Lines.Count <= 2000, $"{kind}: {page.Lines.Count} lines");
        Assert.All(page.Lines, l => Assert.True(OcrGeometry.IsFinite(l.Quad)));
        Assert.All(page.Lines, l => Assert.InRange(l.Confidence, 0f, 1f));
    }

    private static byte[] CostlyPixels(string kind, int w, int h)
    {
        var gray = new byte[w * h];
        uint state = 7;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                gray[(y * w) + x] = kind switch
                {
                    "noise" => (byte)state,
                    "stripes" => (byte)((y / 2) % 2 == 0 ? 0 : 255),
                    "vertical-stripes" => (byte)(x % 3 == 0 ? 0 : 255),
                    "checkerboard" => (byte)(((x / 3) + (y / 3)) % 2 == 0 ? 0 : 255),
                    // Short dark dashes in rows, like lines of text everywhere: the most boxes for recognition.
                    "text-like" => (byte)((y % 40) < 22 && (x % 37) < 25 ? 0 : 255),
                    // Black but for one white pixel in 997, so the page is not skipped as blank and the detector
                    // sees one region the size of the page.
                    _ => (byte)(((y * w) + x) % 997 == 0 ? 255 : 0),
                };
            }
        }

        return gray;
    }

    /// <summary>
    /// Pages of one colour: the detector used to run on them at 736 pixels a side or more, which cost over 100 ms for
    /// any blank page (ten thousand of them took more than two minutes), and on a pure green page it found a box that
    /// read as "1" at a low confidence.
    /// </summary>
    [Fact]
    public void UniformPagesHaveNoLinesAndCostLittle()
    {
        var options = new OcrOptions { MinConfidence = 0 };
        foreach (var (w, h) in new[] { (32, 32), (100, 100), (1240, 1754) })
        {
            var green = new byte[w * h * 4];
            for (int i = 0; i < w * h; i++)
            {
                green[(i * 4) + 1] = 255;
                green[(i * 4) + 3] = 255;
            }

            Assert.Empty(ImageOcr.RecognizeRgba(green, w, h, options).Lines);
            var black = new byte[w * h];
            Assert.Empty(ImageOcr.RecognizeGray(black, w, h, options).Lines);
        }

        var white = new byte[100 * 100];
        white.AsSpan().Fill(255);
        ImageOcr.RecognizeGray(white, 100, 100);
        var watch = Stopwatch.StartNew();
        for (int i = 0; i < 100; i++) ImageOcr.RecognizeGray(white, 100, 100);
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(2), $"100 blank pages took {watch.Elapsed}.");

        // One pixel that differs is a page to read, not a blank one.
        white[5050] = 0;
        Assert.False(Layout.OcrEngine.IsUniform(new Detection.PixelView(white, 100, 100, 1)));
        white[5050] = 255;
        Assert.True(Layout.OcrEngine.IsUniform(new Detection.PixelView(white, 100, 100, 1)));
        Assert.True(Layout.OcrEngine.IsUniform(new Detection.PixelView(new byte[3], 1, 1, 3)));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 20000)]
    [InlineData(20000, 1)]
    [InlineData(30000, 3)]
    [InlineData(2, 2)]
    [InlineData(100_000, 40)]
    public void ExtremeShapesFinish(int width, int height)
    {
        var gray = new byte[width * height];
        for (int i = 0; i < gray.Length; i++)
        {
            gray[i] = (byte)(i % 7 == 0 ? 0 : 255);
        }

        OcrPage page = Bounded(ct => ImageOcr.RecognizeGray(gray, width, height, new OcrOptions { MinConfidence = 0 }, ct));
        Assert.Equal((width, height), (page.Width, page.Height));
    }

    // ---- Cancellation ----

    [Fact]
    public void CancellationStopsAPageQuickly()
    {
        RenderedPage invoice = TestPages.Invoice;
        byte[] gray = invoice.Gray();
        foreach (int delay in new[] { 50, 400, 900 })
        {
            using var cts = new CancellationTokenSource();
            var watch = Stopwatch.StartNew();
            cts.CancelAfter(delay);
            try
            {
                Bounded(_ => ImageOcr.RecognizeGray(gray, invoice.Width, invoice.Height, new OcrOptions { MaxDegreeOfParallelism = 2 }, cts.Token), Quick);
            }
            catch (OperationCanceledException)
            {
                // Cancelled as asked; how soon after the token fired is what matters.
            }

            Assert.True(watch.ElapsedMilliseconds < delay + 3000, $"Cancelled after {delay} ms, stopped after {watch.ElapsedMilliseconds} ms.");
        }

        // The pool still works after cancelled runs.
        Assert.Equal(TestPages.SmallText, ImageOcr.RecognizeRgba(TestPages.Small.Rgba, TestPages.Small.Width, TestPages.Small.Height).Text);
    }

    [Fact]
    public void CancellationStopsMakeSearchable()
    {
        byte[] pdf = TestPages.Invoice.ScannedPdf();
        using var cts = new CancellationTokenSource(300);
        var watch = Stopwatch.StartNew();
        Assert.ThrowsAny<OperationCanceledException>(() => Bounded(_ => PdfOcr.MakeSearchable(pdf, null, cts.Token), Quick));
        Assert.True(watch.ElapsedMilliseconds < 4000, watch.ElapsedMilliseconds.ToString(CultureInfo.InvariantCulture));
    }

    // ---- Helpers ----

    /// <summary>A PDF of <paramref name="count"/> pages of one media box, each with its own content stream, written in one pass.</summary>
    private static byte[] ManyPages(int count, string mediaBox, string content)
    {
        var pdf = new StringBuilder("%PDF-1.7\n");
        var offsets = new List<int>();
        void Object(string body)
        {
            offsets.Add(pdf.Length);
            pdf.Append(CultureInfo.InvariantCulture, $"{offsets.Count} 0 obj\n{body}\nendobj\n");
        }

        Object("<< /Type /Catalog /Pages 2 0 R >>");
        var kids = new StringBuilder();
        for (int i = 0; i < count; i++) kids.Append(CultureInfo.InvariantCulture, $"{3 + (2 * i)} 0 R ");
        Object($"<< /Type /Pages /Count {count} /Kids [{kids}] >>");
        for (int i = 0; i < count; i++)
        {
            Object($"<< /Type /Page /Parent 2 0 R /MediaBox [{mediaBox}] /Resources << >> /Contents {4 + (2 * i)} 0 R >>");
            Object($"<< /Length {content.Length} >>\nstream\n{content}\nendstream");
        }

        return Finish(pdf, offsets);
    }

    /// <summary>An A4 page whose only content is an 8-bit gray image of the given size, stretched over the page; the data is a short pattern.</summary>
    private static byte[] PdfWithImage(int width, int height)
    {
        long size = (long)width * height;
        // Huge images get only part of their data, as a damaged or hostile file would hold.
        int dataLength = (int)Math.Min(size, 1 << 20);
        var data = new StringBuilder(dataLength);
        for (int i = 0; i < dataLength; i++) data.Append((char)(i % 5 == 0 ? 0 : 255));
        var pdf = new StringBuilder("%PDF-1.7\n");
        var offsets = new List<int>();
        void Object(string body)
        {
            offsets.Add(pdf.Length);
            pdf.Append(CultureInfo.InvariantCulture, $"{offsets.Count} 0 obj\n{body}\nendobj\n");
        }

        const string Content = "q 595 0 0 842 0 0 cm /Im1 Do Q";
        Object("<< /Type /Catalog /Pages 2 0 R >>");
        Object("<< /Type /Pages /Count 1 /Kids [3 0 R] >>");
        Object("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>");
        Object($"<< /Length {Content.Length} >>\nstream\n{Content}\nendstream");
        Object($"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8 /Length {dataLength} >>\nstream\n{data}\nendstream");
        return Finish(pdf, offsets);
    }

    /// <summary>Appends the cross-reference table and trailer; the objects are Latin-1, one byte per character.</summary>
    private static byte[] Finish(StringBuilder pdf, List<int> offsets)
    {
        int xref = pdf.Length;
        pdf.Append(CultureInfo.InvariantCulture, $"xref\n0 {offsets.Count + 1}\n0000000000 65535 f \n");
        foreach (int offset in offsets) pdf.Append(offset.ToString("D10", CultureInfo.InvariantCulture)).Append(" 00000 n \n");
        pdf.Append(CultureInfo.InvariantCulture, $"trailer\n<< /Size {offsets.Count + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n");
        return Encoding.Latin1.GetBytes(pdf.ToString());
    }

    private static int IndexOf(byte[] data, ReadOnlySpan<byte> value) => data.AsSpan().IndexOf(value);

    private sealed class UnreadableStream : MemoryStream
    {
        public override bool CanRead => false;
    }
}

/// <summary>
/// One small page of text in each image format PdfPecker reads: PNG, JPEG, WebP and TIFF written by PdfPecker, GIF and
/// BMP by <see cref="TestImages"/>. They are about 240 by 80 pixels, so a variant that still decodes is quick to read.
/// </summary>
internal static class SampleImages
{
    private static readonly Lazy<Dictionary<string, byte[]>> Images = new(Create);

    public static byte[] Get(string format) => Images.Value[format];

    private static Dictionary<string, byte[]> Create()
    {
        byte[] pdf = HtmlToPdf.Convert(TestPages.Document("60mm 20mm", "p{font:10pt sans-serif;margin:0}", "<p>Sample 42</p><p>Text page</p>"));
        RenderedImage image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = 100 });
        var rgb = new byte[image.Width * image.Height * 3];
        var gray = new byte[image.Width * image.Height];
        for (int i = 0; i < gray.Length; i++)
        {
            image.Rgba.AsSpan(i * 4, 3).CopyTo(rgb.AsSpan(i * 3));
            gray[i] = image.Rgba[(i * 4) + 1];
        }

        return new Dictionary<string, byte[]>(StringComparer.Ordinal)
        {
            ["png"] = image.ToPng(),
            ["jpeg"] = image.ToJpeg(90),
            ["webp"] = image.ToWebP(null),
            ["tiff"] = image.ToTiff(null),
            ["gif"] = TestImages.Gif(gray, image.Width, image.Height),
            ["bmp"] = TestImages.Bmp(rgb, image.Width, image.Height, 3937),
        };
    }
}
