using System.Diagnostics;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>
/// The pipeline on degenerate and hostile input: probability maps that are empty, full, noise, a checkerboard or not
/// numbers; one-pixel and extremely long images; boxes that collapse or leave the page. Each must finish quickly with
/// a bounded result and no exception.
/// </summary>
public sealed class PipelineHostileInputTests
{
    private static readonly CharacterSet Characters = new(PipelineFixture.Characters[1..^1]);
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(20);

    public static TheoryData<string> Maps => ["zero", "one", "noise", "checkerboard", "stripes", "nan", "infinity", "dots"];

    [Theory]
    [MemberData(nameof(Maps))]
    public void PostProcessingIsBoundedOnHostileMaps(string kind)
    {
        const int W = 1536, H = 1536;
        float[] map = MakeMap(kind, W, H);
        var processor = new DbPostProcessor();
        var watch = Stopwatch.StartNew();
        List<DetectedQuad> boxes = processor.Process(map, W, H, 3000, 3000, new DetectionOptions());
        Assert.True(watch.Elapsed < Bound, $"{kind}: {watch.Elapsed}");
        Assert.True(boxes.Count <= new DetectionOptions().MaxCandidates);
        foreach (DetectedQuad box in boxes)
        {
            for (int i = 0; i < 4; i++)
            {
                Assert.InRange(box.Quad[i].X, 0, 2999);
                Assert.InRange(box.Quad[i].Y, 0, 2999);
            }
        }

        switch (kind)
        {
            case "zero":
            case "nan":
                Assert.Empty(boxes);
                break;
            case "one":
            case "infinity":
                Assert.Single(boxes);
                break;
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 10000)]
    [InlineData(10000, 1)]
    [InlineData(20000, 3)]
    [InlineData(3, 20000)]
    [InlineData(31, 29)]
    public void DetectorHandlesTinyAndExtremeImages(int width, int height)
    {
        var options = new DetectionOptions();
        long maxPixels = 0;
        var network = new FunctionNetwork((input, shape) =>
        {
            maxPixels = Math.Max(maxPixels, (long)shape[2] * shape[3]);
            var map = new float[shape[2] * shape[3]];
            Array.Fill(map, 0.9f);
            return (map, [1, 1, shape[2], shape[3]]);
        });

        var image = new byte[width * height];
        var watch = Stopwatch.StartNew();
        List<DetectedQuad> boxes = new TextDetector(network, options).Detect(new PixelView(image, width, height, 1));
        Assert.True(watch.Elapsed < Bound);

        // Rounding to multiples of 32 can add a little to the bound.
        Assert.True(maxPixels <= options.MaxInputPixels * 1.1, $"{maxPixels} input pixels");
        foreach (DetectedQuad box in boxes)
        {
            for (int i = 0; i < 4; i++)
            {
                Assert.InRange(box.Quad[i].X, 0, width);
                Assert.InRange(box.Quad[i].Y, 0, height);
            }
        }
    }

    [Fact]
    public void RecognizerHandlesDegenerateBoxes()
    {
        var page = new PixelView(new byte[100 * 50 * 3], 100, 50, 3);
        Quad[] quads =
        [
            new(new(5, 5), new(5, 5), new(5, 5), new(5, 5)),
            new(new(-500, -500), new(-400, -500), new(-400, -450), new(-500, -450)),
            new(new(0, 0), new(100000, 0), new(100000, 2), new(0, 2)),
            new(new(10, 10), new(20, 40), new(10, 10), new(20, 40)),
            new(new(0, 0), new(0, 49), new(1, 49), new(1, 0)),
        ];

        int widest = 0;
        var rec = new FunctionNetwork((input, shape) =>
        {
            widest = Math.Max(widest, shape[3]);
            int steps = shape[3] / 8;
            var output = new float[shape[0] * steps * Characters.Count];
            for (int i = 0; i < shape[0] * steps; i++)
            {
                output[(i * Characters.Count) + 1 + (i % 50)] = float.NaN;
                output[(i * Characters.Count) + 2 + (i % 50)] = 0.7f;
            }

            return (output, [shape[0], steps, Characters.Count]);
        });

        var recognizer = new TextRecognizer(rec, ReplayNetworks.Orientation(upsideDown: true), Characters);
        var watch = Stopwatch.StartNew();
        List<RecognizedLine> lines = recognizer.Recognize(page, [.. quads.Select(q => new DetectedQuad(q, 1))]);
        Assert.True(watch.Elapsed < Bound);
        Assert.True(widest <= recognizer.Options.MaxInputWidth);
        Assert.All(lines, l => Assert.All(l.Words, w => Assert.True(float.IsFinite(w.Quad.P0.X) && float.IsFinite(w.Quad.P2.Y))));
        PageResult result = PageAssembler.Assemble(100, 50, 72, lines);
        Assert.Equal(lines.Count, result.Lines.Count);
    }

    [Fact]
    public void WrongNetworkShapesAreReported()
    {
        byte[] pixels = new byte[64 * 64];
        var bad = new FunctionNetwork((input, shape) => (new float[4], [1, 4]));
        Assert.Throws<InvalidOperationException>(() => new TextDetector(bad).Detect(new PixelView(pixels, 64, 64, 1)));
        var recognizer = new TextRecognizer(bad, null, Characters);
        Assert.Throws<InvalidOperationException>(() => recognizer.Recognize(new PixelView(pixels, 64, 64, 1), [new DetectedQuad(LayoutTests.Box(0, 0, 60, 20), 1)]));
    }

    [Fact]
    public void BadPixelBuffersAreArgumentErrors()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PixelView(new byte[4], 0, 4, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PixelView(new byte[4], 2, 2, 2));
        Assert.Throws<ArgumentException>(() => new PixelView(new byte[3], 2, 2, 1));
    }

    [Fact]
    public void CancellationStopsTheDetector()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var network = new FunctionNetwork((input, shape) => (new float[shape[2] * shape[3]], [1, 1, shape[2], shape[3]]));
        Assert.ThrowsAny<OperationCanceledException>(() => new TextDetector(network).Detect(new PixelView(new byte[100 * 100], 100, 100, 1), cts.Token));
    }

    internal static float[] MakeMap(string kind, int w, int h)
    {
        var map = new float[w * h];
        var random = new Random(7);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                map[(y * w) + x] = kind switch
                {
                    "one" => 1f,
                    "noise" => (float)random.NextDouble(),
                    "checkerboard" => ((x + y) & 1) == 0 ? 1f : 0f,
                    "stripes" => (y % 3) == 0 ? 1f : 0f,
                    "nan" => float.NaN,
                    "infinity" => float.PositiveInfinity,
                    "dots" => (x % 4 == 0 && y % 4 == 0) ? 1f : 0f,
                    _ => 0f,
                };
            }
        }

        return map;
    }
}
