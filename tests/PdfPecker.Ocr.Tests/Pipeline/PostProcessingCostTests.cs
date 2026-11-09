using System.Diagnostics;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Recognition;
using Xunit.Abstractions;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>
/// The time and allocations of the work around the networks on a full page, written to the test output. The bounds
/// are loose on purpose, so a slow CI machine passes; they catch a change that makes the work grow by an order of
/// magnitude, such as a quadratic pass or a buffer allocated per region.
/// </summary>
public sealed class PostProcessingCostTests(ITestOutputHelper output)
{
    private const int Runs = 10;

    [Fact]
    public void DetectionPostProcessingOnAFullPage()
    {
        using PipelineFixture fx = PipelineFixture.Load("invoice");
        float[] map = fx.ProbabilityMap();
        var processor = new DbPostProcessor();
        int count = 0;
        (double ms, long bytes) = Measure(() => count = processor.Process(map, fx.DetInputWidth, fx.DetInputHeight, fx.DetImageWidth, fx.DetImageHeight, DetectionOptions.Rapidocr).Count);
        output.WriteLine($"DB post-processing, {fx.DetInputWidth}x{fx.DetInputHeight} map, {processor.LastRegionCount} regions, {count} boxes: {ms:F2} ms, {bytes / 1024.0:F1} KiB allocated per page");
        Assert.True(ms < 500, $"{ms} ms");

        // The reused buffers leave the boxes, their list and the per-region hull and offset arrays.
        Assert.True(bytes < 4 * 1024 * 1024, $"{bytes} bytes");
    }

    [Fact]
    public void DetectionInputForA300DpiPage()
    {
        const int W = 2480, H = 3508;
        var image = new byte[W * H * 3];
        new Random(3).NextBytes(image);
        DetectionInput.Layout layout = DetectionInput.Plan(W, H, new DetectionOptions());
        var input = new float[3 * layout.InputWidth * layout.InputHeight];
        (double ms, long bytes) = Measure(() => DetectionInput.Fill(new PixelView(image, W, H, 3), layout, input));
        output.WriteLine($"Detection input, A4 at 300 dpi to {layout.InputWidth}x{layout.InputHeight}: {ms:F2} ms, {bytes / 1024.0:F1} KiB allocated per page");
        Assert.True(ms < 1000, $"{ms} ms");
        Assert.True(bytes < 1024 * 1024, $"{bytes} bytes");
    }

    [Fact]
    public void DecodingAndWordsForAPageOfLines()
    {
        using PipelineFixture fx = PipelineFixture.Load("invoice");
        var characters = new CharacterSet(PipelineFixture.Characters[1..^1]);
        List<RecordedLine> lines = RecordedLine.Read(fx);
        List<float[]> boxes = fx.Boxes("det_sorted_boxes");
        var inputs = new List<(float[] Probabilities, int Steps, LineCrop Crop, int Width)>();
        foreach (RecordedLine line in lines)
        {
            var crop = new LineCrop(QuadMath.FromArray(boxes[line.Line]), line.Line);
            inputs.Add((line.Probabilities(characters.Count), line.Steps, crop, line.InputShape[3]));
        }

        int words = 0;
        (double ms, long bytes) = Measure(() =>
        {
            words = 0;
            foreach ((float[] probabilities, int steps, LineCrop crop, int width) in inputs)
            {
                DecodedLine decoded = CtcDecoder.Decode(probabilities, steps, characters.Count, characters);
                int resized = Math.Clamp((int)Math.Ceiling(48 * crop.AspectRatio), 1, width);
                words += WordBuilder.Build(decoded, crop, characters, width, resized, 5).Words.Count;
            }
        });
        output.WriteLine($"CTC decoding and words, {inputs.Count} lines x {characters.Count} classes, {words} words: {ms:F2} ms, {bytes / 1024.0:F1} KiB allocated per page");
        Assert.True(ms < 1000, $"{ms} ms");
    }

    /// <summary>The median time and the mean allocations of a run, after one run to warm up.</summary>
    private static (double Milliseconds, long Bytes) Measure(Action run)
    {
        run();
        var times = new double[Runs];
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Runs; i++)
        {
            long start = Stopwatch.GetTimestamp();
            run();
            times[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }

        long bytes = (GC.GetAllocatedBytesForCurrentThread() - before) / Runs;
        Array.Sort(times);
        return (times[Runs / 2], bytes);
    }
}
