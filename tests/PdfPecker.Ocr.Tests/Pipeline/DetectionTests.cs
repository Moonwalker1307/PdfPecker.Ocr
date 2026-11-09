using System.Numerics;
using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>
/// The detection stage against rapidocr 3.10: the input recipe on the recorded images, and DB post-processing on the
/// recorded probability maps, compared with the boxes rapidocr's own post-processing found on the same maps.
/// </summary>
public sealed class DetectionTests
{
    public static TheoryData<string> Cases => [.. PipelineFixture.Cases];

    [Theory]
    [MemberData(nameof(Cases))]
    public void PostProcessingMatchesRapidocrBoxes(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        float[] map = fx.ProbabilityMap();
        var processor = new DbPostProcessor();
        List<DetectedQuad> boxes = processor.Process(map, fx.DetInputWidth, fx.DetInputHeight, fx.DetImageWidth, fx.DetImageHeight, DetectionOptions.Rapidocr);
        List<float[]> expected = fx.Boxes("det_sorted_boxes");

        Assert.Equal(expected.Count, boxes.Count);
        int exact = 0;
        for (int i = 0; i < expected.Count; i++)
        {
            Quad want = QuadMath.FromArray(expected[i]);
            double iou = QuadMath.IoU(want, boxes[i].Quad);
            Assert.True(iou >= 0.95, $"{name} box {i}: IoU {iou:F3}, expected {string.Join(',', expected[i])}, got {boxes[i].Quad}");
            exact += want == boxes[i].Quad ? 1 : 0;
        }

        // The geometry follows OpenCV's single-precision rounding, so the boxes are not just close but equal.
        Assert.Equal(expected.Count, exact);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PostProcessingScoresMatchRapidocr(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        var processor = new DbPostProcessor();
        List<DetectedQuad> boxes = processor.Process(fx.ProbabilityMap(), fx.DetInputWidth, fx.DetInputHeight, fx.DetImageWidth, fx.DetImageHeight, DetectionOptions.Rapidocr);

        // rapidocr's detector returns its scores in the order before sorting, so match boxes by position.
        List<float[]> unsorted = fx.Boxes("det_boxes");
        float[] scores = [.. fx.Root.GetProperty("det_scores").EnumerateArray().Select(e => e.GetSingle())];
        foreach (DetectedQuad box in boxes)
        {
            int best = -1;
            double bestIou = 0;
            for (int i = 0; i < unsorted.Count; i++)
            {
                double iou = QuadMath.IoU(QuadMath.FromArray(unsorted[i]), box.Quad);
                if (iou > bestIou)
                {
                    bestIou = iou;
                    best = i;
                }
            }

            Assert.True(best >= 0);
            Assert.True(Math.Abs(scores[best] - box.Score) < 0.01, $"{name}: score {box.Score} against rapidocr's {scores[best]}");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void InputSizeMatchesRapidocr(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        DetectionInput.Layout layout = DetectionInput.Plan(fx.DetImageWidth, fx.DetImageHeight, DetectionOptions.Rapidocr);
        Assert.Equal(0, layout.PaddingTop);
        Assert.Equal(fx.DetInputWidth, layout.InputWidth);
        Assert.Equal(fx.DetInputHeight, layout.InputHeight);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void InputPixelsMatchOpenCvResize(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        byte[] image = fx.DetImage();
        var view = new PixelView(image, fx.DetImageWidth, fx.DetImageHeight, fx.Channels);
        DetectionInput.Layout layout = DetectionInput.Plan(view.Width, view.Height, DetectionOptions.Rapidocr);
        int plane = layout.InputWidth * layout.InputHeight;
        var input = new float[3 * plane];
        DetectionInput.Fill(view, layout, input);

        byte[] expected = fx.DetInput();
        int planes = expected.Length / plane;
        Assert.Equal(fx.Channels == 1 ? 1 : 3, planes);
        long exact = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            // Undo the normalisation: v = px / 127.5 - 1.
            int px = (int)MathF.Round((input[i] + 1) * 127.5f);
            int diff = Math.Abs(px - expected[i]);
            Assert.True(diff <= 1, $"{name}: pixel {i} is {px}, OpenCV gives {expected[i]}");
            exact += diff == 0 ? 1 : 0;
        }

        Assert.True(exact >= expected.Length * 0.999, $"{name}: only {exact} of {expected.Length} pixels equal OpenCV's");

        // A grey image fills all three planes with the same values.
        if (planes == 1)
        {
            Assert.True(input.AsSpan(0, plane).SequenceEqual(input.AsSpan(plane, plane)));
            Assert.True(input.AsSpan(0, plane).SequenceEqual(input.AsSpan(2 * plane, plane)));
        }
    }

    [Fact]
    public void NormalisationMatchesRapidocr()
    {
        // numpy: (float32(px) * float32(1 / 255) - 0.5) / 0.5 with the subtraction in double, stored as float32.
        var options = DetectionOptions.Rapidocr with { MinSideLength = 1, VerticalPadding = false };
        foreach ((byte px, float expected) in new[] { ((byte)0, -1f), ((byte)255, 1f), ((byte)128, 0.003921628f), ((byte)64, -0.498039186f) })
        {
            var view = new PixelView(new[] { px }, 1, 1, 1);
            DetectionInput.Layout layout = DetectionInput.Plan(1, 1, options);
            Assert.Equal(32, layout.InputWidth);
            Assert.Equal(32, layout.InputHeight);
            var input = new float[3 * 32 * 32];
            DetectionInput.Fill(view, layout, input);
            Assert.All(input, v => Assert.Equal(expected, v));
        }
    }

    [Fact]
    public void FitScalesA4At300DpiToMaxSide()
    {
        DetectionInput.Layout layout = DetectionInput.Plan(2480, 3508, new DetectionOptions());
        Assert.Equal(1536, layout.InputHeight);
        Assert.Equal(1088, layout.InputWidth);
    }

    [Fact]
    public void FitScalesSmallImagesUpButNotPastMaxSide()
    {
        // 900 x 260: rapidocr scales the short side to 736 (2560 wide); Fit stops at 1536 wide.
        DetectionInput.Layout layout = DetectionInput.Plan(900, 260, new DetectionOptions());
        Assert.Equal(1536, layout.InputWidth);
        Assert.Equal(448, layout.InputHeight);

        // A 300 x 400 image scales up to 736 on the short side.
        layout = DetectionInput.Plan(300, 400, new DetectionOptions());
        Assert.Equal(736, layout.InputWidth);
        Assert.Equal(992, layout.InputHeight);
    }

    [Fact]
    public void FitGivesElongatedPagesTheAreaOfAPageOfThreeToOne()
    {
        // A4 at 300 dpi keeps the longer side of 1,536.
        DetectionInput.Layout layout = DetectionInput.Plan(2480, 3508, new DetectionOptions());
        Assert.Equal((1088, 1536), (layout.InputWidth, layout.InputHeight));

        // A receipt of 567 x 7,087 is scaled by 1536 / sqrt(3 * 567 * 7087), not 1536 / 7087, which shrank its text
        // fourfold.
        layout = DetectionInput.Plan(567, 7087, new DetectionOptions());
        Assert.Equal((256, 3136), (layout.InputWidth, layout.InputHeight));

        // Never past the page's own size: a strip of 100 x 3,000 is detected as it is.
        layout = DetectionInput.Plan(100, 3000, new DetectionOptions());
        Assert.Equal((96, 3008), (layout.InputWidth, layout.InputHeight));
    }

    [Fact]
    public void DenseMapsKeepTheirTopRows()
    {
        // 1,200 text-sized regions, more than rapidocr's 1,000 candidates: looked at from the bottom up and all counted,
        // the top rows were lost. The budget grows with the map, so every region is found.
        const int W = 1024, H = 512;
        var map = new float[W * H];
        for (int row = 0; row < 30; row++)
        {
            for (int col = 0; col < 40; col++)
            {
                Blob(map, W, 20 + (col * 24), 10 + (row * 16), 12, 6);
            }
        }

        var processor = new DbPostProcessor();
        List<DetectedQuad> boxes = processor.Process(map, W, H, W, H, new DetectionOptions());
        Assert.Equal(1200, boxes.Count);
        Assert.False(processor.LastCandidatesCapped);
        Assert.InRange(boxes.Min(b => b.Quad.P0.Y), 0, 10);
    }

    [Fact]
    public void SpecksDoNotUseUpTheCandidates()
    {
        // A noisy scan: 16,384 isolated specks below a row of 40 text regions. Specks are no longer counted against the
        // candidates, so the text is found.
        const int W = 1024, H = 512;
        var map = new float[W * H];
        for (int col = 0; col < 40; col++)
        {
            Blob(map, W, 20 + (col * 24), 10, 12, 6);
        }

        for (int y = 256; y < H; y += 4)
        {
            for (int x = 0; x < W; x += 4)
            {
                map[(y * W) + x] = 1;
            }
        }

        var processor = new DbPostProcessor();
        List<DetectedQuad> boxes = processor.Process(map, W, H, W, H, new DetectionOptions());
        Assert.Equal(40, boxes.Count);
        Assert.True(processor.LastRegionCount > 16_000);
    }

    private static void Blob(float[] map, int width, int x0, int y0, int w, int h)
    {
        for (int y = y0; y < y0 + h; y++)
        {
            map.AsSpan((y * width) + x0, w).Fill(1);
        }
    }

    [Fact]
    public void WideStripsArePaddedAsRapidocrDoes()
    {
        // rapidocr: h = 40 <= 30 is false, but 1000 / 40 > 8, so the padded height is max(125, 30) * 2 = 250.
        DetectionInput.Layout layout = DetectionInput.Plan(1000, 40, DetectionOptions.Rapidocr);
        Assert.Equal(105, layout.PaddingTop);
        Assert.Equal(250, layout.PaddedHeight);
    }

    [Fact]
    public void PaddingIsRemovedFromBoxes()
    {
        // A single line, 1000 x 40, through a stand-in network that marks the text rows of the padded input.
        var image = new byte[1000 * 40];
        Array.Fill(image, (byte)255);
        var network = new FunctionNetwork((input, shape) =>
        {
            int h = shape[2], w = shape[3];
            var map = new float[h * w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    // Text from x 100 to 900 and from y 10 to 30 of the image, i.e. rows 115 to 135 of the padded one.
                    double py = (y + 0.5) * 250.0 / h, px = (x + 0.5) * 1000.0 / w;
                    map[(y * w) + x] = px > 100 && px < 900 && py > 117 && py < 133 ? 0.9f : 0f;
                }
            }

            return (map, [1, 1, h, w]);
        });

        var detector = new TextDetector(network, DetectionOptions.Rapidocr);
        List<DetectedQuad> boxes = detector.Detect(new PixelView(image, 1000, 40, 1));
        DetectedQuad box = Assert.Single(boxes);
        (float left, float top, float right, float bottom) = box.Quad.Bounds();
        Assert.InRange(left, 85, 105);
        Assert.InRange(right, 895, 915);
        Assert.InRange(top, 0, 12);
        Assert.InRange(bottom, 28, 40);
    }

    [Fact]
    public void OrderPointsClockwiseMatchesRapidocr()
    {
        Span<Vector2> p = [new(10, 20), new(0, 0), new(12, 2), new(-2, 18)];
        DbPostProcessor.OrderPointsClockwise(p);
        Assert.Equal(new Vector2(0, 0), p[0]);
        Assert.Equal(new Vector2(12, 2), p[1]);
        Assert.Equal(new Vector2(10, 20), p[2]);
        Assert.Equal(new Vector2(-2, 18), p[3]);
    }
}
