using System.Text.Json;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Recognition;
using Xunit.Abstractions;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>
/// Crops, CTC decoding and words against rapidocr 3.10's recorded crops, recognition outputs, texts and word boxes.
/// </summary>
public sealed class RecognitionTests(ITestOutputHelper output)
{
    private static readonly CharacterSet Characters = new(PipelineFixture.Characters[1..^1]);

    public static TheoryData<string> Cases => [.. PipelineFixture.Cases];

    public static TheoryData<string> CropCases => ["rotated", "upside_down", "table", "nordic", "gap"];

    [Fact]
    public void CharacterSetAddsBlankAndSpaceAsRapidocr()
    {
        Assert.Equal(18710, Characters.Count);
        Assert.Equal(string.Empty, Characters[0]);
        Assert.Equal(" ", Characters[Characters.Space]);
        Assert.Equal(PipelineFixture.Characters[1], Characters[1]);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CtcDecodingMatchesRapidocrTexts(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        JsonElement texts = fx.Root.GetProperty("rec_texts");
        foreach (RecordedLine line in RecordedLine.Read(fx))
        {
            DecodedLine decoded = CtcDecoder.Decode(line.Probabilities(Characters.Count), line.Steps, Characters.Count, Characters);
            JsonElement expected = texts[line.Line];
            Assert.Equal(expected.GetProperty("text").GetString(), decoded.Text);
            Assert.Equal(expected.GetProperty("score").GetSingle(), decoded.Score, 0.00002f);
        }
    }

    [Theory]
    [InlineData("invoice")]
    [InlineData("nordic")]
    [InlineData("columns")]
    [InlineData("gap")]
    public void LatinSubsetReadsLatinTextTheSame(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        int[] latin = Characters.Subset(CharacterSet.IsLatin);
        Assert.InRange(latin.Length, 200, 1200);
        foreach (RecordedLine line in RecordedLine.Read(fx))
        {
            float[] p = line.Probabilities(Characters.Count);
            DecodedLine all = CtcDecoder.Decode(p, line.Steps, Characters.Count, Characters);
            DecodedLine subset = CtcDecoder.Decode(p, line.Steps, Characters.Count, Characters, latin);
            Assert.Equal(all.Text, subset.Text);
        }
    }

    [Theory]
    [InlineData("invoice")]
    [InlineData("table")]
    public void BestClassOutputDecodesTheSame(string name)
    {
        // The engine's recogniser returns the best class and its probability per time step instead of every class.
        using PipelineFixture fx = PipelineFixture.Load(name);
        foreach (RecordedLine line in RecordedLine.Read(fx))
        {
            float[] p = line.Probabilities(Characters.Count);
            var best = new float[line.Steps * 2];
            for (int t = 0; t < line.Steps; t++)
            {
                (int index, float value) = CtcDecoder.ArgMax(p.AsSpan(t * Characters.Count, Characters.Count));
                best[2 * t] = index;
                best[(2 * t) + 1] = value;
            }

            DecodedLine all = CtcDecoder.Decode(p, line.Steps, Characters.Count, Characters);
            DecodedLine fromBest = CtcDecoder.Decode(best, line.Steps, 2, Characters);
            Assert.Equal(all.Text, fromBest.Text);
            Assert.Equal(all.Score, fromBest.Score);
            Assert.Equal(all.Columns, fromBest.Columns);
            Assert.Equal(all.Probabilities, fromBest.Probabilities);
        }

        int[] latin = Characters.Subset(CharacterSet.IsLatin);
        Assert.Throws<InvalidOperationException>(() => CtcDecoder.Decode(new float[4], 2, 2, Characters, latin));
        Assert.Throws<InvalidOperationException>(() => CtcDecoder.Decode([Characters.Count, 1f], 1, 2, Characters));
        Assert.Throws<InvalidOperationException>(() => CtcDecoder.Decode([float.NaN, 1f], 1, 2, Characters));
    }

    [Fact]
    public void SubsetKeepsOnlyAllowedClasses()
    {
        // The table's "39%" was turned upside down by rapidocr and read as "%6ε"; with Latin only, no ε.
        using PipelineFixture fx = PipelineFixture.Load("table");
        int[] latin = Characters.Subset(CharacterSet.IsLatin);
        RecordedLine line = RecordedLine.Read(fx).Single(l => fx.Root.GetProperty("rec_texts")[l.Line].GetProperty("text").GetString() == "%6ε");
        DecodedLine decoded = CtcDecoder.Decode(line.Probabilities(Characters.Count), line.Steps, Characters.Count, Characters, latin);
        Assert.DoesNotContain('ε', decoded.Text);
        Assert.All(decoded.Text, c => Assert.True(CharacterSet.IsLatin(c.ToString())));
    }

    [Fact]
    public void DecodingCollapsesRepeatsAndDropsBlanks()
    {
        // Classes over 6 steps: a, a, blank, a, space, b -> "aa b".
        int a = Array.IndexOf(PipelineFixture.Characters, "a"), b = Array.IndexOf(PipelineFixture.Characters, "b");
        int[] steps = [a, a, 0, a, Characters.Space, b];
        var p = new float[steps.Length * Characters.Count];
        for (int t = 0; t < steps.Length; t++)
        {
            p[(t * Characters.Count) + steps[t]] = 0.9f;
        }

        DecodedLine line = CtcDecoder.Decode(p, steps.Length, Characters.Count, Characters);
        Assert.Equal("aa b", line.Text);
        Assert.Equal([0, 3, 4, 5], line.Columns);
        Assert.Equal(0.9f, line.Score, 0.00001f);

        DecodedLine empty = CtcDecoder.Decode(new float[3 * Characters.Count], 3, Characters.Count, Characters);
        Assert.Equal(string.Empty, empty.Text);
        Assert.Equal(0f, empty.Score);
    }

    [Fact]
    public void ArgMaxTakesTheFirstOfEqualValues()
    {
        var row = new float[100];
        row[17] = 0.5f;
        row[60] = 0.5f;
        Assert.Equal((17, 0.5f), CtcDecoder.ArgMax(row));
        Assert.Equal((60, 0.5f), CtcDecoder.ArgMax(row, [0, 60, 99]));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CropSizesMatchRapidocr(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        List<float[]> boxes = fx.Boxes("det_sorted_boxes");
        foreach (RecordedLine line in RecordedLine.Read(fx))
        {
            var crop = new LineCrop(QuadMath.FromArray(boxes[line.Line]), line.Line);
            Assert.Equal(line.CropWidth, crop.CropWidth);
            Assert.Equal(line.CropHeight, crop.CropHeight);
        }
    }

    [Theory]
    [MemberData(nameof(CropCases))]
    public void CropPixelsAreCloseToRapidocrs(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        List<float[]> boxes = fx.Boxes("det_sorted_boxes");
        byte[] image = fx.Image();
        byte[] recorded = fx.RecPixels();
        var page = new PixelView(image, fx.Width, fx.Height, fx.Channels);
        JsonElement cls = fx.Root.GetProperty("cls_outputs");
        int planes = fx.Channels == 1 ? 1 : 3;
        double worst = 0;
        foreach (RecordedLine line in RecordedLine.Read(fx))
        {
            var crop = new LineCrop(QuadMath.FromArray(boxes[line.Line]), line.Line)
            {
                Rotated180 = cls[line.Line][1].GetSingle() > 0.9f,
            };
            int width = line.InputShape[3];
            int resized = Math.Clamp((int)Math.Ceiling(48 * crop.AspectRatio), 1, width);
            int plane = 48 * width;
            var tensor = new float[3 * plane];
            crop.Sample(page, tensor, plane, width, resized, 48, 0, crop.CropWidth);

            // Compare the filled part as grey levels; rapidocr warps bicubically and then resizes, this samples once.
            double sum = 0;
            int count = 0;
            for (int p = 0; p < planes; p++)
            {
                for (int y = 0; y < 48; y++)
                {
                    for (int x = 0; x < resized; x++)
                    {
                        int mine = (int)MathF.Round(((tensor[(p * plane) + (y * width) + x] * 0.5f) + 0.5f) * 255f);
                        int theirs = recorded[line.PixelOffset + (p * plane) + (y * width) + x];
                        sum += Math.Abs(mine - theirs);
                        count++;
                    }
                }
            }

            double mad = sum / count;
            worst = Math.Max(worst, mad);
            Assert.True(mad < 6, $"{name} line {line.Line}: mean difference {mad:F2} grey levels");
        }

        output.WriteLine($"{name}: worst mean absolute difference {worst:F2} grey levels");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void WordsMatchRapidocrWords(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        List<float[]> boxes = fx.Boxes("det_sorted_boxes");
        JsonElement cls = fx.Root.GetProperty("cls_outputs");
        var finals = fx.Root.GetProperty("final").EnumerateArray().ToList();
        double iouSum = 0;
        int iouCount = 0;
        foreach (RecordedLine line in RecordedLine.Read(fx))
        {
            var crop = new LineCrop(QuadMath.FromArray(boxes[line.Line]), line.Line) { Rotated180 = cls[line.Line][1].GetSingle() > 0.9f };
            DecodedLine decoded = CtcDecoder.Decode(line.Probabilities(Characters.Count), line.Steps, Characters.Count, Characters);
            int width = line.InputShape[3];
            int resized = Math.Clamp((int)Math.Ceiling(48 * crop.AspectRatio), 1, width);
            (string text, List<RecognizedWord> words) = WordBuilder.Build(decoded, crop, Characters, width, resized, 5);

            JsonElement? final = finals.FirstOrDefault(f => f.GetProperty("text").GetString() == decoded.Text && QuadMath.IoU(QuadMath.FromArray(Flatten(f.GetProperty("box"))), crop.Quad) > 0.9);
            if (final is null)
            {
                continue;
            }

            string[] expectedWords = [.. final.Value.GetProperty("words").EnumerateArray().Select(w => w.GetProperty("text").GetString()!)];
            Assert.Equal(expectedWords, words.Select(w => w.Text).ToArray());

            // Text keeps rapidocr's, plus a space wherever a word ended at a gap without one.
            Assert.Equal(string.Join(' ', expectedWords), string.Join(' ', text.Split(' ', StringSplitOptions.RemoveEmptyEntries)));

            // On a line turned upside down, the first word sits at the right end of the line on the page. rapidocr
            // maps word boxes back without undoing the classifier's turn, which mirrors them, so they are not compared.
            if (crop.Rotated180 && words.Count > 1)
            {
                Assert.True(words[0].Quad.Center.X > words[^1].Quad.Center.X);
                continue;
            }

            // rapidocr's word boxes are axis-aligned; compare them with ours where the line is horizontal.
            if (!crop.Rotated90 && Math.Abs(crop.Quad.P1.Y - crop.Quad.P0.Y) < 2)
            {
                int i = 0;
                foreach (JsonElement w in final.Value.GetProperty("words").EnumerateArray())
                {
                    double iou = QuadMath.IoU(QuadMath.FromArray(Flatten(w.GetProperty("box"))), words[i++].Quad);
                    iouSum += iou;
                    iouCount++;
                    Assert.True(iou > 0.5, $"{name}: word '{words[i - 1].Text}' IoU {iou:F2} with rapidocr's box");
                }
            }
        }

        if (iouCount > 0)
        {
            output.WriteLine($"{name}: mean word box IoU with rapidocr {iouSum / iouCount:F3} over {iouCount} words");
        }
    }

    [Fact]
    public void GapWithoutSpaceGetsOne()
    {
        // rapidocr reads the first line of the gap case as "EUR 1,234.56VAT 25%".
        using PipelineFixture fx = PipelineFixture.Load("gap");
        List<float[]> boxes = fx.Boxes("det_sorted_boxes");
        RecordedLine line = RecordedLine.Read(fx).Single(l => fx.Root.GetProperty("rec_texts")[l.Line].GetProperty("text").GetString() == "EUR 1,234.56VAT 25%");
        var crop = new LineCrop(QuadMath.FromArray(boxes[line.Line]), line.Line);
        DecodedLine decoded = CtcDecoder.Decode(line.Probabilities(Characters.Count), line.Steps, Characters.Count, Characters);
        int width = line.InputShape[3];
        int resized = Math.Clamp((int)Math.Ceiling(48 * crop.AspectRatio), 1, width);

        (string text, List<RecognizedWord> words) = WordBuilder.Build(decoded, crop, Characters, width, resized, 5);
        Assert.Equal("EUR 1,234.56 VAT 25%", text);
        Assert.Equal(["EUR", "1,234.56", "VAT", "25%"], words.Select(w => w.Text).ToArray());

        // Words run left to right without overlapping, inside the line.
        for (int i = 1; i < words.Count; i++)
        {
            Assert.True(words[i].Quad.P0.X >= words[i - 1].Quad.P1.X - 0.01f);
        }

        Assert.True(words[0].Quad.P0.X >= crop.Quad.P0.X - 0.01f);
        Assert.True(words[^1].Quad.P1.X <= crop.Quad.P1.X + 0.01f);

        // rapidocr's settings keep its text.
        (string kept, _) = WordBuilder.Build(decoded, crop, Characters, width, resized, int.MaxValue);
        Assert.Equal("EUR 1,234.56VAT 25%", kept);
    }

    [Fact]
    public void UnspacedScriptsGetOneWordPerCharacter()
    {
        int han = Array.IndexOf(PipelineFixture.Characters, "中"), wen = Array.IndexOf(PipelineFixture.Characters, "文");
        int a = Array.IndexOf(PipelineFixture.Characters, "A");
        int[] steps = [han, 0, 0, wen, 0, a, 0, a];
        var p = new float[steps.Length * Characters.Count];
        for (int t = 0; t < steps.Length; t++)
        {
            p[(t * Characters.Count) + steps[t]] = 1;
        }

        DecodedLine decoded = CtcDecoder.Decode(p, steps.Length, Characters.Count, Characters);
        var crop = new LineCrop(new Quad(new(0, 0), new(64, 0), new(64, 48), new(0, 48)), 0);
        (string text, List<RecognizedWord> words) = WordBuilder.Build(decoded, crop, Characters, 64, 64, 5);
        Assert.Equal("中文AA", text);
        Assert.Equal(["中", "文", "AA"], words.Select(w => w.Text).ToArray());
    }

    private static float[] Flatten(JsonElement box)
    {
        var b = new float[8];
        int i = 0;
        foreach (JsonElement p in box.EnumerateArray())
        {
            b[i++] = p[0].GetSingle();
            b[i++] = p[1].GetSingle();
        }

        return b;
    }
}
