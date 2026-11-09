using System.Text.Json;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>Turning upside-down lines: rapidocr's decisions on its recorded classifier outputs, windows and the page vote.</summary>
public sealed class OrientationTests
{
    [Theory]
    [InlineData("table")]
    [InlineData("rotated")]
    [InlineData("upside_down")]
    [InlineData("invoice")]
    public void RapidocrSettingsTurnTheLinesRapidocrTurned(string name)
    {
        using PipelineFixture fx = PipelineFixture.Load(name);
        (LineCrop[] crops, float[] p180) = Recorded(fx);
        bool[] turn = OrientationClassifier.Decide(crops, p180, 0.9f, pageVote: false);
        for (int i = 0; i < crops.Length; i++)
        {
            Assert.Equal(p180[i] > 0.9f, turn[i]);
        }
    }

    [Fact]
    public void PageVoteKeepsAShortCellUpright()
    {
        // rapidocr turns the table's "39%" (0.949) and reads "%6ε"; the rest of the table reads upright.
        using PipelineFixture fx = PipelineFixture.Load("table");
        (LineCrop[] crops, float[] p180) = Recorded(fx);
        Assert.Contains(p180, p => p > 0.9f);
        Assert.All(OrientationClassifier.Decide(crops, p180, 0.9f, pageVote: true), Assert.False);
    }

    [Fact]
    public void PageVoteStillTurnsAConfidentLine()
    {
        // The rotated page's vertical label is upside down after the quarter turn, at 1.0.
        using PipelineFixture fx = PipelineFixture.Load("rotated");
        (LineCrop[] crops, float[] p180) = Recorded(fx);
        bool[] turn = OrientationClassifier.Decide(crops, p180, 0.9f, pageVote: true);
        Assert.Equal(1, turn.Count(t => t));
        Assert.True(crops[Array.IndexOf(turn, true)].Rotated90);

        using PipelineFixture upside = PipelineFixture.Load("upside_down");
        (crops, p180) = Recorded(upside);
        Assert.All(OrientationClassifier.Decide(crops, p180, 0.9f, pageVote: true), Assert.True);
    }

    [Fact]
    public void LongLinesAreClassifiedOnThreeWindows()
    {
        // One line of 20:1 and one of 3:1: four windows, at the classifier's 48 x 192 input.
        var page = new PixelView(new byte[1200 * 200], 1200, 200, 1);
        LineCrop[] crops =
        [
            new(LayoutTests.Box(0, 0, 1000, 50), 0),
            new(LayoutTests.Box(0, 100, 150, 150), 1),
        ];
        int call = 0;
        float[] answers = [0.6f, 0.9f, 0.9f, 0.2f];
        var network = new FunctionNetwork((input, shape) =>
        {
            Assert.Equal([4, 3, 48, 192], shape);
            var output = new float[shape[0] * 2];
            for (int k = 0; k < shape[0]; k++)
            {
                output[(2 * k) + 1] = answers[call++];
                output[2 * k] = 1 - output[(2 * k) + 1];
            }

            return (output, [shape[0], 2]);
        });

        var classifier = new OrientationClassifier(network, new RecognitionOptions { PageVote = false });
        classifier.Apply(page, crops);
        Assert.Equal(0.8f, classifier.LastUpsideDown[0], 0.0001f);
        Assert.Equal(0.2f, classifier.LastUpsideDown[1], 0.0001f);
        Assert.False(crops[0].Rotated180);
        Assert.False(crops[1].Rotated180);
        Assert.Equal(1, network.Calls);
    }

    [Fact]
    public void RapidocrSqueezesLongLinesIntoOneWindow()
    {
        var page = new PixelView(new byte[1200 * 100], 1200, 100, 1);
        LineCrop[] crops = [new(LayoutTests.Box(0, 0, 1000, 50), 0)];
        var network = new FunctionNetwork((input, shape) =>
        {
            Assert.Equal([1, 3, 48, 192], shape);
            return (new float[] { 0.05f, 0.95f }, [1, 2]);
        });

        new OrientationClassifier(network, RecognitionOptions.Rapidocr).Apply(page, crops);
        Assert.True(crops[0].Rotated180);
    }

    private static (LineCrop[] Crops, float[] UpsideDown) Recorded(PipelineFixture fx)
    {
        List<float[]> boxes = fx.Boxes("det_sorted_boxes");
        JsonElement cls = fx.Root.GetProperty("cls_outputs");
        var crops = new LineCrop[boxes.Count];
        var p180 = new float[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            crops[i] = new LineCrop(QuadMath.FromArray(boxes[i]), i);
            p180[i] = cls[i][1].GetSingle();
        }

        return (crops, p180);
    }
}
