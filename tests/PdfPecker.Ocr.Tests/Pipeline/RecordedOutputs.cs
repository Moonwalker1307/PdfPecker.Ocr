using System.Text.Json;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>One line's recorded recognition input and output from a fixture.</summary>
internal sealed record RecordedLine(int Line, int CropWidth, int CropHeight, int[] InputShape, int PixelOffset, int[][] TopClasses, float[][] TopProbs)
{
    public int Steps => TopClasses.Length;

    /// <summary>
    /// The network output rebuilt from the top three classes of each time step, the other classes at 0. Greedy
    /// decoding only looks at the top class, so this decodes as the full output did.
    /// </summary>
    public float[] Probabilities(int classCount, int steps = -1)
    {
        if (steps < 0)
        {
            steps = Steps;
        }

        var p = new float[steps * classCount];
        for (int t = 0; t < steps; t++)
        {
            if (t >= Steps)
            {
                // Padding beyond the recorded steps reads as blank.
                p[t * classCount] = 1;
                continue;
            }

            for (int k = 0; k < TopClasses[t].Length; k++)
            {
                p[(t * classCount) + TopClasses[t][k]] = TopProbs[t][k];
            }
        }

        return p;
    }

    public static List<RecordedLine> Read(PipelineFixture fx)
    {
        var lines = new List<RecordedLine>();
        foreach (JsonElement e in fx.Root.GetProperty("rec_lines").EnumerateArray())
        {
            int[][] classes = [.. e.GetProperty("top_classes").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray())];
            float[][] probs = [.. e.GetProperty("top_probs").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray())];
            lines.Add(new RecordedLine(
                e.GetProperty("line").GetInt32(),
                e.GetProperty("crop_width").GetInt32(),
                e.GetProperty("crop_height").GetInt32(),
                [.. e.GetProperty("input_shape").EnumerateArray().Select(v => v.GetInt32())],
                e.GetProperty("pixel_offset").GetInt32(),
                classes,
                probs));
        }

        return lines;
    }
}
