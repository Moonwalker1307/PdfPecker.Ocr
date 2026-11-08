namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>
/// Stand-ins for the networks that replay what ONNX Runtime returned to rapidocr: the detection network returns the
/// recorded probability map, and the recognition network finds, for each line of a batch, the recorded line whose
/// input pixels are closest and returns that line's output. The pipeline's plumbing (sizes, batching, sorting,
/// turning, mapping back) can then be checked end to end without the real networks.
/// </summary>
internal static class ReplayNetworks
{
    public static FunctionNetwork Detection(PipelineFixture fx)
    {
        float[] map = fx.ProbabilityMap();
        return new FunctionNetwork((input, shape) =>
        {
            Assert.Equal([1, 3, fx.DetInputHeight, fx.DetInputWidth], shape);
            return (map, [1, 1, fx.DetInputHeight, fx.DetInputWidth]);
        });
    }

    public static FunctionNetwork Recognition(PipelineFixture fx, int classCount)
    {
        List<RecordedLine> lines = RecordedLine.Read(fx);
        byte[] pixels = fx.RecPixels();
        return new FunctionNetwork((input, shape) =>
        {
            int n = shape[0], width = shape[3], plane = 48 * width;

            // The recognition model downsamples the width by 8: 513 pixels give 64 time steps.
            int steps = width / 8;
            var output = new float[n * steps * classCount];
            for (int k = 0; k < n; k++)
            {
                RecordedLine best = lines[0];
                double bestDiff = double.MaxValue;
                foreach (RecordedLine line in lines)
                {
                    int theirWidth = line.InputShape[3];
                    int w = Math.Min(width, theirWidth);
                    double diff = 0;
                    for (int y = 0; y < 48; y += 2)
                    {
                        for (int x = 0; x < w; x += 2)
                        {
                            float mine = input[(k * 3 * plane) + (y * width) + x];
                            int theirs = pixels[line.PixelOffset + (y * theirWidth) + x];
                            diff += Math.Abs((((mine * 0.5f) + 0.5f) * 255f) - theirs);
                        }
                    }

                    diff /= w;
                    if (diff < bestDiff)
                    {
                        bestDiff = diff;
                        best = line;
                    }
                }

                best.Probabilities(classCount, steps).CopyTo(output, k * steps * classCount);
            }

            return (output, [n, steps, classCount]);
        });
    }

    /// <summary>An orientation network that calls every window upside down, or none.</summary>
    public static FunctionNetwork Orientation(bool upsideDown) => new((input, shape) =>
    {
        var output = new float[shape[0] * 2];
        for (int k = 0; k < shape[0]; k++)
        {
            output[(2 * k) + (upsideDown ? 1 : 0)] = 1;
        }

        return (output, [shape[0], 2]);
    });
}
