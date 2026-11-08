using System.Buffers;
using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// Decides which line crops are upside down with the 0/180 degree classifier (PaddleOCR's
/// <c>ch_ppocr_mobile_v2.0_cls</c>, input 48 x 192), and turns them.
/// </summary>
/// <remarks>
/// <para>
/// rapidocr's <c>TextClassifier</c> squeezes a whole line into 48 x 192 pixels, so a long line arrives squashed to a
/// tenth of its width and the classifier guesses: on rendered A4 pages, rapidocr turned 2 to 5 of 50 to 58 upright
/// lines upside down and read them as garbage, which was most of its error. Here a line longer than 4:1 is classified
/// on three windows of 4:1 at its natural aspect (start, middle, end), and the three probabilities are averaged; on the
/// same pages no line was turned wrongly. A line of 4:1 or less is one window, exactly as rapidocr feeds it.
/// </para>
/// <para>
/// Short crops such as "39%" are still unreliable (rapidocr turns that table cell upside down at 0.949), so the page
/// votes: when at least 80% of the text, weighted by length, reads upright, a line is turned only above 0.99; when at
/// least 80% reads upside down, a line is turned unless it is upright above 0.99. Otherwise rapidocr's threshold
/// applies.
/// </para>
/// </remarks>
internal sealed class OrientationClassifier
{
    public const int InputHeight = 48;
    public const int InputWidth = 192;

    private readonly INetwork _network;

    public OrientationClassifier(INetwork network, RecognitionOptions options)
    {
        _network = network;
        Options = options;
    }

    public RecognitionOptions Options { get; }

    /// <summary>The averaged probability that each crop is upside down, from the last call, for tests.</summary>
    public float[] LastUpsideDown { get; private set; } = [];

    /// <summary>Classifies the crops and sets <see cref="LineCrop.Rotated180"/> on those that are upside down.</summary>
    public void Apply(PixelView page, IReadOnlyList<LineCrop> crops, CancellationToken cancellationToken = default)
    {
        float[] p180 = Classify(page, crops, cancellationToken);
        LastUpsideDown = p180;
        bool[] turn = Decide(crops, p180, Options.OrientationThreshold, Options.PageVote);
        for (int i = 0; i < crops.Count; i++)
        {
            crops[i].Rotated180 = turn[i];
        }
    }

    /// <summary>Returns, for each crop, the mean probability of its windows being upside down.</summary>
    public float[] Classify(PixelView page, IReadOnlyList<LineCrop> crops, CancellationToken cancellationToken = default)
    {
        // Windows as (crop, x offset, width) in strip pixels.
        var windows = new List<(int Crop, double X, double Width)>();
        var counts = new int[crops.Count];
        for (int i = 0; i < crops.Count; i++)
        {
            LineCrop crop = crops[i];
            double window = 4.0 * crop.CropHeight;
            if (!Options.OrientationWindows || crop.CropWidth <= window)
            {
                windows.Add((i, 0, crop.CropWidth));
                counts[i] = 1;
                continue;
            }

            windows.Add((i, 0, window));
            windows.Add((i, Math.Floor((crop.CropWidth - window) / 2), window));
            windows.Add((i, crop.CropWidth - window, window));
            counts[i] = 3;
        }

        var sums = new double[crops.Count];
        int batch = Math.Max(1, Options.OrientationBatchSize);
        int plane = InputHeight * InputWidth;
        float[] input = ArrayPool<float>.Shared.Rent(batch * 3 * plane);
        try
        {
            for (int start = 0; start < windows.Count; start += batch)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int n = Math.Min(batch, windows.Count - start);
                Span<float> tensor = input.AsSpan(0, n * 3 * plane);
                tensor.Clear();
                for (int k = 0; k < n; k++)
                {
                    (int c, double x, double width) = windows[start + k];
                    LineCrop crop = crops[c];

                    // rapidocr's resize_norm_img: height 48, width ceil(48 * ratio) up to 192, zero padding after.
                    double ratio = width / crop.CropHeight;
                    int resized = Math.Clamp((int)Math.Ceiling(InputHeight * ratio), 1, InputWidth);
                    crop.Sample(page, tensor.Slice(k * 3 * plane), plane, InputWidth, resized, InputHeight, x, width);
                }

                ReadOnlySpan<float> output = _network.Run(tensor, [n, 3, InputHeight, InputWidth], out int[] shape);
                if (shape.Length != 2 || shape[0] != n || shape[1] < 2)
                {
                    throw new InvalidOperationException("The orientation network returned an unexpected shape.");
                }

                for (int k = 0; k < n; k++)
                {
                    sums[windows[start + k].Crop] += output[(k * shape[1]) + 1];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(input);
        }

        var result = new float[crops.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = (float)(sums[i] / counts[i]);
        }

        return result;
    }

    /// <summary>Turns the probabilities into decisions, with the page vote described on the class.</summary>
    internal static bool[] Decide(IReadOnlyList<LineCrop> crops, float[] upsideDown, float threshold, bool pageVote)
    {
        double upright = 0, flipped = 0;
        for (int i = 0; i < crops.Count; i++)
        {
            // Long lines count more, up to the length of three windows.
            double weight = Math.Min(crops[i].AspectRatio, 12);
            if (upsideDown[i] > 0.5f)
            {
                flipped += weight;
            }
            else
            {
                upright += weight;
            }
        }

        double total = upright + flipped;
        bool mostlyUpright = pageVote && total > 0 && upright >= 0.8 * total;
        bool mostlyFlipped = pageVote && total > 0 && flipped >= 0.8 * total;
        var turn = new bool[crops.Count];
        for (int i = 0; i < turn.Length; i++)
        {
            float p = upsideDown[i];
            turn[i] = mostlyUpright ? p > 0.99f : mostlyFlipped ? 1 - p <= 0.99f : p > threshold;
        }

        return turn;
    }
}
