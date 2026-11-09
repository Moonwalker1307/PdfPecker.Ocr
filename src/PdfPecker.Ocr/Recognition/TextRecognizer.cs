using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;

namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// Reads the text of detected lines: crops each quad from the page, turns upside-down lines, batches lines of
/// similar aspect ratio for the recognition network, decodes its output and splits lines into words. The batching
/// and the input recipe follow rapidocr's <c>TextRecognizer</c> (Apache-2.0, https://github.com/RapidAI/RapidOCR):
/// lines sorted by width over height, height 48, width <c>ceil(48 * ratio)</c>, each batch padded with zeros to
/// <c>48 * max(320 / 48, widest ratio)</c>. One instance per worker thread.
/// </summary>
internal sealed class TextRecognizer
{
    public const int InputHeight = 48;

    // rapidocr's rec_img_shape is 3 x 48 x 320: no batch is narrower than 320 / 48.
    private const double MinBatchRatio = 320.0 / 48.0;

    private readonly INetwork _recognition;
    private readonly OrientationClassifier? _orientation;
    private readonly CropBatch _batch;

    /// <summary>
    /// A recogniser whose networks run in <paramref name="context"/>, if given, whose threads then also sample the
    /// line crops.
    /// </summary>
    public TextRecognizer(INetwork recognition, INetwork? orientation, CharacterSet characters, RecognitionOptions? options = null, InferenceContext? context = null)
    {
        _recognition = recognition;
        Characters = characters;
        Options = options ?? new RecognitionOptions();
        _orientation = orientation is null ? null : new OrientationClassifier(orientation, Options, context);
        _batch = new CropBatch(context);
    }

    public CharacterSet Characters { get; }

    public RecognitionOptions Options { get; }

    /// <summary>The decoded lines of the last call in detection order, before any filtering, for tests.</summary>
    public DecodedLine?[] LastDecoded { get; private set; } = [];

    /// <summary>The upside-down probabilities of the last call, or empty when orientation was not checked.</summary>
    public float[] LastUpsideDown => _orientation?.LastUpsideDown ?? [];

    /// <summary>
    /// Recognises the lines in <paramref name="quads"/> and returns those with text and enough confidence, in the
    /// order of <paramref name="quads"/>.
    /// </summary>
    public List<RecognizedLine> Recognize(PixelView page, IReadOnlyList<DetectedQuad> quads, CancellationToken cancellationToken = default)
    {
        var crops = new LineCrop[quads.Count];
        for (int i = 0; i < crops.Length; i++)
        {
            crops[i] = new LineCrop(quads[i].Quad, i);
        }

        KeepShortTallCropsUpright(crops);

        if (Options.DetectRotation && _orientation is not null && crops.Length > 0)
        {
            _orientation.Apply(page, crops, cancellationToken);
        }

        // A stable sort by aspect ratio, so equal ratios keep the detector's order.
        int[] order = new int[crops.Length];
        for (int i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }

        Array.Sort(order, (a, b) =>
        {
            int c = crops[a].AspectRatio.CompareTo(crops[b].AspectRatio);
            return c != 0 ? c : a.CompareTo(b);
        });

        var decoded = new DecodedLine?[crops.Length];
        var results = new RecognizedLine?[crops.Length];
        int batch = Math.Max(1, Options.BatchSize);
        for (int start = 0; start < order.Length; start += batch)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int n = Math.Min(batch, order.Length - start);
            RecognizeBatch(page, crops, order.AsSpan(start, n), decoded, results);
        }

        LastDecoded = decoded;
        var list = new List<RecognizedLine>(crops.Length);
        foreach (RecognizedLine? line in results)
        {
            if (line is not null)
            {
                list.Add(line);
            }
        }

        return list;
    }

    /// <summary>
    /// How many times the median height of the horizontal lines a tall crop may be and still be read upright on a page
    /// of horizontal text.
    /// </summary>
    private const double UprightHeightRatio = 2;

    /// <summary>
    /// A quad at least 1.5 times as tall as wide is read turned a quarter, as rapidocr does, which is right for a line on
    /// a page turned sideways but not for a lone narrow character on an upright page: I, l or ! in a table cell was read
    /// sideways, as the CJK character for one, or lost, and the orientation classifier only tells 0 from 180 degrees.
    /// On a page whose text is mostly horizontal, a tall crop less than twice as high as the median horizontal line is
    /// such a character, and is read upright; longer ones are text running up or down the page and are still turned.
    /// </summary>
    internal static void KeepShortTallCropsUpright(LineCrop[] crops)
    {
        double horizontal = 0, vertical = 0;
        var heights = new List<int>();
        foreach (LineCrop crop in crops)
        {
            if (crop.Rotated90)
            {
                vertical += crop.WarpHeight;
            }
            else
            {
                horizontal += crop.WarpWidth;
                heights.Add(crop.WarpHeight);
            }
        }

        if (vertical == 0 || horizontal <= vertical)
        {
            return;
        }

        heights.Sort();
        double limit = UprightHeightRatio * heights[heights.Count / 2];
        for (int i = 0; i < crops.Length; i++)
        {
            if (crops[i].Rotated90 && crops[i].WarpHeight < limit)
            {
                crops[i] = new LineCrop(crops[i].Quad, crops[i].Index, turnTall: false);
            }
        }
    }

    private void RecognizeBatch(PixelView page, LineCrop[] crops, ReadOnlySpan<int> members, DecodedLine?[] decoded, RecognizedLine?[] results)
    {
        double maxRatio = MinBatchRatio;
        foreach (int i in members)
        {
            maxRatio = Math.Max(maxRatio, crops[i].AspectRatio);
        }

        int width = Math.Min((int)(InputHeight * maxRatio), Math.Max(Options.MaxInputWidth, InputHeight));
        int n = members.Length;
        int plane = InputHeight * width;
        Span<int> resized = n <= 64 ? stackalloc int[n] : new int[n];
        ReadOnlySpan<int> inputShape = [n, 3, InputHeight, width];
        Span<float> tensor = _recognition.InputBuffer(inputShape);
        tensor.Clear();
        _batch.Clear();
        for (int k = 0; k < n; k++)
        {
            LineCrop crop = crops[members[k]];
            double ratio = crop.AspectRatio;
            resized[k] = Math.Clamp((int)Math.Ceiling(InputHeight * ratio), 1, width);
            _batch.Add(crop, resized[k], 0, crop.CropWidth);
        }

        _batch.Sample(page, tensor, plane, width, InputHeight);

        ReadOnlySpan<float> output = _recognition.Run(tensor, inputShape, out int[] shape);
        if (shape.Length != 3 || shape[0] != n || shape[1] <= 0 || (long)shape[0] * shape[1] * shape[2] > output.Length)
        {
            throw new InvalidOperationException("The recognition network returned an unexpected shape.");
        }

        int steps = shape[1], classes = shape[2];
        for (int k = 0; k < n; k++)
        {
            int index = members[k];
            LineCrop crop = crops[index];
            DecodedLine line = CtcDecoder.Decode(output.Slice(k * steps * classes, steps * classes), steps, classes, Characters, Options.CharacterSubset);
            decoded[index] = line;
            if (line.Score < Options.MinConfidence)
            {
                continue;
            }

            (string text, List<RecognizedWord> words) = WordBuilder.Build(line, crop, Characters, width, resized[k], Options.WordGapSteps);
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            int h = crop.CropHeight, w = crop.CropWidth;
            var quad = new Quad(crop.EdgeToPage(0, 0), crop.EdgeToPage(w, 0), crop.EdgeToPage(w, h), crop.EdgeToPage(0, h));
            results[index] = new RecognizedLine(text.Trim(), line.Score, quad, words, index, crop.Rotated180);
        }
    }
}
