using System.Text;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Layout;

/// <summary>
/// The OCR engine behind the public API: a page's pixels through the bundled models and the pipeline
/// (<see cref="OcrPipeline"/>) to the public <see cref="OcrPage"/>. Pages are read one after another, each with up to
/// <see cref="OcrOptions.MaxDegreeOfParallelism"/> threads inside the network kernels, which give the same results
/// whatever the number of threads; the workers that hold the threads and buffers come from <see cref="EnginePool"/>.
/// </summary>
internal sealed class OcrEngine : IPageRecognizer
{
    private static readonly LoadOnce<CharacterSet> s_characters = new(() => new CharacterSet(ModelStore.Characters.AsSpan(1, ModelStore.Characters.Length - 2).ToArray()));

    private OcrEngine()
    {
    }

    /// <summary>The one instance; the engine has no state of its own besides the pool.</summary>
    public static OcrEngine Instance { get; } = new();

    /// <summary>The recognition model's classes, built once from the model's character list.</summary>
    internal static CharacterSet Characters => s_characters.Value;

    /// <summary>
    /// The detection settings: the pipeline's defaults, which detect at most 1,536 pixels on the longer side (or the
    /// same area for pages more elongated than 3 to 1) and crop the lines from the full-resolution page (see
    /// <see cref="DetectionOptions.MaxSideLength"/>).
    /// </summary>
    internal static DetectionOptions DetectionOptions { get; } = new();

    /// <summary>
    /// The recognition settings for a call: the pipeline's defaults (orientation on three windows of long lines with
    /// a page vote, batches of six as rapidocr, spaces at word gaps) with the caller's rotation and confidence choices.
    /// </summary>
    internal static RecognitionOptions RecognitionOptions(OcrOptions options) => s_recognitionDefaults with
    {
        DetectRotation = options.DetectRotation,
        MinConfidence = options.MinConfidence,
    };

    private static readonly RecognitionOptions s_recognitionDefaults = new();

    /// <summary>
    /// The threads a page uses: the caller's limit, but no more than the machine has processors, since more threads
    /// than processors only take turns. The results do not depend on it.
    /// </summary>
    internal static int Threads(OcrOptions options) => Math.Clamp(options.MaxDegreeOfParallelism, 1, Environment.ProcessorCount);

    /// <inheritdoc/>
    public OcrPage Recognize(PagePixels pixels, int pageIndex, float dpi, OcrOptions options, CancellationToken cancellationToken)
        => Recognize(new PixelView(pixels.Pixels, pixels.Width, pixels.Height, pixels.Channels), pageIndex, dpi, options, cancellationToken);

    /// <summary>Reads a page of pixels that are already in the pipeline's layout, such as a caller's gray buffer.</summary>
    public static OcrPage Recognize(PixelView page, int pageIndex, float dpi, OcrOptions options, CancellationToken cancellationToken)
        => Recognize(page, pageIndex, dpi, options, Threads(options), cancellationToken);

    /// <summary>Reads a page with an exact number of threads, which tests use to check that it does not change results.</summary>
    internal static OcrPage Recognize(PixelView page, int pageIndex, float dpi, OcrOptions options, int threads, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (IsUniform(page))
        {
            // A page of one colour holds no text, yet the detector would still run on at least 736 pixels a side (small
            // pages are scaled up to that), which made a blank page cost over 100 ms whatever its size, and on some
            // colours it found a stray box that read as a character. Blank pages are common in scans.
            return new OcrPage(pageIndex, page.Width, page.Height, dpi, []);
        }

        var recognition = RecognitionOptions(options);
        EngineWorker worker = EnginePool.Rent(threads);
        PageResult result;
        try
        {
            result = worker.Run(page, dpi, recognition, s_observer?.Value, cancellationToken);
        }
        catch
        {
            // A worker that failed half way may hold a half-written plan or a stopped thread; a fresh one is cheap
            // next to the page that failed.
            worker.Dispose();
            throw;
        }

        EnginePool.Return(worker);
        return ToPage(result, pageIndex);
    }

    /// <summary>
    /// The observer of the network runs of the pages read in the current flow of execution, for tests (see
    /// <see cref="Observe"/>). It is created by the first test that observes, so the normal path checks only for null.
    /// </summary>
    private static AsyncLocal<RunObserver?>? s_observer;

    /// <summary>
    /// Hands <paramref name="observer"/> to the worker of every page read in the calling flow of execution until the
    /// result is disposed, so tests can cancel at a known operator through the public entry points, which take no
    /// observer of their own. The observer flows with the execution context: tests that run at the same time on other
    /// flows do not see it, and work the call starts on other tasks does.
    /// </summary>
    internal static IDisposable Observe(RunObserver observer)
    {
        ArgumentNullException.ThrowIfNull(observer);
        var created = new AsyncLocal<RunObserver?>();
        AsyncLocal<RunObserver?> local = Interlocked.CompareExchange(ref s_observer, created, null) ?? created;
        var scope = new ObserverScope(local, local.Value);
        local.Value = observer;
        return scope;
    }

    /// <summary>Puts back the observer that was there before <see cref="Observe"/>.</summary>
    private sealed class ObserverScope(AsyncLocal<RunObserver?> local, RunObserver? previous) : IDisposable
    {
        public void Dispose() => local.Value = previous;
    }

    /// <summary>
    /// Whether every pixel of the page is the same, alpha included: the data repeats with the period of one pixel, which
    /// one vectorised comparison finds.
    /// </summary>
    internal static bool IsUniform(PixelView page)
    {
        // PixelView has checked that width x height x channels bytes are there, so the product fits in an int.
        ReadOnlySpan<byte> data = page.Data[..(page.Width * page.Height * page.Channels)];
        int channels = page.Channels;
        return data[channels..].SequenceEqual(data[..^channels]);
    }

    /// <summary>The public page for a pipeline result: the same lines, words, quads and confidences.</summary>
    internal static OcrPage ToPage(PageResult result, int pageIndex)
    {
        var lines = new List<OcrLine>(result.Lines.Count);
        foreach (RecognizedLine line in result.Lines)
        {
            var words = new OcrWord[line.Words.Count];
            for (int i = 0; i < words.Length; i++)
            {
                RecognizedWord word = line.Words[i];
                words[i] = new OcrWord(word.Text, ToQuad(word.Quad), word.Confidence);
            }

            lines.Add(new OcrLine(SingleSpaces(line.Text), ToQuad(line.Quad), line.Confidence, words));
        }

        return new OcrPage(pageIndex, result.Width, result.Height, result.Dpi, lines);
    }

    private static OcrQuad ToQuad(Quad q) => new(
        new OcrPoint(q.P0.X, q.P0.Y),
        new OcrPoint(q.P1.X, q.P1.Y),
        new OcrPoint(q.P2.X, q.P2.Y),
        new OcrPoint(q.P3.X, q.P3.Y));

    /// <summary>
    /// The line's text with every run of white space as one space. The recogniser can emit two spaces for a wide gap,
    /// and the public line promises single spaces between its words.
    /// </summary>
    internal static string SingleSpaces(string text)
    {
        bool clean = true;
        for (int i = 0; i < text.Length && clean; i++)
        {
            char c = text[i];
            if (char.IsWhiteSpace(c) && (c != ' ' || (i + 1 < text.Length && char.IsWhiteSpace(text[i + 1]))))
            {
                clean = false;
            }
        }

        if (clean)
        {
            return text;
        }

        var sb = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                space = true;
                continue;
            }

            if (space && sb.Length > 0)
            {
                sb.Append(' ');
            }

            space = false;
            sb.Append(c);
        }

        return sb.ToString();
    }
}
