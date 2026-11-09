using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Layout;

/// <summary>
/// One page through the whole pipeline: detection, orientation, recognition, words and reading order. The networks
/// come in through <see cref="INetwork"/>; one instance per worker thread, since the stages reuse their buffers.
/// </summary>
internal sealed class OcrPipeline
{
    private readonly TextDetector _detector;
    private readonly TextRecognizer _recognizer;

    /// <summary>
    /// A pipeline over the given networks; <paramref name="context"/>, the one the networks run in, lends its threads to
    /// the work between them as well.
    /// </summary>
    public OcrPipeline(INetwork detection, INetwork recognition, INetwork? orientation, CharacterSet characters, DetectionOptions? detectionOptions = null, RecognitionOptions? recognitionOptions = null, INetwork? detectionGray = null, InferenceContext? context = null)
    {
        _detector = new TextDetector(detection, detectionOptions, detectionGray);
        _recognizer = new TextRecognizer(recognition, orientation, characters, recognitionOptions, context);
    }

    public TextDetector Detector => _detector;

    public TextRecognizer Recognizer => _recognizer;

    /// <summary>Reads the text of a page image whose resolution is <paramref name="dpi"/>.</summary>
    public PageResult Run(PixelView page, float dpi, CancellationToken cancellationToken = default)
    {
        List<DetectedQuad> quads = _detector.Detect(page, dpi, cancellationToken);
        List<RecognizedLine> lines = _recognizer.Recognize(page, quads, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return PageAssembler.Assemble(page.Width, page.Height, dpi, lines);
    }
}
