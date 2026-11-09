using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Recognition;

/// <summary>A recognised word: its text, confidence and quad in page pixels, P0 to P1 along the text.</summary>
internal readonly record struct RecognizedWord(string Text, float Confidence, Quad Quad);

/// <summary>
/// A recognised line in page pixels. This is the pipeline's internal result; <c>Layout/PageAssembler</c> orders lines
/// and maps them to the public result types.
/// </summary>
internal sealed class RecognizedLine
{
    public RecognizedLine(string text, float confidence, Quad quad, IReadOnlyList<RecognizedWord> words, int detectionIndex, bool upsideDown)
    {
        Text = text;
        Confidence = confidence;
        Quad = quad;
        Words = words;
        DetectionIndex = detectionIndex;
        UpsideDown = upsideDown;
    }

    /// <summary>The line's text, with a space wherever words were split at a gap the recogniser did not mark.</summary>
    public string Text { get; }

    /// <summary>rapidocr's line confidence, between 0 and 1.</summary>
    public float Confidence { get; }

    /// <summary>The line's quad, P0 to P1 along the text, P0 to P3 from the top of the text to its bottom.</summary>
    public Quad Quad { get; }

    public IReadOnlyList<RecognizedWord> Words { get; }

    /// <summary>The position of the line's box in the detector's output.</summary>
    public int DetectionIndex { get; }

    /// <summary>Whether the orientation classifier turned the line.</summary>
    public bool UpsideDown { get; }
}
