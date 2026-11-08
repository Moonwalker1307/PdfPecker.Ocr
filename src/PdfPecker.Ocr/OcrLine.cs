using System.Collections.ObjectModel;

namespace PdfPecker.Ocr;

/// <summary>A line of text that OCR read, with its words, position and confidence.</summary>
public sealed class OcrLine
{
    internal OcrLine(string text, OcrQuad quad, float confidence, IEnumerable<OcrWord> words)
    {
        Text = text ?? throw new ArgumentNullException(nameof(text));
        ArgumentNullException.ThrowIfNull(words);
        Quad = quad;
        Box = OcrGeometry.Bounds(quad);
        Confidence = confidence;
        Words = new ReadOnlyCollection<OcrWord>(words.ToArray());
    }

    /// <summary>The line's text, its words separated by single spaces.</summary>
    public string Text { get; }

    /// <summary>The line's corners, in pixels of the page; the text runs from the top-left to the top-right corner.</summary>
    public OcrQuad Quad { get; }

    /// <summary>The upright rectangle around <see cref="Quad"/>, in pixels of the page.</summary>
    public OcrBox Box { get; }

    /// <summary>How sure the recognizer is of the line, from 0 to 1.</summary>
    public float Confidence { get; }

    /// <summary>The line's words, in reading order.</summary>
    public IReadOnlyList<OcrWord> Words { get; }

    /// <summary>The line's text, for debugging and logs.</summary>
    public override string ToString() => Text;
}
