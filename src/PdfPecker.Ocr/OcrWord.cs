namespace PdfPecker.Ocr;

/// <summary>A word that OCR read: the text between two spaces of a line, with its position and confidence.</summary>
public sealed class OcrWord
{
    internal OcrWord(string text, OcrQuad quad, float confidence)
    {
        Text = text ?? throw new ArgumentNullException(nameof(text));
        Quad = quad;
        Box = OcrGeometry.Bounds(quad);
        Confidence = confidence;
    }

    /// <summary>The word's text, without spaces.</summary>
    public string Text { get; }

    /// <summary>The word's corners, in pixels of the page, following the line's direction.</summary>
    public OcrQuad Quad { get; }

    /// <summary>The upright rectangle around <see cref="Quad"/>, in pixels of the page.</summary>
    public OcrBox Box { get; }

    /// <summary>How sure the recognizer is of the word, from 0 to 1.</summary>
    public float Confidence { get; }

    /// <summary>The word's text, for debugging and logs.</summary>
    public override string ToString() => Text;
}
