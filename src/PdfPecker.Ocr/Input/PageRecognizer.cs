namespace PdfPecker.Ocr.Input;

/// <summary>The recognizer the public API uses.</summary>
internal static class PageRecognizer
{
    /// <summary>
    /// The OCR engine. Until the engine is wired in, this throws, so the public entry points fail plainly instead of
    /// returning documents without text.
    /// </summary>
    /// <exception cref="NotSupportedException">No engine is available.</exception>
    public static IPageRecognizer Default => throw new NotSupportedException("The OCR engine is not part of this build.");
}
