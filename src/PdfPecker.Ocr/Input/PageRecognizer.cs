using PdfPecker.Ocr.Layout;

namespace PdfPecker.Ocr.Input;

/// <summary>The recognizer the public API uses.</summary>
internal static class PageRecognizer
{
    /// <summary>The OCR engine: the bundled models through the pipeline (see <see cref="OcrEngine"/>).</summary>
    public static IPageRecognizer Default => OcrEngine.Instance;
}
