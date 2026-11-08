namespace PdfPecker.Ocr;

/// <summary>Information about the OCR package.</summary>
public static class OcrInfo
{
    /// <summary>The package's informational version.</summary>
    public static string Version => typeof(OcrInfo).Assembly.GetName().Version?.ToString() ?? "0.0.0";
}
