using System.Text;

namespace PdfPecker.Ocr.Output;

/// <summary>Joins lines and pages into plain text.</summary>
internal static class PlainText
{
    /// <summary>The lines' texts, separated by line feeds.</summary>
    public static string Join(IReadOnlyList<OcrLine> lines)
    {
        if (lines.Count == 0) return "";
        if (lines.Count == 1) return lines[0].Text;
        int length = lines.Count - 1;
        foreach (var line in lines) length += line.Text.Length;
        var sb = new StringBuilder(length);
        for (int i = 0; i < lines.Count; i++)
        {
            if (i > 0) sb.Append('\n');
            sb.Append(lines[i].Text);
        }
        return sb.ToString();
    }

    /// <summary>The pages' texts, separated by form feeds as PdfPecker's text extraction and pdftotext separate them.</summary>
    public static string Join(IReadOnlyList<OcrPage> pages)
    {
        if (pages.Count == 0) return "";
        if (pages.Count == 1) return pages[0].Text;
        var sb = new StringBuilder();
        for (int i = 0; i < pages.Count; i++)
        {
            if (i > 0) sb.Append('\f');
            sb.Append(pages[i].Text);
        }
        return sb.ToString();
    }
}
