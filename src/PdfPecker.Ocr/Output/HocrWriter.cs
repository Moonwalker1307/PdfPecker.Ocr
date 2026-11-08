using System.Text;

namespace PdfPecker.Ocr.Output;

/// <summary>
/// Writes results as hOCR 1.2 (http://kba.github.io/hocr-spec/1.2/): XHTML in which <c>ocr_page</c>,
/// <c>ocr_line</c> and <c>ocrx_word</c> elements carry their boxes in the <c>title</c> attribute.
/// </summary>
/// <remarks>
/// Recognizer output is not trusted to be well-formed: characters XML forbids (control characters, lone surrogates,
/// U+FFFE and U+FFFF) become U+FFFD or a space, and boxes with coordinates that are not finite or lie off the page are
/// clamped to it, so the document always parses as XML.
/// </remarks>
internal static class HocrWriter
{
    /// <summary>Writes the pages as one hOCR document.</summary>
    public static string Write(IReadOnlyList<OcrPage> pages)
    {
        var sb = new StringBuilder(1024 + 256 * CountWords(pages));
        sb.Append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            .Append("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n")
            .Append("<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n")
            .Append(" <head>\n  <title></title>\n")
            .Append("  <meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />\n")
            .Append("  <meta name=\"ocr-system\" content=\"PdfPecker.Ocr ").Append(OcrInfo.Version).Append("\" />\n")
            .Append("  <meta name=\"ocr-number-of-pages\" content=\"").Append(pages.Count).Append("\" />\n")
            .Append("  <meta name=\"ocr-capabilities\" content=\"ocr_page ocr_line ocrx_word ocrp_wconf\" />\n")
            .Append(" </head>\n <body>\n");
        for (int p = 0; p < pages.Count; p++) WritePage(sb, pages[p], p + 1);
        sb.Append(" </body>\n</html>\n");
        return sb.ToString();
    }

    private static int CountWords(IReadOnlyList<OcrPage> pages)
    {
        long n = 0;
        foreach (var page in pages)
            foreach (var line in page.Lines) n += 1 + line.Words.Count;
        return (int)Math.Min(n, 1_000_000);
    }

    private static void WritePage(StringBuilder sb, OcrPage page, int number)
    {
        int width = Math.Max(0, page.Width), height = Math.Max(0, page.Height);
        sb.Append("  <div class=\"ocr_page\" id=\"page_").Append(number)
            .Append("\" title=\"bbox 0 0 ").Append(width).Append(' ').Append(height)
            .Append("; ppageno ").Append(Math.Max(0, page.PageIndex));
        if (float.IsFinite(page.Dpi) && page.Dpi > 0)
        {
            int dpi = (int)MathF.Round(Math.Min(page.Dpi, 100_000f));
            sb.Append("; scan_res ").Append(dpi).Append(' ').Append(dpi);
        }
        sb.Append("\">\n");
        for (int l = 0; l < page.Lines.Count; l++)
        {
            var line = page.Lines[l];
            sb.Append("   <span class=\"ocr_line\" id=\"line_").Append(number).Append('_').Append(l + 1).Append("\" title=\"");
            AppendBox(sb, line.Box, width, height);
            int angle = (int)Math.Round(OcrGeometry.IsFinite(line.Quad) ? OcrGeometry.AngleDegrees(line.Quad) : 0);
            angle = ((angle % 360) + 360) % 360;
            if (angle != 0) sb.Append("; textangle ").Append(angle);
            AppendConfidence(sb, line.Confidence);
            sb.Append("\">");
            if (line.Words.Count == 0)
            {
                // A line the recognizer gave no words for is still one word, so tools that read only words see its text.
                if (line.Text.Length > 0) WriteWord(sb, number, l + 1, 1, line.Text, line.Box, line.Confidence, width, height);
            }
            else
            {
                for (int w = 0; w < line.Words.Count; w++)
                {
                    var word = line.Words[w];
                    WriteWord(sb, number, l + 1, w + 1, word.Text, word.Box, word.Confidence, width, height);
                }
            }
            sb.Append("\n   </span>\n");
        }
        sb.Append("  </div>\n");
    }

    private static void WriteWord(StringBuilder sb, int page, int line, int index, string text, OcrBox box, float confidence, int width, int height)
    {
        sb.Append("\n    <span class=\"ocrx_word\" id=\"word_").Append(page).Append('_').Append(line).Append('_').Append(index)
            .Append("\" title=\"");
        AppendBox(sb, box, width, height);
        AppendConfidence(sb, confidence);
        sb.Append("\">");
        AppendText(sb, text);
        sb.Append("</span>");
    }

    /// <summary>The box as whole pixels that cover it, within the page.</summary>
    private static void AppendBox(StringBuilder sb, OcrBox box, int width, int height)
    {
        int x0 = Clamp(MathF.Floor(box.X), width), y0 = Clamp(MathF.Floor(box.Y), height);
        int x1 = Clamp(MathF.Ceiling(box.X + box.Width), width), y1 = Clamp(MathF.Ceiling(box.Y + box.Height), height);
        if (x1 < x0) x1 = x0;
        if (y1 < y0) y1 = y0;
        sb.Append("bbox ").Append(x0).Append(' ').Append(y0).Append(' ').Append(x1).Append(' ').Append(y1);
    }

    private static int Clamp(float value, int max)
        => float.IsNaN(value) ? 0 : value <= 0 ? 0 : value >= max ? max : (int)value;

    private static void AppendConfidence(StringBuilder sb, float confidence)
    {
        float c = float.IsFinite(confidence) ? Math.Clamp(confidence, 0f, 1f) : 0f;
        sb.Append("; x_wconf ").Append((int)MathF.Round(c * 100));
    }

    /// <summary>Appends text escaped for XML, replacing the characters XML does not allow.</summary>
    internal static void AppendText(StringBuilder sb, string text)
    {
        var span = text.AsSpan();
        int start = 0;
        for (int i = 0; i < span.Length; i++)
        {
            char c = span[i];
            string? replacement;
            if (c == '<') replacement = "&lt;";
            else if (c == '>') replacement = "&gt;";
            else if (c == '&') replacement = "&amp;";
            else if (c == '"') replacement = "&quot;";
            else if (c < 0x20 || c is '￾' or '￿') replacement = " ";
            else if (char.IsHighSurrogate(c) && i + 1 < span.Length && char.IsLowSurrogate(span[i + 1])) { i++; continue; }
            else if (char.IsSurrogate(c)) replacement = "�";
            else continue;
            sb.Append(span[start..i]).Append(replacement);
            start = i + 1;
        }
        sb.Append(span[start..]);
    }
}
