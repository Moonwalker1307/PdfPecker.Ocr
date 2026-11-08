using System.Net;
using System.Text;

namespace Corpus;

/// <summary>
/// Builds the HTML of one page and its ground truth text together, so that the two cannot drift apart. Every line of
/// text is a block with <c>white-space: nowrap</c>, which makes the line breaks the ones written here: the layout engine
/// never re-wraps, and the ground truth lines are exact. Long paragraphs are wrapped by <see cref="Wrap"/> before they
/// are written.
/// </summary>
sealed class Doc
{
    readonly StringBuilder _body = new();

    /// <summary>The ground truth lines in reading order.</summary>
    public List<string> Lines { get; } = [];

    public string Body => _body.ToString();

    static string Enc(string text) => WebUtility.HtmlEncode(text);

    /// <summary>One line of text. The style is extra CSS for the line's block.</summary>
    public Doc Line(string text, string style = "")
    {
        _body.Append("<div class=\"l\"").Append(style.Length > 0 ? $" style=\"{style}\"" : "").Append('>').Append(Enc(text)).Append("</div>\n");
        Lines.Add(text);
        return this;
    }

    public Doc Lines_(IEnumerable<string> lines, string style = "")
    {
        foreach (var l in lines) Line(l, style);
        return this;
    }

    /// <summary>A vertical gap that carries no text.</summary>
    public Doc Gap(string height)
    {
        _body.Append($"<div style=\"height:{height}\"></div>\n");
        return this;
    }

    /// <summary>
    /// A table. Each row is one ground truth line: its cells joined by single spaces, which is how an OCR engine reads a
    /// row. Right-aligned columns are given by <paramref name="right"/>; widths are CSS widths.
    /// </summary>
    public Doc Table(IReadOnlyList<string[]> rows, bool[] right, string[] widths, bool headerRow, string tableStyle = "",
        string headerStyle = "font-weight:bold", string stripe = "", string border = "")
    {
        _body.Append("<table class=\"t\" style=\"").Append(tableStyle).Append("\">\n");
        _body.Append("<colgroup>");
        foreach (var w in widths) _body.Append($"<col style=\"width:{w}\">");
        _body.Append("</colgroup>\n");
        for (int r = 0; r < rows.Count; r++)
        {
            bool head = headerRow && r == 0;
            string rowStyle = head ? headerStyle : (stripe.Length > 0 && r % 2 == 0 ? $"background:{stripe}" : "");
            _body.Append("<tr").Append(rowStyle.Length > 0 ? $" style=\"{rowStyle}\"" : "").Append('>');
            for (int c = 0; c < rows[r].Length; c++)
            {
                string cell = $"text-align:{(right[c] ? "right" : "left")};{border}";
                _body.Append($"<td style=\"{cell}\">").Append(Enc(rows[r][c])).Append("</td>");
            }
            _body.Append("</tr>\n");
            Lines.Add(string.Join(' ', rows[r]));
        }
        _body.Append("</table>\n");
        return this;
    }

    /// <summary>
    /// Columns set side by side. The ground truth reads the first column whole, then the next, which is how a reader
    /// takes a two-column article.
    /// </summary>
    public Doc Columns(IReadOnlyList<IReadOnlyList<string>> columns, string gap, string style = "")
    {
        _body.Append($"<div style=\"display:flex;gap:{gap}\">\n");
        foreach (var col in columns)
        {
            _body.Append($"<div style=\"flex:1 1 0;min-width:0;{style}\">\n");
            foreach (var line in col)
            {
                if (line.Length == 0) { _body.Append("<div style=\"height:0.9em\"></div>\n"); continue; }
                _body.Append("<div class=\"l\">").Append(Enc(line)).Append("</div>\n");
                Lines.Add(line);
            }
            _body.Append("</div>\n");
        }
        _body.Append("</div>\n");
        return this;
    }

    /// <summary>Wraps a paragraph to lines of at most <paramref name="width"/> characters, breaking only at spaces.</summary>
    public static List<string> Wrap(string paragraph, int width)
    {
        var lines = new List<string>();
        var line = new StringBuilder();
        foreach (var word in paragraph.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            if (line.Length > 0 && line.Length + 1 + word.Length > width)
            {
                lines.Add(line.ToString());
                line.Clear();
            }
            if (line.Length > 0) line.Append(' ');
            line.Append(word);
        }
        if (line.Length > 0) lines.Add(line.ToString());
        return lines;
    }

    /// <summary>The page's HTML. The margins are the ones the line widths in <see cref="Pages"/> are chosen for.</summary>
    public string ToHtml(string lang, string fontFamily, string fontSize, string lineHeight, string extraCss = "") => $$"""
        <!doctype html>
        <html lang="{{lang}}"><head><meta charset="utf-8"><title>corpus page</title>
        <style>
        @page { size: A4; margin: 20mm 18mm; }
        html, body { margin: 0; padding: 0; }
        body { font-family: {{fontFamily}}; font-size: {{fontSize}}; line-height: {{lineHeight}}; color: #000; }
        .l { white-space: nowrap; }
        table.t { border-collapse: collapse; }
        td { white-space: nowrap; padding: 2pt 6pt; }
        {{extraCss}}
        </style></head><body>
        {{Body}}
        </body></html>
        """;
}
