using System.Text;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Layout;

/// <summary>
/// A recognised page in the pipeline's internal types: its lines in reading order, grouped into blocks, and the page
/// text. <see cref="PageAssembler"/> builds it; the public <c>OcrPage</c> is made from it in one place, so the pipeline
/// does not depend on the public result types.
/// </summary>
internal sealed class PageResult
{
    public PageResult(int width, int height, float dpi, IReadOnlyList<IReadOnlyList<RecognizedLine>> blocks)
    {
        Width = width;
        Height = height;
        Dpi = dpi;
        Blocks = blocks;
        var lines = new List<RecognizedLine>();
        foreach (IReadOnlyList<RecognizedLine> block in blocks)
        {
            lines.AddRange(block);
        }

        Lines = lines;
    }

    public int Width { get; }

    public int Height { get; }

    public float Dpi { get; }

    /// <summary>The lines in reading order.</summary>
    public IReadOnlyList<RecognizedLine> Lines { get; }

    /// <summary>The same lines in blocks (columns, rows, cells) as the reading order found them.</summary>
    public IReadOnlyList<IReadOnlyList<RecognizedLine>> Blocks { get; }

    /// <summary>The lines' text in reading order, one line per line.</summary>
    public string Text
    {
        get
        {
            var sb = new StringBuilder();
            foreach (RecognizedLine line in Lines)
            {
                if (sb.Length > 0)
                {
                    sb.Append('\n');
                }

                sb.Append(line.Text);
            }

            return sb.ToString();
        }
    }

    /// <summary>The mean line confidence weighted by the lines' lengths, 0 for a page without text.</summary>
    public float Confidence
    {
        get
        {
            double sum = 0, weight = 0;
            foreach (RecognizedLine line in Lines)
            {
                sum += line.Confidence * line.Text.Length;
                weight += line.Text.Length;
            }

            return weight == 0 ? 0 : (float)(sum / weight);
        }
    }
}

/// <summary>Builds a page from recognised lines.</summary>
internal static class PageAssembler
{
    public static PageResult Assemble(int width, int height, float dpi, IReadOnlyList<RecognizedLine> lines)
    {
        List<List<RecognizedLine>> blocks = ReadingOrder.Order(lines);
        return new PageResult(width, height, dpi, blocks);
    }
}
