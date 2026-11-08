using System.Globalization;
using System.Text;
using PdfPecker.Ocr.Input;

namespace PdfPecker.Ocr.Tests;

/// <summary>A word for <see cref="FakeRecognizer"/>, in points on the page as displayed, from its top-left corner.</summary>
/// <param name="Text">The word.</param>
/// <param name="X">The left edge of the word's start, in points.</param>
/// <param name="Y">The top edge of the word's start, in points from the top.</param>
/// <param name="Width">The length along the text direction, in points.</param>
/// <param name="Height">The height across the text direction, in points.</param>
/// <param name="Angle">The text direction in degrees counter-clockwise; 90 reads upwards.</param>
internal readonly record struct WordSpec(string Text, double X, double Y, double Width, double Height, double Angle = 0);

/// <summary>
/// A recognizer that "reads" words given in advance, one line per inner list, converting them to the pixels it is
/// handed so the tests know exactly what the text layer must hold and where.
/// </summary>
internal sealed class FakeRecognizer : IPageRecognizer
{
    private readonly Func<int, IReadOnlyList<IReadOnlyList<WordSpec>>> _lines;

    public FakeRecognizer(Func<int, IReadOnlyList<IReadOnlyList<WordSpec>>> lines) => _lines = lines;

    public FakeRecognizer(params WordSpec[][] lines) : this(_ => lines) { }

    public List<(int PageIndex, int Width, int Height, float Dpi, PixelFormat Format)> Calls { get; } = new();

    public OcrPage Recognize(PagePixels pixels, int pageIndex, float dpi, OcrOptions options, CancellationToken cancellationToken)
    {
        lock (Calls) Calls.Add((pageIndex, pixels.Width, pixels.Height, dpi, pixels.Format));
        double s = dpi / 72.0;
        var lines = new List<OcrLine>();
        foreach (var specs in _lines(pageIndex))
        {
            var words = specs.Select(w => new OcrWord(w.Text, Quad(w, s), 0.9f)).ToList();
            var quad = words.Count == 0 ? default : new OcrQuad(words[0].Quad.TopLeft, words[^1].Quad.TopRight, words[^1].Quad.BottomRight, words[0].Quad.BottomLeft);
            lines.Add(new OcrLine(string.Join(' ', specs.Select(w => w.Text)), quad, 0.9f, words));
        }
        return new OcrPage(pageIndex, pixels.Width, pixels.Height, dpi, lines);
    }

    /// <summary>The quad of a word in pixels at <paramref name="scale"/> pixels per point.</summary>
    public static OcrQuad Quad(WordSpec w, double scale)
    {
        double a = w.Angle * Math.PI / 180;
        // In page pixels the y axis points down: the text direction and the direction from its top to its bottom.
        double dx = Math.Cos(a), dy = -Math.Sin(a), downX = Math.Sin(a), downY = Math.Cos(a);
        double x = w.X * scale, y = w.Y * scale, len = w.Width * scale, h = w.Height * scale;
        OcrPoint P(double px, double py) => new((float)px, (float)py);
        return new OcrQuad(P(x, y), P(x + len * dx, y + len * dy), P(x + len * dx + h * downX, y + len * dy + h * downY), P(x + h * downX, y + h * downY));
    }
}

/// <summary>Builds small PDFs by hand, for page boxes and rotations that HTML conversion does not make.</summary>
internal static class RawPdf
{
    /// <summary>A page: its media box, optional crop box, rotation and content stream.</summary>
    public sealed record Page(double[] MediaBox, double[]? CropBox = null, int Rotate = 0, string Content = "", bool Helvetica = false);

    public static byte[] Build(params Page[] pages)
    {
        var objects = new List<string>
        {
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Count " + pages.Length + " /Kids [" + string.Join(" ", pages.Select((_, i) => (4 + 2 * i) + " 0 R")) + "] >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        };
        foreach (var p in pages)
        {
            var dict = new StringBuilder("<< /Type /Page /Parent 2 0 R /MediaBox [" + Nums(p.MediaBox) + "]");
            if (p.CropBox != null) dict.Append(" /CropBox [").Append(Nums(p.CropBox)).Append(']');
            if (p.Rotate != 0) dict.Append(" /Rotate ").Append(p.Rotate.ToString(CultureInfo.InvariantCulture));
            dict.Append(" /Resources << ").Append(p.Helvetica ? "/Font << /F1 3 0 R >> " : "").Append(">>");
            dict.Append(" /Contents ").Append((objects.Count + 2).ToString(CultureInfo.InvariantCulture)).Append(" 0 R >>");
            objects.Add(dict.ToString());
            objects.Add("<< /Length " + Encoding.Latin1.GetByteCount(p.Content) + " >>\nstream\n" + p.Content + "\nendstream");
        }
        var pdf = new StringBuilder("%PDF-1.7\n");
        var offsets = new List<int>();
        for (int i = 0; i < objects.Count; i++)
        {
            offsets.Add(Encoding.Latin1.GetByteCount(pdf.ToString()));
            pdf.Append(i + 1).Append(" 0 obj\n").Append(objects[i]).Append("\nendobj\n");
        }
        int xref = Encoding.Latin1.GetByteCount(pdf.ToString());
        pdf.Append("xref\n0 ").Append(objects.Count + 1).Append("\n0000000000 65535 f \n");
        foreach (var o in offsets) pdf.Append(o.ToString("D10", CultureInfo.InvariantCulture)).Append(" 00000 n \n");
        pdf.Append("trailer\n<< /Size ").Append(objects.Count + 1).Append(" /Root 1 0 R >>\nstartxref\n").Append(xref).Append("\n%%EOF\n");
        return Encoding.Latin1.GetBytes(pdf.ToString());
    }

    /// <summary>Content that paints a few grey and black rectangles, so a scan is not blank.</summary>
    public static string Shapes(double w, double h) => FormattableString.Invariant(
        $"0.8 g 0 0 {w} {h} re f 0 g {w * 0.1} {h * 0.1} {w * 0.3} {h * 0.05} re f {w * 0.5} {h * 0.7} {w * 0.2} {h * 0.2} re f");

    private static string Nums(double[] v) => string.Join(" ", v.Select(x => x.ToString(CultureInfo.InvariantCulture)));
}

internal static class Pixels
{
    /// <summary>Renders every page at the resolution and returns their RGBA bytes.</summary>
    public static List<byte[]> Render(byte[] pdf, float dpi)
    {
        var r = PdfRenderer.Open(pdf);
        return Enumerable.Range(0, r.PageCount).Select(i => r.RenderPage(i, new ImageOptions { Dpi = dpi }).Rgba).ToList();
    }
}
