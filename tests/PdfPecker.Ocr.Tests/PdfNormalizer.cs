using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace PdfPecker.Ocr.Tests;

/// <summary>
/// Turns a PDF with a classic cross-reference table into text that stays the same when only its compression differs:
/// every object in file order with its dictionary, Flate streams inflated, text streams written out and binary ones
/// (images, the font program) as their length and SHA-256, and stream lengths, the cross-reference offsets and the
/// file identifier left out. .NET 9's zlib compresses the same bytes differently from .NET 8 and 10, so raw bytes
/// cannot be a baseline that every runtime meets, but the inflated content can.
/// </summary>
internal static partial class PdfNormalizer
{
    public static string Normalize(byte[] pdf)
    {
        string latin1 = Encoding.Latin1.GetString(pdf);
        var sb = new StringBuilder();
        Match header = Header().Match(latin1);
        Assert.True(header.Success, "The output does not start with a PDF header.");
        sb.Append(header.Value).Append('\n');

        int position = 0;
        int objects = 0;
        while (true)
        {
            Match start = ObjectStart().Match(latin1, position);
            if (!start.Success) break;
            objects++;
            int bodyStart = start.Index + start.Length;
            int streamKeyword = StreamKeyword().Match(latin1, bodyStart) is { Success: true } s ? s.Index : -1;
            int end = latin1.IndexOf("endobj", bodyStart, StringComparison.Ordinal);
            Assert.True(end > 0, $"Object {start.Groups[1].Value} has no endobj.");
            sb.Append("obj ").Append(start.Groups[1].Value).Append(' ').Append(start.Groups[2].Value).Append('\n');
            if (streamKeyword < 0 || streamKeyword > end)
            {
                sb.Append(Tidy(latin1[bodyStart..end])).Append('\n');
                position = end + 6;
                continue;
            }

            string dictionary = latin1[bodyStart..streamKeyword];
            Match length = LengthEntry().Match(dictionary);
            Assert.True(length.Success, $"Object {start.Groups[1].Value} has no direct /Length.");
            int dataStart = streamKeyword + "stream".Length;
            if (latin1[dataStart] == '\r') dataStart++;
            if (latin1[dataStart] == '\n') dataStart++;
            int dataLength = int.Parse(length.Groups[1].Value, CultureInfo.InvariantCulture);
            byte[] data = pdf.AsSpan(dataStart, dataLength).ToArray();
            bool flate = dictionary.Contains("/FlateDecode", StringComparison.Ordinal);
            if (flate) data = Inflate(data);

            sb.Append(Tidy(LengthEntry().Replace(dictionary, "/Length *"))).Append('\n');
            if (IsText(data))
            {
                sb.Append("stream (").Append(data.Length.ToString(CultureInfo.InvariantCulture)).Append(" bytes)\n");
                sb.Append(Encoding.Latin1.GetString(data).Replace("\r\n", "\n", StringComparison.Ordinal).TrimEnd('\n')).Append('\n');
                sb.Append("endstream\n");
            }
            else
            {
                sb.Append("binary stream (").Append(data.Length.ToString(CultureInfo.InvariantCulture)).Append(" bytes) sha256 ")
                    .Append(Convert.ToHexString(SHA256.HashData(data))).Append('\n');
            }

            position = latin1.IndexOf("endobj", dataStart + dataLength, StringComparison.Ordinal) + 6;
        }

        Assert.True(objects > 0, "The output has no objects.");
        int trailer = latin1.LastIndexOf("trailer", StringComparison.Ordinal);
        Assert.True(trailer > 0, "The output has no trailer; only files with a cross-reference table are normalised.");
        int startxref = latin1.IndexOf("startxref", trailer, StringComparison.Ordinal);
        string trailerDictionary = IdEntry().Replace(latin1[(trailer + 7)..startxref], "");
        sb.Append("trailer\n").Append(Tidy(trailerDictionary)).Append('\n');
        return sb.ToString();
    }

    private static byte[] Inflate(byte[] data)
    {
        using var input = new MemoryStream(data);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);
        return output.ToArray();
    }

    /// <summary>Content streams, CMaps and the like are text; images and font programs are not.</summary>
    private static bool IsText(byte[] data)
    {
        foreach (byte b in data)
        {
            if (b is < 0x20 and not (0x0A or 0x0D or 0x09) || b > 0x7E) return false;
        }

        return true;
    }

    private static string Tidy(string s) => Whitespace().Replace(s, " ").Trim();

    [GeneratedRegex(@"^%PDF-\d\.\d")]
    private static partial Regex Header();

    [GeneratedRegex(@"(?<=[\r\n])(\d+) (\d+) obj\b")]
    private static partial Regex ObjectStart();

    [GeneratedRegex(@"\bstream\r?\n")]
    private static partial Regex StreamKeyword();

    [GeneratedRegex(@"/Length (\d+)(?! \d+ R)")]
    private static partial Regex LengthEntry();

    [GeneratedRegex(@"/ID\s*\[\s*<[0-9A-Fa-f]*>\s*<[0-9A-Fa-f]*>\s*\]")]
    private static partial Regex IdEntry();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();
}
