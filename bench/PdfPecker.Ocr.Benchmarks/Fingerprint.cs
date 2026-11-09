using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using PdfPecker.Ocr;
using TesseractComparison;

namespace OcrBench;

/// <summary>
/// A fingerprint of everything the public API returns for each corpus page (line and word texts, quads and confidences,
/// to the last bit), and of the whole corpus, so two versions can be shown to give identical output:
///   dotnet run -c Release -- fingerprint [--corpus dir] [--threads n] [--only id,=id]
/// It uses only the public API, so it builds against any version that has one.
/// </summary>
static class Fingerprint
{
    public static void Run(string corpus, string[]? only, int threads)
    {
        using var all = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        int pages = 0;
        foreach (var page in CorpusPage.Load(corpus, only))
        {
            var (pixels, w, h, c) = PngDecoder.Decode(File.ReadAllBytes(page.ImagePath));
            var options = new OcrOptions { MaxDegreeOfParallelism = threads, Dpi = page.Dpi > 0 ? page.Dpi : 300 };
            OcrPage result = c == 1 ? ImageOcr.RecognizeGray(pixels, w, h, options) : ImageOcr.RecognizeRgba(ToRgba(pixels, w, h, c), w, h, options);
            var sb = new StringBuilder();
            foreach (var line in result.Lines)
            {
                sb.Append("L\t").Append(line.Text).Append('\t').Append(Bits(line.Confidence)).Append('\t');
                Quad(sb, line.Quad);
                sb.Append('\n');
                foreach (var word in line.Words)
                {
                    sb.Append("W\t").Append(word.Text).Append('\t').Append(Bits(word.Confidence)).Append('\t');
                    Quad(sb, word.Quad);
                    sb.Append('\n');
                }
            }

            byte[] bytes = Encoding.UTF8.GetBytes(sb.ToString());
            all.AppendData(Encoding.UTF8.GetBytes(page.Id + "\n"));
            all.AppendData(bytes);
            pages++;
            Console.WriteLine($"{page.Id}\t{result.Lines.Count}\t{Convert.ToHexString(SHA256.HashData(bytes))[..16]}");
        }

        Console.WriteLine($"corpus\t{pages}\t{Convert.ToHexString(all.GetHashAndReset())[..16]}");
    }

    static string Bits(float value) => BitConverter.SingleToInt32Bits(value).ToString("x8", CultureInfo.InvariantCulture);

    static void Quad(StringBuilder sb, OcrQuad q)
    {
        foreach (var p in new[] { q.TopLeft, q.TopRight, q.BottomRight, q.BottomLeft })
            sb.Append(Bits(p.X)).Append(',').Append(Bits(p.Y)).Append(' ');
    }

    static byte[] ToRgba(byte[] pixels, int w, int h, int channels)
    {
        var rgba = new byte[w * h * 4];
        for (int i = 0; i < w * h; i++)
        {
            rgba[i * 4] = pixels[i * channels];
            rgba[i * 4 + 1] = pixels[i * channels + (channels >= 3 ? 1 : 0)];
            rgba[i * 4 + 2] = pixels[i * channels + (channels >= 3 ? 2 : 0)];
            rgba[i * 4 + 3] = channels == 4 ? pixels[i * channels + 3] : (byte)255;
        }

        return rgba;
    }
}
