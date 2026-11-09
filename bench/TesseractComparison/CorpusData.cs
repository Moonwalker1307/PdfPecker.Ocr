using System.Globalization;
using System.IO.Compression;
using System.Text;

namespace TesseractComparison;

/// <summary>One page of the corpus, as listed in manifest.tsv.</summary>
sealed record CorpusPage(string Id, string ImagePath, int Width, int Height, int Channels, int Dpi, string Lang, string Category,
    string Degradation, int Rotation, string TextPath, string BoxesPath)
{
    /// <summary>The ground truth lines, in reading order.</summary>
    public string[] LoadLines() => File.ReadAllLines(TextPath, Encoding.UTF8);

    public List<Box> LoadBoxes()
    {
        var list = new List<Box>();
        foreach (var line in File.ReadLines(BoxesPath, Encoding.UTF8))
        {
            var p = line.Split('\t');
            if (p.Length < 5) continue;
            list.Add(new Box(p[4], double.Parse(p[0], CultureInfo.InvariantCulture), double.Parse(p[1], CultureInfo.InvariantCulture),
                double.Parse(p[2], CultureInfo.InvariantCulture), double.Parse(p[3], CultureInfo.InvariantCulture), 1));
        }
        return list;
    }

    public static List<CorpusPage> Load(string dir, string[]? only)
    {
        var pages = new List<CorpusPage>();
        var lines = File.ReadAllLines(Path.Combine(dir, "manifest.tsv"), Encoding.UTF8);
        for (int i = 1; i < lines.Length; i++)
        {
            var p = lines[i].Split('\t');
            if (p.Length < 13) continue;
            // An entry that starts with "=" names one page exactly; any other entry selects the pages whose id contains it.
            if (only is not null && !only.Any(o => o.StartsWith('=') ? p[0] == o[1..] : p[0].Contains(o, StringComparison.Ordinal))) continue;
            pages.Add(new CorpusPage(p[0], Path.Combine(dir, p[1]), I(p[2]), I(p[3]), I(p[4]), I(p[5]), p[6], p[7], p[9], I(p[10]),
                Path.Combine(dir, p[11]), Path.Combine(dir, p[12])));
        }
        // Grouped by language so that an engine that loads a model per language loads each once.
        return pages.OrderBy(p => p.Lang, StringComparer.Ordinal).ThenBy(p => p.Id, StringComparer.Ordinal).ToList();
    }

    static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);
}

/// <summary>A box in page pixels, origin top-left, with the engine's confidence in [0, 1] (1 for ground truth).</summary>
readonly record struct Box(string Text, double X, double Y, double Width, double Height, double Confidence);

/// <summary>Decodes 8-bit, non-interlaced PNGs (gray, gray+alpha, RGB, RGBA) to gray or RGB pixels.</summary>
static class PngDecoder
{
    public static (byte[] Pixels, int Width, int Height, int Channels) Decode(byte[] png)
    {
        if (png.Length < 8 || png[0] != 0x89 || png[1] != 0x50) throw new InvalidDataException("not a PNG");
        int pos = 8, width = 0, height = 0, colorType = 0;
        using var idat = new MemoryStream();
        while (pos + 8 <= png.Length)
        {
            int len = (png[pos] << 24) | (png[pos + 1] << 16) | (png[pos + 2] << 8) | png[pos + 3];
            string type = Encoding.ASCII.GetString(png, pos + 4, 4);
            int data = pos + 8;
            if (type == "IHDR")
            {
                width = (png[data] << 24) | (png[data + 1] << 16) | (png[data + 2] << 8) | png[data + 3];
                height = (png[data + 4] << 24) | (png[data + 5] << 16) | (png[data + 6] << 8) | png[data + 7];
                if (png[data + 8] != 8 || png[data + 12] != 0) throw new NotSupportedException("only 8-bit non-interlaced PNGs");
                colorType = png[data + 9];
            }
            else if (type == "IDAT") idat.Write(png, data, len);
            else if (type == "IEND") break;
            pos = data + len + 4;
        }
        int channels = colorType switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => throw new NotSupportedException("palette PNGs") };
        int stride = width * channels;
        var raw = new byte[(stride + 1) * height];
        idat.Position = 0;
        using (var z = new ZLibStream(idat, CompressionMode.Decompress)) z.ReadExactly(raw);
        var pixels = new byte[stride * height];
        for (int y = 0; y < height; y++)
        {
            int filter = raw[y * (stride + 1)];
            int src = y * (stride + 1) + 1, dst = y * stride;
            for (int x = 0; x < stride; x++)
            {
                int a = x >= channels ? pixels[dst + x - channels] : 0;
                int b = y > 0 ? pixels[dst - stride + x] : 0;
                int c = x >= channels && y > 0 ? pixels[dst - stride + x - channels] : 0;
                int v = raw[src + x];
                pixels[dst + x] = (byte)(filter switch
                {
                    0 => v,
                    1 => v + a,
                    2 => v + b,
                    3 => v + ((a + b) >> 1),
                    4 => v + Paeth(a, b, c),
                    _ => throw new InvalidDataException("bad PNG filter"),
                });
            }
        }
        if (channels == 1 || channels == 3) return (pixels, width, height, channels);
        // Drop the alpha channel: pages are opaque.
        int outChannels = channels == 2 ? 1 : 3;
        var outPixels = new byte[width * height * outChannels];
        for (int i = 0, o = 0; i < pixels.Length; i += channels)
            for (int k = 0; k < outChannels; k++) outPixels[o++] = pixels[i + k];
        return (outPixels, width, height, outChannels);
    }

    static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}
