using System.Globalization;
using System.Net;
using PdfPecker;

namespace HeadTuner;

/// <summary>How a training page looks: font, size, weight, style, resolution, and how it is degraded like a scan.</summary>
internal sealed record PageStyle(string Family, int Points, bool Bold, bool Italic, int Dpi, int BlurPasses, int Noise, int LowContrast, uint Seed)
{
    /// <summary>The fonts the tuner trains on: PdfPecker's bundled ones, so every machine renders the same pages.</summary>
    public static readonly string[] Families = ["Liberation Serif", "Liberation Sans", "Liberation Mono"];

    private static readonly int[] Sizes = [8, 9, 10, 11, 12, 14, 16, 20];

    /// <summary>A style drawn from <paramref name="random"/>: four in ten pages clean, the others blurred and noisy.</summary>
    public static PageStyle Draw(Random random)
    {
        int degraded = random.Next(10);
        return new PageStyle(
            Families[random.Next(Families.Length)],
            Sizes[random.Next(Sizes.Length)],
            random.Next(5) == 0,
            random.Next(6) == 0,
            random.Next(4) == 0 ? 200 : 300,
            degraded < 4 ? 0 : random.Next(1, 3),
            degraded < 4 ? 0 : random.Next(10, 50),
            degraded < 6 ? 0 : random.Next(1, 3),
            (uint)random.Next());
    }

    public override string ToString() => string.Create(CultureInfo.InvariantCulture,
        $"{Family} {Points}pt{(Bold ? " bold" : "")}{(Italic ? " italic" : "")} {Dpi} dpi blur {BlurPasses} noise {Noise} contrast {LowContrast}");
}

/// <summary>Renders lines of text as a page of pixels with PdfPecker, one paragraph per line.</summary>
internal static class Pages
{
    public static (byte[] Rgba, int Width, int Height) Render(IReadOnlyList<string> lines, PageStyle style)
    {
        string body = string.Concat(lines.Select(l => "<p>" + WebUtility.HtmlEncode(l) + "</p>"));
        string css = string.Create(CultureInfo.InvariantCulture,
            $"@page{{size:200mm {Math.Max(30, (lines.Count * style.Points * 0.62) + 16):0}mm;margin:8mm}}body{{margin:0}}" +
            $"p{{font-family:'{style.Family}';font-size:{style.Points}pt;line-height:1.7;margin:0;white-space:pre;" +
            $"font-weight:{(style.Bold ? 700 : 400)};font-style:{(style.Italic ? "italic" : "normal")}}}");
        string html = $"<!doctype html><html><head><meta charset=\"utf-8\"><style>{css}</style></head><body>{body}</body></html>";
        byte[] pdf = HtmlToPdf.Convert(html);
        var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = style.Dpi });
        byte[] rgba = image.Rgba;
        if (style.BlurPasses > 0 || style.Noise > 0 || style.LowContrast > 0) rgba = Degrade(rgba, image.Width, image.Height, style);
        return (rgba, image.Width, image.Height);
    }

    /// <summary>
    /// A poor copy: blurred with a 3 by 3 box, its contrast lowered, and noise added from a fixed xorshift sequence, as
    /// the golden tests' degraded page is made.
    /// </summary>
    private static byte[] Degrade(byte[] rgba, int width, int height, PageStyle style)
    {
        var gray = new int[width * height];
        for (int i = 0; i < gray.Length; i++) gray[i] = rgba[(i * 4) + 1];
        for (int pass = 0; pass < style.BlurPasses; pass++) gray = BoxBlur(gray, width, height);
        uint state = style.Seed | 1;
        int low = style.LowContrast * 30, high = 255 - (style.LowContrast * 20);
        var result = new byte[rgba.Length];
        for (int i = 0; i < gray.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            int noise = style.Noise == 0 ? 0 : (int)(state % (uint)((2 * style.Noise) + 1)) - style.Noise;
            byte v = (byte)Math.Clamp(low + (gray[i] * (high - low) / 255) + noise, 0, 255);
            result[i * 4] = result[(i * 4) + 1] = result[(i * 4) + 2] = v;
            result[(i * 4) + 3] = 255;
        }

        return result;
    }

    private static int[] BoxBlur(int[] source, int width, int height)
    {
        var target = new int[source.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int sum = 0, n = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    int yy = y + dy;
                    if (yy < 0 || yy >= height) continue;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = x + dx;
                        if (xx < 0 || xx >= width) continue;
                        sum += source[(yy * width) + xx];
                        n++;
                    }
                }

                target[(y * width) + x] = sum / n;
            }
        }

        return target;
    }
}
