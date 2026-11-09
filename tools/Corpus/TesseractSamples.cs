using System.Security.Cryptography;
using PdfPecker;

namespace Corpus;

/// <summary>
/// Tesseract's own sample images and their text, from the tesseract-ocr/test repository (Apache-2.0, like Tesseract). They are
/// fetched only when asked for with --tesseract-samples, pinned by SHA-256, and never checked in: they are small photographs
/// of printed text with a hand-written text file, a different kind of input from the generated pages. The TIFFs are decoded
/// by PdfPecker (ImageToPdf, then a render at the image's own size), because the public API has no image decoder.
/// </summary>
static class TesseractSamples
{
    // The commit of tesseract-ocr/test the hashes below were taken from.
    const string Commit = "main";

    sealed record Sample(string Name, string Tiff, string TiffSha256, string Text, string TextSha256, int Width, int Height, string Lang);

    static readonly Sample[] Samples =
    [
        new("tesseract-phototest", "testing/phototest.tif", "d2241a1eb6d6cd2eb6544b8e228c2862c852b7467d16ed636caa32179b780692",
            "testing/phototest.txt", "76ef157baeb6ab275b7d90f310477a8060d5822aea2b2e05d480f459b016a6ed", 640, 480, "en"),
        new("tesseract-eurotext", "testing/eurotext.tif", "7b9bd14aba7d5e30df686fbb6f71782a97f48f81b32dc201a1b75afe6de747d6",
            "testing/eurotext.txt", "26a43b5107b51054910b89f6fea67a03125b5737c0200e588121d4d132c7f3cb", 1024, 800, "multi"),
    ];

    /// <summary>Fetches, checks and decodes the samples; returns each as a gray bitmap with its text lines.</summary>
    public static List<(string Name, string Lang, Bitmap Image, string[] Lines)> Load()
    {
        using var http = new HttpClient();
        var result = new List<(string, string, Bitmap, string[])>();
        foreach (var s in Samples)
        {
            var tiff = Fetch(http, s.Tiff, s.TiffSha256);
            var text = Fetch(http, s.Text, s.TextSha256);
            var pdf = ImageToPdf.Convert([tiff]);
            var page = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Width = s.Width, Height = s.Height });
            var bitmap = Bitmap.FromRgba(page.Rgba, page.Width, page.Height, colour: false);
            var lines = System.Text.Encoding.UTF8.GetString(text).Split('\n', StringSplitOptions.RemoveEmptyEntries);
            result.Add((s.Name, s.Lang, bitmap, lines));
        }
        return result;
    }

    static byte[] Fetch(HttpClient http, string path, string sha256)
    {
        var data = http.GetByteArrayAsync($"https://raw.githubusercontent.com/tesseract-ocr/test/{Commit}/{path}").GetAwaiter().GetResult();
        string actual = Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
        if (actual != sha256) throw new InvalidOperationException($"{path} has SHA-256 {actual}, expected {sha256}: the sample changed upstream.");
        return data;
    }
}
