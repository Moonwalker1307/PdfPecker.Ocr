using System.Text;
using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference.Kernels;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Tests.Inference;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>
/// Whole pages read with every matrix kernel the processor runs and with the portable kernel, which must give the same
/// text, quads and confidences to the bit (see <see cref="KernelAgreementTests"/>): on Arm64 the NEON kernel against the
/// portable one it replaces, on x64 the AVX kernels.
/// </summary>
[Collection("EndToEnd")]
public sealed class KernelPageTests
{
    [Theory]
    [MemberData(nameof(KernelTests.Kernels), MemberType = typeof(KernelTests))]
    public void PagesReadTheSameWithEveryKernel(string kernel)
    {
        var gemm = GemmKernel.Supported.Single(k => k.Name == kernel);
        var pages = new (string Name, byte[] Pixels, int Width, int Height, int Channels)[]
        {
            ("invoice gray", TestPages.Invoice.Gray(), TestPages.Invoice.Width, TestPages.Invoice.Height, 1),
            ("latin", TestPages.Latin.Rgba, TestPages.Latin.Width, TestPages.Latin.Height, 4),
            ("numbers upside down", TestPages.Numbers.UpsideDown(), TestPages.Numbers.Width, TestPages.Numbers.Height, 4),
            ("table", TestPages.Table.Rgba, TestPages.Table.Width, TestPages.Table.Height, 4),
            ("small", TestPages.Small.Rgba, TestPages.Small.Width, TestPages.Small.Height, 4),
        };
        var options = OcrEngine.RecognitionOptions(new OcrOptions());
        using var portable = new EngineWorker(2, ModelStore.LoadWith(GemmKernel.Portable));
        using var other = new EngineWorker(3, ModelStore.LoadWith(gemm));
        foreach (var (name, pixels, width, height, channels) in pages)
        {
            var view = new PixelView(pixels, width, height, channels);
            string expected = Describe(OcrEngine.ToPage(portable.Run(view, 300, options, null, default), 0));
            string actual = Describe(OcrEngine.ToPage(other.Run(view, 300, options, null, default), 0));
            Assert.True(expected.Length > 100, $"{name}: {expected}");
            Assert.Equal(expected, actual);
        }
    }

    /// <summary>The page's text, confidences and the corners of every line and word, the numbers as their bits.</summary>
    private static string Describe(OcrPage page)
    {
        var text = new StringBuilder();
        void Bits(float value) => text.Append(BitConverter.SingleToUInt32Bits(value).ToString("x8", System.Globalization.CultureInfo.InvariantCulture)).Append(' ');
        void Quad(OcrQuad q)
        {
            foreach (var p in new[] { q.TopLeft, q.TopRight, q.BottomRight, q.BottomLeft })
            {
                Bits(p.X);
                Bits(p.Y);
            }
        }

        Bits(page.Confidence);
        text.Append('\n');
        foreach (OcrLine line in page.Lines)
        {
            text.Append(line.Text).Append(" | ");
            Bits(line.Confidence);
            Quad(line.Quad);
            text.Append('\n');
            foreach (OcrWord word in line.Words)
            {
                text.Append("  ").Append(word.Text).Append(" | ");
                Bits(word.Confidence);
                Quad(word.Quad);
                text.Append('\n');
            }
        }

        return text.ToString();
    }
}
