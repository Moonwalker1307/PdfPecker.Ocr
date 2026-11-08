using PdfPecker.Ocr.Input;

namespace PdfPecker.Ocr.Tests.Input;

public sealed class PagePixelsTests
{
    private static byte Reference(byte value, byte alpha) => (byte)((value * alpha + 255 * (255 - alpha) + 127) / 255);

    [Theory]
    [InlineData(1, false)]
    [InlineData(3, true)]
    [InlineData(4, false)]
    [InlineData(17, true)]
    [InlineData(63, false)]
    [InlineData(1000, true)]
    [InlineData(1001, false)]
    public void ColourConversionMatchesTheScalarDefinition(int pixels, bool someTransparent)
    {
        var rnd = new Random(pixels);
        var rgba = new byte[pixels * 4];
        rnd.NextBytes(rgba);
        for (int p = 0; p < pixels; p++)
            if (!someTransparent || rnd.Next(4) != 0) rgba[p * 4 + 3] = 255;
        using var result = PagePixels.FromRgba(rgba, pixels, 1);
        Assert.Equal(PixelFormat.Rgb24, result.Format);
        var expected = new byte[pixels * 3];
        for (int p = 0; p < pixels; p++)
            for (int c = 0; c < 3; c++) expected[p * 3 + c] = Reference(rgba[p * 4 + c], rgba[p * 4 + 3]);
        Assert.Equal(expected, result.Pixels.ToArray());
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(15, true)]
    [InlineData(16, false)]
    [InlineData(33, true)]
    [InlineData(4097, false)]
    public void GrayPagesAreStoredAsGray(int pixels, bool someTransparent)
    {
        var rnd = new Random(pixels);
        var rgba = new byte[pixels * 4];
        for (int p = 0; p < pixels; p++)
        {
            byte v = (byte)rnd.Next(256);
            rgba[p * 4] = rgba[p * 4 + 1] = rgba[p * 4 + 2] = v;
            rgba[p * 4 + 3] = someTransparent && rnd.Next(3) == 0 ? (byte)rnd.Next(256) : (byte)255;
        }
        using var result = PagePixels.FromRgba(rgba, pixels, 1);
        Assert.Equal(PixelFormat.Gray8, result.Format);
        Assert.Equal(1, result.Channels);
        Assert.Equal(Enumerable.Range(0, pixels).Select(p => Reference(rgba[p * 4], rgba[p * 4 + 3])), result.Pixels.ToArray());

        // One coloured pixel anywhere, even the last, makes the page RGB.
        rgba[^2] ^= 1;
        using var colour = PagePixels.FromRgba(rgba, pixels, 1);
        Assert.Equal(PixelFormat.Rgb24, colour.Format);
    }

    [Fact]
    public void RowsAndSizesAndDispose()
    {
        var gray = Enumerable.Range(0, 12).Select(i => (byte)i).ToArray();
        var pixels = PagePixels.FromGray(gray, 4, 3);
        Assert.Equal((4, 3, 4, 12), (pixels.Width, pixels.Height, pixels.Stride, pixels.Length));
        Assert.Equal(new byte[] { 8, 9, 10, 11 }, pixels.Row(2).ToArray());
        using (var rgb = PagePixels.FromRgb(new byte[] { 1, 2, 3, 4, 5, 6, 7 }, 2, 1))
            Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6 }, rgb.Pixels.ToArray());
        pixels.Dispose();
        pixels.Dispose();
        Assert.Throws<ObjectDisposedException>(() => pixels.Pixels.Length);
    }

    [Fact]
    public void BadSizesFailWithArgumentExceptions()
    {
        Assert.Throws<ArgumentException>(() => PagePixels.FromRgba(new byte[15], 2, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => PagePixels.FromGray(new byte[4], 0, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => PagePixels.FromGray(new byte[4], 4, -1));
        Assert.Throws<PdfLimitException>(() => PagePixels.FromRgba(new byte[4], 50_000, 50_000));
        Assert.Throws<PdfLimitException>(() => PagePixels.Rent(int.MaxValue, 2, PixelFormat.Gray8));
    }
}
