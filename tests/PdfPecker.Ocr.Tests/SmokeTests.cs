namespace PdfPecker.Ocr.Tests;

public sealed class SmokeTests
{
    [Fact]
    public void VersionIsSet() => Assert.False(string.IsNullOrEmpty(OcrInfo.Version));
}
