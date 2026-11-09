using PdfPecker.Ocr.Output;

namespace PdfPecker.Ocr.Tests.Output;

public sealed class OutputFileTests
{
    [Fact]
    public void FilesAreReplacedWhole()
    {
        var dir = Directory.CreateTempSubdirectory("ocr-output-");
        try
        {
            var path = Path.Combine(dir.FullName, "out.pdf");
            OutputFile.WriteAllBytes(path, new byte[] { 1, 2, 3 });
            OutputFile.WriteAllBytes(path, new byte[] { 4, 5 });
            Assert.Equal(new byte[] { 4, 5 }, File.ReadAllBytes(path));
            Assert.Single(Directory.GetFiles(dir.FullName));
            Assert.ThrowsAny<IOException>(() => OutputFile.WriteAllBytes(Path.Combine(dir.FullName, "missing", "x.pdf"), new byte[1]));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void StreamsAreReadFromTheirPosition()
    {
        var ms = new MemoryStream(new byte[] { 1, 2, 3, 4 }) { Position = 1 };
        Assert.Equal(new byte[] { 2, 3, 4 }, PdfOcr.ReadAll(ms, default));
        Assert.Equal(4, ms.Position);
        using var buffered = new BufferedStream(new MemoryStream(new byte[] { 9, 8 }));
        Assert.Equal(new byte[] { 9, 8 }, PdfOcr.ReadAll(buffered, default));
    }
}
