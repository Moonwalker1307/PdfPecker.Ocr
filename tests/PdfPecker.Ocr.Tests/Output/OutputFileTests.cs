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
    public void LongFileNamesCanBeWritten()
    {
        // A name of 240 characters is valid on ext4, APFS and NTFS (255), but not with a long temporary suffix. On
        // Windows the whole path would pass the 260-character limit, so the case is a Unix one.
        if (OperatingSystem.IsWindows()) return;
        var dir = Directory.CreateTempSubdirectory("ocr-output-");
        try
        {
            var path = Path.Combine(dir.FullName, new string('n', 236) + ".pdf");
            OutputFile.WriteAllBytes(path, new byte[] { 1, 2, 3 });
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(path));
            Assert.Single(Directory.GetFiles(dir.FullName));
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
