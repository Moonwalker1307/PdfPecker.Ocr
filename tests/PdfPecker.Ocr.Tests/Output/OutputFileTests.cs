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
    public void ReplacedFilesKeepTheirModeAndLinks()
    {
        // Writing a private file in place must not make it readable by others, and a symbolic link must stay a link to
        // the file it names, which gets the new bytes.
        if (OperatingSystem.IsWindows()) return;
        var dir = Directory.CreateTempSubdirectory("ocr-output-");
        try
        {
            var path = Path.Combine(dir.FullName, "secret.pdf");
            File.WriteAllBytes(path, new byte[] { 1 });
            File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite);
            OutputFile.WriteAllBytes(path, new byte[] { 2 });
            Assert.Equal(new byte[] { 2 }, File.ReadAllBytes(path));
            Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(path));

            var link = Path.Combine(dir.FullName, "link.pdf");
            File.CreateSymbolicLink(link, path);
            OutputFile.WriteAllBytes(link, new byte[] { 3 });
            Assert.NotNull(new FileInfo(link).LinkTarget);
            Assert.Equal(new byte[] { 3 }, File.ReadAllBytes(path));
            Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(path));
            Assert.Equal(2, Directory.GetFiles(dir.FullName).Length);
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
