using System.Text;

namespace PdfPecker.Ocr.Tests;

/// <summary>
/// The checked-in baselines in <c>Golden/</c>, read from the source tree rather than the build output, so that
/// <c>PDFPECKER_OCR_UPDATE_GOLDEN=1</c> rewrites the files that are committed.
/// </summary>
internal static class GoldenFiles
{
    public const string UpdateVariable = "PDFPECKER_OCR_UPDATE_GOLDEN";

    private static readonly Lazy<string> Folder = new(FindFolder);

    public static string Directory => Folder.Value;

    public static bool Updating => Environment.GetEnvironmentVariable(UpdateVariable) == "1";

    public static string Path(string fileName) => System.IO.Path.Combine(Directory, fileName);

    /// <summary>The names of the golden pages: every <c>Golden/*.html</c> without its extension, in ordinal order.</summary>
    public static IEnumerable<string> PageNames()
        => System.IO.Directory.EnumerateFiles(Directory, "*.html")
            .Select(p => System.IO.Path.GetFileNameWithoutExtension(p))
            .Order(StringComparer.Ordinal);

    public static string Html(string name) => File.ReadAllText(Path(name + ".html"), Encoding.UTF8);

    /// <summary>
    /// Compares <paramref name="actual"/> with the baseline <paramref name="fileName"/>, or writes it when updating. A
    /// mismatch names the first line that differs, since the files are long and xUnit cuts long strings.
    /// </summary>
    public static void Check(string fileName, string actual)
    {
        string path = Path(fileName);
        if (Updating)
        {
            File.WriteAllText(path, actual, new UTF8Encoding(false));
            return;
        }

        Assert.True(File.Exists(path), $"{fileName} is missing; run the golden tests with {UpdateVariable}=1 to write it.");
        // A checkout with CRLF line ends must not fail every golden, although .gitattributes asks for LF.
        string expected = File.ReadAllText(path, Encoding.UTF8).Replace("\r\n", "\n", StringComparison.Ordinal);
        if (string.Equals(expected, actual, StringComparison.Ordinal)) return;

        string[] e = expected.Split('\n'), a = actual.Split('\n');
        int line = 0;
        while (line < e.Length && line < a.Length && string.Equals(e[line], a[line], StringComparison.Ordinal)) line++;
        string Show(string[] lines) => line < lines.Length ? lines[line] : "(end of file)";
        Assert.Fail($"{fileName} differs from the output at line {line + 1}:\n  expected: {Show(e)}\n  actual:   {Show(a)}\n" +
            $"If the change is intended, run the golden tests with {UpdateVariable}=1 and say in the commit which baselines changed and why.");
    }

    private static string FindFolder()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            if (File.Exists(System.IO.Path.Combine(dir.FullName, "PdfPecker.Ocr.slnx")))
                return System.IO.Path.Combine(dir.FullName, "tests", "PdfPecker.Ocr.Tests", "Golden");
        }

        throw new InvalidOperationException("Cannot find the repository root (PdfPecker.Ocr.slnx) above the test output.");
    }
}
