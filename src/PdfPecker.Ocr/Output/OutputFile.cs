namespace PdfPecker.Ocr.Output;

/// <summary>Writes files so a reader never sees half of one.</summary>
internal static class OutputFile
{
    /// <summary>
    /// Writes the bytes to a temporary file next to <paramref name="path"/> and moves it into place, so the file at
    /// <paramref name="path"/> is either the old one or the complete new one.
    /// </summary>
    public static void WriteAllBytes(string path, byte[] bytes)
    {
        var full = Path.GetFullPath(path);
        var temp = Path.Combine(Path.GetDirectoryName(full) ?? ".", "." + Path.GetFileName(full) + "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            File.WriteAllBytes(temp, bytes);
            File.Move(temp, full, overwrite: true);
        }
        catch
        {
            File.Delete(temp);
            throw;
        }
    }
}
