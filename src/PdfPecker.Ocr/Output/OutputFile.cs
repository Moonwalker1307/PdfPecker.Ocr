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
        // The temporary name does not embed the target's, which would add to it and could pass the 255-character limit
        // on a name that is itself valid.
        var temp = Path.Combine(Path.GetDirectoryName(full) ?? ".", "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            File.WriteAllBytes(temp, bytes);
            File.Move(temp, full, overwrite: true);
        }
        catch
        {
            try
            {
                File.Delete(temp);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // The original failure is the one the caller sees.
            }

            throw;
        }
    }
}
