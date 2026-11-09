namespace PdfPecker.Ocr.Output;

/// <summary>Writes files so a reader never sees half of one.</summary>
internal static class OutputFile
{
    /// <summary>
    /// Writes the bytes to a temporary file next to <paramref name="path"/> and moves it into place, so the file at
    /// <paramref name="path"/> is either the old one or the complete new one.
    /// </summary>
    /// <remarks>
    /// A symbolic link at <paramref name="path"/> is followed, so the file it points at is replaced and the link stays.
    /// On Unix the new file gets the replaced file's permissions, so writing a private file in place does not make it
    /// readable by others; it is created with them, never briefly wider. Its owner, ACLs and hard links are not kept,
    /// since the file is a new one.
    /// </remarks>
    public static void WriteAllBytes(string path, byte[] bytes)
    {
        var full = Path.GetFullPath(path);
        var info = new FileInfo(full);
        if (info.LinkTarget != null) full = info.ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? full;
        UnixFileMode? mode = !OperatingSystem.IsWindows() && File.Exists(full) ? File.GetUnixFileMode(full) : null;
        // The temporary name does not embed the target's, which would add to it and could pass the 255-character limit
        // on a name that is itself valid.
        var temp = Path.Combine(Path.GetDirectoryName(full) ?? ".", "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write, Share = FileShare.None };
            if (mode is { } create && !OperatingSystem.IsWindows()) options.UnixCreateMode = create;
            using (var stream = new FileStream(temp, options))
            {
                stream.Write(bytes);
            }

            // The umask may have narrowed the mode the file was created with.
            if (mode is { } keep && !OperatingSystem.IsWindows()) File.SetUnixFileMode(temp, keep);
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
