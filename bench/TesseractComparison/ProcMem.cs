namespace TesseractComparison;

/// <summary>Resident memory of this process from /proc (Linux), including the peak, which can be reset between pages.</summary>
static class ProcMem
{
    /// <summary>
    /// Resets the kernel's peak resident set counter (VmHWM) by writing 5 to /proc/self/clear_refs, so that the next
    /// <see cref="PeakKb"/> is the peak since now. Returns false where the kernel does not allow it.
    /// </summary>
    public static bool ResetPeak()
    {
        try { File.WriteAllText("/proc/self/clear_refs", "5"); return true; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return false; }
    }

    public static long PeakKb() => Field("VmHWM:");

    public static long RssKb() => Field("VmRSS:");

    static long Field(string name)
    {
        try
        {
            foreach (var line in File.ReadLines("/proc/self/status"))
                if (line.StartsWith(name, StringComparison.Ordinal))
                    return long.Parse(line.AsSpan(name.Length).Trim().ToString().Split(' ')[0], System.Globalization.CultureInfo.InvariantCulture);
        }
        catch (IOException) { }
        return 0;
    }
}
