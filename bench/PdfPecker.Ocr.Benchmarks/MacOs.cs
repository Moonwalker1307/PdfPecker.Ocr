using System.Runtime.InteropServices;
using System.Text;

namespace OcrBench;

/// <summary>
/// What the harness reads from /proc on Linux, read from the kernel's own calls on macOS, which has no /proc and whose
/// Process.PeakWorkingSet64 is always 0. macOS keeps the resident peak for the life of the process and offers no way to
/// reset it, so every peak read here is the running peak.
/// </summary>
static unsafe class MacOs
{
    /// <summary>The process's resident peak in bytes (ru_maxrss, which macOS counts in bytes and Linux in KiB).</summary>
    public static long PeakResidentBytes()
    {
        // struct rusage on 64-bit macOS: two struct timevals of 16 bytes each, then ru_maxrss and thirteen more longs.
        long* usage = stackalloc long[18];
        return getrusage(0 /* RUSAGE_SELF */, usage) == 0 ? usage[4] : 0;
    }

    /// <summary>The processor's name, such as "Apple M4 Pro", or null when sysctl does not know it.</summary>
    public static string? CpuBrand()
    {
        byte* buffer = stackalloc byte[256];
        nint length = 256;
        fixed (byte* name = "machdep.cpu.brand_string\0"u8)
            if (sysctlbyname(name, buffer, &length, 0, 0) != 0 || length <= 1)
                return null;
        return Encoding.UTF8.GetString(buffer, (int)length - 1).Trim();
    }

    [DllImport("libc")]
    static extern int getrusage(int who, long* usage);

    [DllImport("libc")]
    static extern int sysctlbyname(byte* name, byte* oldp, nint* oldlenp, nint newp, nint newlen);
}
