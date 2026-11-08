using System.Diagnostics;
using System.Reflection;
using System.Text.RegularExpressions;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>What went wrong with one input: a kind (exception, timeout, memory), a dedupe signature and the details.</summary>
internal sealed record Verdict(string Kind, string Signature, string Detail);

/// <summary>
/// Runs one input on a worker thread under a wall-clock timeout and a heap ceiling, and classifies the outcome. The
/// token handed to the target is cancelled first; a thread that still does not stop is abandoned and counted, because a
/// .NET thread cannot be killed.
/// </summary>
internal sealed class Executor
{
    private static readonly TimeSpan Grace = TimeSpan.FromSeconds(3);

    private readonly TimeSpan _timeout;
    private readonly long _memoryCeiling;

    // The heap that stays live between inputs: the seeds, the corpus and whatever the library keeps in static caches.
    // The ceiling applies to what an input adds on top of it, because the heap is shared by the whole process and an
    // input should not be blamed for what earlier ones left behind.
    private long _retained;

    /// <summary>Threads that ignored cancellation and are still running; the caller should restart the process.</summary>
    public int AbandonedThreads { get; private set; }

    public Executor(TimeSpan timeout, long memoryCeiling)
    {
        _timeout = timeout;
        _memoryCeiling = memoryCeiling;
        _retained = GC.GetTotalMemory(true);
    }

    /// <summary>The live heap measured at the last <see cref="MeasureRetained"/>, or when the executor was made.</summary>
    public long Retained => _retained;

    /// <summary>
    /// Collects garbage and records the live heap as the baseline for the next inputs. Call it between inputs, before
    /// confirming a memory finding, so that heap kept by earlier inputs is not counted against the one under test.
    /// </summary>
    public long MeasureRetained() => _retained = GC.GetTotalMemory(true);

    /// <param name="tolerated">The type name of an exception the target tolerates, for the statistics.</param>
    public Verdict? Run(Target target, byte[] input, out string? tolerated)
    {
        tolerated = null;
        using var cts = new CancellationTokenSource();
        Exception? error = null;
        using var done = new ManualResetEventSlim();
        var thread = new Thread(() =>
        {
            try { target.Run(input, cts.Token); }
            catch (Exception ex) { error = ex; }
            finally { done.Set(); }
        }) { IsBackground = true, Name = "fuzz-" + target.Name };

        var clock = Stopwatch.StartNew();
        thread.Start();
        string? abort = null;
        TimeSpan abortedAt = default;
        while (!done.Wait(20))
        {
            if (abort == null)
            {
                if (clock.Elapsed > _timeout) abort = "timeout";
                else if (GC.GetTotalMemory(false) - _retained > _memoryCeiling && GC.GetTotalMemory(true) - _retained > _memoryCeiling) abort = "memory";
                if (abort == null) continue;
                abortedAt = clock.Elapsed;
                cts.Cancel();
            }
            else if (clock.Elapsed - abortedAt > Grace)
            {
                AbandonedThreads++;
                var what = abort == "timeout"
                    ? $"The input ran for more than {_timeout.TotalSeconds:0.#} s and the target did not react to cancellation."
                    : $"The heap grew by more than {_memoryCeiling >> 20} MB and the target did not react to cancellation.";
                return new Verdict(abort, abort + "|uncancellable", what);
            }
        }

        var ex = Unwrap(error);
        if (abort != null)
        {
            var stack = ex?.StackTrace;
            string detail = abort == "timeout"
                ? $"The input ran for more than {_timeout.TotalSeconds:0.#} s; cancellation stopped it."
                : $"The heap grew by more than {_memoryCeiling >> 20} MB; cancellation stopped it.";
            if (ex != null && stack != null) detail += "\nStopped at (the cancellation point, not necessarily the slow loop):\n" + ex.GetType().FullName + "\n" + stack;
            return new Verdict(abort, abort + "|" + TopFrames(stack), detail);
        }
        if (ex == null) return null;
        if (ex is OutOfMemoryException or InsufficientMemoryException)
            return new Verdict("memory", "memory|" + TopFrames(ex.StackTrace), "The heap hit its hard limit.\n" + Describe(ex));
        if (target.IsExpected(ex))
        {
            tolerated = ex.GetType().Name;
            return null;
        }
        return new Verdict("exception", ex.GetType().FullName + "|" + TopFrames(ex.StackTrace), Describe(ex));
    }

    private static Exception? Unwrap(Exception? ex)
    {
        while (ex is AggregateException { InnerExceptions.Count: 1 } or TargetInvocationException)
            ex = ex.InnerException!;
        return ex;
    }

    private static string Describe(Exception ex)
    {
        var text = ex.GetType().FullName + ": " + ex.Message + "\n" + ex.StackTrace;
        for (var inner = ex.InnerException; inner != null; inner = inner.InnerException)
            text += "\n--- inner: " + inner.GetType().FullName + ": " + inner.Message + "\n" + inner.StackTrace;
        return text;
    }

    /// <summary>The top frame of PdfPecker.Ocr or PdfPecker, without file names and line numbers, so one bug has one signature.</summary>
    public static string TopFrames(string? stack)
    {
        if (string.IsNullOrEmpty(stack)) return "no-stack";
        var frames = stack.Split('\n')
            .Select(l => Regex.Replace(l.Trim(), @"\s+in\s+.*$", ""))
            .Where(l => l.StartsWith("at PdfPecker.", StringComparison.Ordinal) && !l.StartsWith("at PdfPecker.Ocr.Fuzz.", StringComparison.Ordinal))
            .Select(l => Regex.Replace(l, @"\(.*\)", "()"))
            .Take(1)
            .ToList();
        return frames.Count == 0 ? "no-library-frame" : string.Join(" < ", frames);
    }
}
