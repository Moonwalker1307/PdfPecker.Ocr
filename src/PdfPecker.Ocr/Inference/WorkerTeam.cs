using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// The threads of one <see cref="InferenceContext"/>: the calling thread plus up to n - 1 threads of its own, which
/// stay with the context and wait for work between the operators of a run.
/// </summary>
/// <remarks>
/// Own threads rather than the thread pool for two reasons. Worker w is always the same thread and starts on the same
/// share of each operator's items, so the activations it wrote for one layer are still in its core's cache when it
/// packs them for the next; with the thread pool any thread took any item, and the recogniser ran 1.8 times slower at
/// four threads than with hot caches. And a run of about eighty operators should not allocate or wait for a pool
/// thread to wake each time. Items are still taken one at a time from the shares, and a worker that finishes its own
/// share takes items from the others', so a thread that the operating system delays does not hold up the rest.
/// </remarks>
internal sealed unsafe class WorkerTeam : IDisposable
{
    // How long a worker keeps polling for the next operator before it sleeps; operators of a run follow each other
    // within microseconds, runs usually do not.
    private static readonly long SpinTicks = Stopwatch.Frequency / 20000;

    private readonly int _size;
    private readonly Cursor[] _cursors;
    private readonly int[] _sleeping;
    private readonly SemaphoreSlim[] _signals;
    private Thread[]? _threads;

    // The published job: the generation in the high 32 bits and the number of workers taking part in the low 32, read
    // and written as one value so a worker never sees one job's generation with another's worker count.
    private long _published;
    private int _pending;
    private volatile bool _disposed;
    private object? _state;
    private delegate*<object, int, int, void> _body;
    private Exception? _failure;

    public WorkerTeam(int size)
    {
        _size = size;
        _cursors = new Cursor[size];
        _sleeping = new int[size];
        _signals = new SemaphoreSlim[size];
        for (int i = 1; i < size; i++) _signals[i] = new SemaphoreSlim(0);
    }

    /// <summary>Runs body(state, i, worker) once for each i in [0, count) on min(size, count) workers.</summary>
    public void For(int count, object state, delegate*<object, int, int, void> body)
    {
        int workers = Math.Min(_size, count);
        _threads ??= StartThreads();
        _state = state;
        _body = body;
        _failure = null;
        for (int w = 0; w < workers; w++)
        {
            _cursors[w].Next = (int)((long)count * w / workers);
            _cursors[w].End = (int)((long)count * (w + 1) / workers);
        }
        Volatile.Write(ref _pending, workers - 1);
        long generation = (Volatile.Read(ref _published) >> 32) + 1;
        Interlocked.Exchange(ref _published, (generation << 32) | (uint)workers);
        for (int w = 1; w < workers; w++)
            if (Volatile.Read(ref _sleeping[w]) != 0) _signals[w].Release();
        Run(0, workers);
        var spin = default(SpinWait);
        while (Volatile.Read(ref _pending) != 0) spin.SpinOnce(sleep1Threshold: -1);
        _state = null;
        if (_failure is { } failure) ExceptionDispatchInfo.Throw(failure);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Interlocked.Add(ref _published, 1L << 32);
        for (int w = 1; w < _size; w++) _signals[w].Release();
    }

    private Thread[] StartThreads()
    {
        var threads = new Thread[_size - 1];
        for (int w = 1; w < _size; w++)
        {
            // Background threads: a context that is never disposed must not keep the process alive.
            var thread = new Thread(Worker, maxStackSize: 1 << 20) { IsBackground = true, Name = "PdfPecker.Ocr worker" };
            thread.Start(w);
            threads[w - 1] = thread;
        }
        return threads;
    }

    private void Worker(object? argument)
    {
        int index = (int)argument!;
        // The threads start inside the first For, before it publishes generation 1, so generation 0 is the one seen.
        long seen = 0;
        while (true)
        {
            long published = Wait(index, seen);
            if (_disposed) return;
            seen = published >> 32;
            int workers = (int)(uint)published;
            if (index < workers)
            {
                Run(index, workers);
                Interlocked.Decrement(ref _pending);
            }
        }
    }

    /// <summary>Waits for a job newer than the given generation: polls for a while, then sleeps until woken.</summary>
    private long Wait(int index, long seen)
    {
        long start = Stopwatch.GetTimestamp();
        var spin = default(SpinWait);
        while (true)
        {
            long published = Volatile.Read(ref _published);
            if (published >> 32 != seen || _disposed) return published;
            if (Stopwatch.GetTimestamp() - start < SpinTicks)
            {
                spin.SpinOnce(sleep1Threshold: -1);
                continue;
            }
            // Announce the sleep, then look once more: the caller publishes before it reads the flag, so either this
            // check sees the new job or the caller sees the flag and wakes the worker. A leftover wake-up only makes a
            // later wait return early, and the loop checks again.
            Interlocked.Exchange(ref _sleeping[index], 1);
            if (Volatile.Read(ref _published) >> 32 == seen && !_disposed) _signals[index].Wait();
            Volatile.Write(ref _sleeping[index], 0);
            start = Stopwatch.GetTimestamp();
        }
    }

    /// <summary>Takes items from the worker's own share first, then from the others' in turn.</summary>
    private void Run(int worker, int workers)
    {
        var body = _body;
        object state = _state!;
        try
        {
            for (int k = 0; k < workers; k++)
            {
                ref var cursor = ref _cursors[(worker + k) % workers];
                int i;
                while ((i = Interlocked.Increment(ref cursor.Next) - 1) < cursor.End) body(state, i, worker);
            }
        }
        catch (Exception e)
        {
            Interlocked.CompareExchange(ref _failure, e, null);
        }
    }

    /// <summary>The next item and the end of one worker's share, on a cache line of its own.</summary>
    [StructLayout(LayoutKind.Explicit, Size = 128)]
    private struct Cursor
    {
        [FieldOffset(64)]
        public int Next;

        [FieldOffset(68)]
        public int End;
    }
}
