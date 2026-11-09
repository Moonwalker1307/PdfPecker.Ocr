using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Models;
using PdfPecker.Ocr.Recognition;

namespace PdfPecker.Ocr.Layout;

/// <summary>
/// What one page needs besides its pixels: an inference context with its threads, buffers and plans, the three
/// networks running in it, and a pipeline whose stages keep their own buffers. Building one costs threads and tens of
/// megabytes of buffers that the first page then grows, so workers are pooled (<see cref="EnginePool"/>) and used by one
/// call at a time.
/// </summary>
internal sealed class EngineWorker : IDisposable
{
    private readonly InferenceContext _context;
    private readonly ModelNetwork _detection;
    private readonly ModelNetwork _detectionGray;
    private readonly ModelNetwork _recognition;
    private readonly ModelNetwork _orientation;
    private OcrPipeline? _pipeline;

    public EngineWorker(int threads)
    {
        Threads = threads;
        _context = new InferenceContext(threads);
        try
        {
            _detection = new ModelNetwork(ModelStore.Detection, _context);
            _detectionGray = new ModelNetwork(ModelStore.DetectionGray, _context);
            _recognition = new ModelNetwork(ModelStore.RecognitionArgMax, _context);
            _orientation = new ModelNetwork(ModelStore.Classifier, _context);
        }
        catch
        {
            _context.Dispose();
            throw;
        }
    }

    /// <summary>The context's degree of parallelism, which the pool matches workers by.</summary>
    public int Threads { get; }

    /// <summary>When the worker was last returned to the pool, in <see cref="Environment.TickCount64"/> milliseconds.</summary>
    public long ReturnedAt { get; set; }

    /// <summary>The bytes the context's activation arena holds, for memory measurements.</summary>
    public long ArenaBytes => _context.ArenaBytes;

    /// <summary>
    /// Reads a page. The pipeline is kept between calls and rebuilt only when the recognition settings change, which
    /// keeps the detector's and recogniser's buffers for the next page.
    /// </summary>
    public PageResult Run(PixelView page, float dpi, RecognitionOptions options, CancellationToken cancellationToken)
    {
        if (_pipeline is null || _pipeline.Recognizer.Options != options)
        {
            _pipeline = new OcrPipeline(_detection, _recognition, _orientation, OcrEngine.Characters, OcrEngine.DetectionOptions, options, _detectionGray, _context);
        }

        _context.CancellationToken = cancellationToken;
        try
        {
            return _pipeline.Run(page, dpi, cancellationToken);
        }
        finally
        {
            _context.CancellationToken = default;
        }
    }

    /// <summary>Stops the context's threads and lets its buffers go.</summary>
    public void Dispose()
    {
        _pipeline = null;
        _context.Dispose();
    }
}

/// <summary>
/// The idle workers, shared by every call, so that a stream of pages and calls reuses threads, buffers and plans and
/// allocates little once warm. A call takes a worker with the number of threads it wants, or builds one, and gives it
/// back afterwards. The pool keeps at most as many idle workers as there are processors (more cannot run at once
/// without taking turns), and a worker that has been idle for <see cref="IdleTimeout"/> is disposed, so an application
/// that reads a document now and then does not keep the buffers of its largest page for ever.
/// </summary>
internal static class EnginePool
{
    private static readonly object s_lock = new();
    private static readonly List<EngineWorker> s_idle = [];
    private static Timer? s_timer;
    private static bool s_timerArmed;

    /// <summary>How long an idle worker is kept. Tests shorten it.</summary>
    internal static TimeSpan IdleTimeout { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>The most idle workers kept.</summary>
    internal static int MaxIdle => Math.Max(2, Environment.ProcessorCount);

    /// <summary>The number of idle workers, for tests.</summary>
    internal static int IdleCount
    {
        get
        {
            lock (s_lock)
            {
                return s_idle.Count;
            }
        }
    }

    /// <summary>A worker with exactly <paramref name="threads"/> threads: the one used most recently, or a new one.</summary>
    public static EngineWorker Rent(int threads)
    {
        lock (s_lock)
        {
            for (int i = s_idle.Count - 1; i >= 0; i--)
            {
                if (s_idle[i].Threads == threads)
                {
                    EngineWorker worker = s_idle[i];
                    s_idle.RemoveAt(i);
                    return worker;
                }
            }
        }

        return new EngineWorker(threads);
    }

    /// <summary>Gives a worker back for the next call; the oldest idle worker is disposed when the pool is full.</summary>
    public static void Return(EngineWorker worker)
    {
        EngineWorker? evicted = null;
        lock (s_lock)
        {
            worker.ReturnedAt = Environment.TickCount64;
            s_idle.Add(worker);
            if (s_idle.Count > MaxIdle)
            {
                evicted = s_idle[0];
                s_idle.RemoveAt(0);
            }

            if (!s_timerArmed)
            {
                Arm(IdleTimeout);
            }
        }

        evicted?.Dispose();
    }

    /// <summary>Disposes every idle worker, so the next call starts cold; for tests and the benchmarks' first-call figures.</summary>
    internal static void Clear()
    {
        EngineWorker[] workers;
        lock (s_lock)
        {
            workers = [.. s_idle];
            s_idle.Clear();
        }

        foreach (EngineWorker worker in workers)
        {
            worker.Dispose();
        }
    }

    /// <summary>Disposes the workers that have been idle for the timeout, and waits for the next one to expire.</summary>
    private static void Trim()
    {
        var expired = new List<EngineWorker>();
        lock (s_lock)
        {
            s_timerArmed = false;
            long now = Environment.TickCount64;
            long timeout = (long)IdleTimeout.TotalMilliseconds;
            long next = long.MaxValue;
            for (int i = s_idle.Count - 1; i >= 0; i--)
            {
                long idle = now - s_idle[i].ReturnedAt;
                if (idle >= timeout)
                {
                    expired.Add(s_idle[i]);
                    s_idle.RemoveAt(i);
                }
                else
                {
                    next = Math.Min(next, timeout - idle);
                }
            }

            if (next != long.MaxValue)
            {
                Arm(TimeSpan.FromMilliseconds(next));
            }
        }

        foreach (EngineWorker worker in expired)
        {
            worker.Dispose();
        }
    }

    private static void Arm(TimeSpan due)
    {
        if (s_timer is null)
        {
            // The timer must not capture the execution context (and its async locals) of whichever call created it.
            using (ExecutionContext.SuppressFlow())
            {
                s_timer = new Timer(static _ => Trim(), null, Timeout.Infinite, Timeout.Infinite);
            }
        }

        s_timer.Change(due, Timeout.InfiniteTimeSpan);
        s_timerArmed = true;
    }
}
