using System.Runtime.CompilerServices;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// The working memory and the degree of parallelism of one inference worker. A context runs one model at a time: it
/// owns the activation arena, per-worker scratch buffers, the execution plans and the output buffers, and reuses them
/// across runs, so a steady stream of runs of similar sizes allocates no tensors. It is not thread-safe; give each
/// concurrent worker its own context. Models are shared.
/// </summary>
internal sealed unsafe class InferenceContext : IDisposable
{
    private readonly ParallelOptions _options;
    private readonly Action<int> _worker;
    private readonly Dictionary<OcrModel, ExecutionPlan> _plans = [];
    private readonly Dictionary<OcrModel, float[]> _outputs = [];
    private float[]? _arena;
    private float[]?[] _scratch;
    private float[]? _shared;
    private bool _disposed;

    // The job that the parallel workers run; set by For for the duration of one parallel loop.
    private object? _jobState;
    private delegate*<object, int, int, void> _jobBody;
    private int _jobCount;
    private int _jobNext;

    /// <summary>Creates a context that uses at most the given number of threads for one run.</summary>
    public InferenceContext(int maxDegreeOfParallelism)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxDegreeOfParallelism, 1);
        MaxDegreeOfParallelism = maxDegreeOfParallelism;
        _options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
        _worker = WorkerLoop;
        _scratch = new float[maxDegreeOfParallelism][];
    }

    /// <summary>The largest number of threads one run uses.</summary>
    public int MaxDegreeOfParallelism { get; }

    /// <summary>Checked between the operators of a run, so a long run can be stopped.</summary>
    public CancellationToken CancellationToken { get; set; }

    /// <summary>The size of the activation arena, in bytes, for memory measurements.</summary>
    public long ArenaBytes => (_arena?.LongLength ?? 0) * sizeof(float);

    /// <summary>Releases the buffers. The context cannot run models afterwards.</summary>
    public void Dispose()
    {
        _disposed = true;
        _arena = null;
        _shared = null;
        _scratch = [];
        _plans.Clear();
        _outputs.Clear();
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>The plan of a model for an input shape, reusing the previous one when the shape is unchanged.</summary>
    internal ExecutionPlan GetPlan(OcrModel model, int[] inputShape)
    {
        if (_plans.TryGetValue(model, out var plan) && plan.InputShape.AsSpan().SequenceEqual(inputShape)) return plan;
        plan = ExecutionPlan.Create(model, inputShape, this);
        _plans[model] = plan;
        return plan;
    }

    /// <summary>The arena for a plan, grown when needed; its contents are not preserved across growth.</summary>
    internal float* Arena(long floats) => Pointer(ref _arena, floats);

    /// <summary>The output buffer of a model, owned by this context and overwritten by its next run of the model.</summary>
    internal float[] OutputBuffer(OcrModel model, int floats)
    {
        if (!_outputs.TryGetValue(model, out var buffer) || buffer.Length < floats)
        {
            _outputs.Remove(model);
            buffer = GC.AllocateUninitializedArray<float>(Math.Max(floats, 16), pinned: true);
            _outputs[model] = buffer;
        }
        return buffer;
    }

    /// <summary>Makes sure each worker has at least the given scratch, and the shared scratch its size.</summary>
    internal void EnsureScratch(long perWorker, long shared)
    {
        for (int i = 0; i < _scratch.Length; i++) Pointer(ref _scratch[i], perWorker);
        Pointer(ref _shared, shared);
    }

    /// <summary>The scratch of one worker, as sized by <see cref="EnsureScratch"/>.</summary>
    internal float* Scratch(int worker) => (float*)Unsafe.AsPointer(ref _scratch[worker]![0]);

    /// <summary>The scratch shared by all workers of one operator, as sized by <see cref="EnsureScratch"/>.</summary>
    internal float* SharedScratch => (float*)Unsafe.AsPointer(ref _shared![0]);

    /// <summary>
    /// Runs body(state, i, worker) for i in [0, count), on up to <see cref="MaxDegreeOfParallelism"/> threads. Each i
    /// runs exactly once and worker is unique among the calls running at the same time, so it can index per-worker
    /// scratch. Kernels give each i a fixed piece of the output and keep every reduction inside one piece, so results do
    /// not depend on the number of threads.
    /// </summary>
    internal void For(int count, object state, delegate*<object, int, int, void> body)
    {
        if (count <= 0) return;
        int workers = Math.Min(MaxDegreeOfParallelism, count);
        if (workers == 1)
        {
            for (int i = 0; i < count; i++) body(state, i, 0);
            return;
        }
        _jobState = state;
        _jobBody = body;
        _jobCount = count;
        _jobNext = 0;
        try
        {
            Parallel.For(0, workers, _options, _worker);
        }
        catch (AggregateException e) when (e.InnerExceptions.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
        }
        finally
        {
            _jobState = null;
        }
    }

    private void WorkerLoop(int worker)
    {
        var body = _jobBody;
        object state = _jobState!;
        int count = _jobCount;
        int i;
        while ((i = Interlocked.Increment(ref _jobNext) - 1) < count) body(state, i, worker);
    }

    private static float* Pointer(ref float[]? buffer, long floats)
    {
        // A little slack lets vector loops read a full vector past the end of the last value without a bounds branch.
        long needed = Math.Max(floats, 0) + 64;
        if (buffer is null || buffer.LongLength < needed)
        {
            buffer = null;
            if (needed > Array.MaxLength) throw new ArgumentException("The tensors of this input do not fit in one buffer.", "input");
            buffer = GC.AllocateUninitializedArray<float>((int)needed, pinned: true);
        }
        return (float*)Unsafe.AsPointer(ref buffer[0]);
    }
}
