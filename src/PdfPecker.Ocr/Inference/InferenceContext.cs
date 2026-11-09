using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// The working memory and the threads of one inference worker. A context runs one model at a time: it owns the
/// activation arena, per-worker scratch buffers, the execution plans, the output buffers and, for a degree of
/// parallelism above one, that many threads less one (the caller's thread is the first worker), and reuses them across
/// runs, so a steady stream of runs of similar sizes allocates nothing. It is not thread-safe; give each concurrent
/// worker its own context, and dispose it to stop its threads. Models are shared.
/// </summary>
/// <remarks>
/// The arena and the shared scratch, the two large buffers, are native memory rather than pinned arrays. When a page
/// needs a larger arena than the last one, the old one is freed at once and its pages go back to the system; a pinned
/// array would stay resident until a full garbage collection and then often in the collector's free regions, so a run
/// of growing pages left several arenas resident. The finalizer frees them if a context is never disposed.
/// </remarks>
internal sealed unsafe class InferenceContext : IDisposable
{
    private readonly WorkerTeam? _team;
    private readonly Dictionary<OcrModel, ExecutionPlan> _plans = [];
    private readonly Dictionary<OcrModel, float[]> _outputs = [];
    private float* _arena;
    private long _arenaFloats;
    private float[]?[] _scratch;
    private float* _shared;
    private long _sharedFloats;
    private bool _disposed;

    /// <summary>Creates a context that uses at most the given number of threads for one run.</summary>
    public InferenceContext(int maxDegreeOfParallelism)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxDegreeOfParallelism, 1);
        MaxDegreeOfParallelism = maxDegreeOfParallelism;
        _team = maxDegreeOfParallelism > 1 ? new WorkerTeam(maxDegreeOfParallelism) : null;
        _scratch = new float[maxDegreeOfParallelism][];
    }

    /// <summary>The largest number of threads one run uses.</summary>
    public int MaxDegreeOfParallelism { get; }

    /// <summary>Checked between the operators of a run, so a long run can be stopped.</summary>
    public CancellationToken CancellationToken { get; set; }

    /// <summary>The size of the activation arena, in bytes, for memory measurements.</summary>
    public long ArenaBytes => _arenaFloats * sizeof(float);

    /// <summary>Releases the buffers and stops the context's threads. The context cannot run models afterwards.</summary>
    public void Dispose()
    {
        _disposed = true;
        _team?.Dispose();
        FreeNative();
        _scratch = [];
        _plans.Clear();
        _outputs.Clear();
        GC.SuppressFinalize(this);
    }

    ~InferenceContext() => FreeNative();

    private void FreeNative()
    {
        Free(ref _arena, ref _arenaFloats);
        Free(ref _shared, ref _sharedFloats);
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>The plan of a model for an input shape, reusing the previous one when the shape is unchanged.</summary>
    internal ExecutionPlan GetPlan(OcrModel model, ReadOnlySpan<int> inputShape)
    {
        if (_plans.TryGetValue(model, out var plan) && plan.InputShape.AsSpan().SequenceEqual(inputShape)) return plan;
        plan = ExecutionPlan.Create(model, inputShape.ToArray(), this);
        _plans[model] = plan;
        return plan;
    }

    /// <summary>The arena for a plan, grown when needed; its contents are not preserved across growth.</summary>
    internal float* Arena(long floats) => Native(ref _arena, ref _arenaFloats, floats);

    /// <summary>
    /// The output buffer of a model, owned by this context and overwritten by its next run of the model; offset is where
    /// its 64-byte aligned part starts.
    /// </summary>
    internal float[] OutputBuffer(OcrModel model, int floats, out int offset)
    {
        if (!_outputs.TryGetValue(model, out var buffer) || buffer.Length < floats + 16)
        {
            _outputs.Remove(model);
            buffer = GC.AllocateUninitializedArray<float>(floats + 16, pinned: true);
            _outputs[model] = buffer;
        }
        offset = (int)((Aligned(buffer) - (float*)Unsafe.AsPointer(ref buffer[0])));
        return buffer;
    }

    /// <summary>Makes sure each worker has at least the given scratch, and the shared scratch its size.</summary>
    internal void EnsureScratch(long perWorker, long shared)
    {
        for (int i = 0; i < _scratch.Length; i++) Pointer(ref _scratch[i], perWorker);
        Native(ref _shared, ref _sharedFloats, shared);
    }

    /// <summary>The scratch of one worker, as sized by <see cref="EnsureScratch"/>.</summary>
    internal float* Scratch(int worker) => Aligned(_scratch[worker]!);

    /// <summary>The scratch shared by all workers of one operator, as sized by <see cref="EnsureScratch"/>.</summary>
    internal float* SharedScratch => _shared;

    /// <summary>
    /// Runs body(state, i, worker) for i in [0, count), on up to <see cref="MaxDegreeOfParallelism"/> threads. Each i
    /// runs exactly once and worker is unique among the calls running at the same time, so it can index per-worker
    /// scratch. Kernels give each i a fixed piece of the output and keep every reduction inside one piece, so results do
    /// not depend on the number of threads.
    /// </summary>
    internal void For(int count, object state, delegate*<object, int, int, void> body)
    {
        if (count <= 0) return;
        if (_team is null || count == 1)
        {
            for (int i = 0; i < count; i++) body(state, i, 0);
            return;
        }
        _team.For(count, state, body);
    }

    /// <summary>A native buffer of at least the given floats plus vector slack, 64-byte aligned, reallocated when too small.</summary>
    private static float* Native(ref float* buffer, ref long capacity, long floats)
    {
        long needed = Math.Max(floats, 0) + 64;
        if (buffer != null && capacity >= needed) return buffer;
        if (needed > Array.MaxLength) throw new ArgumentException("The tensors of this input do not fit in one buffer.", "input");
        // The old contents are not kept, so the old buffer goes first and the two are never held together.
        Free(ref buffer, ref capacity);
        buffer = (float*)NativeMemory.AlignedAlloc((nuint)needed * sizeof(float), 64);
        capacity = needed;
        return buffer;
    }

    private static void Free(ref float* buffer, ref long capacity)
    {
        if (buffer != null) NativeMemory.AlignedFree(buffer);
        buffer = null;
        capacity = 0;
    }

    private static float* Pointer(ref float[]? buffer, long floats)
    {
        // A little slack lets vector loops read a full vector past the end of the last value without a bounds branch,
        // and 16 more floats leave room to align the start.
        long needed = Math.Max(floats, 0) + 64 + 16;
        if (buffer is null || buffer.LongLength < needed)
        {
            buffer = null;
            if (needed > Array.MaxLength) throw new ArgumentException("The tensors of this input do not fit in one buffer.", "input");
            buffer = GC.AllocateUninitializedArray<float>((int)needed, pinned: true);
        }
        return Aligned(buffer);
    }

    /// <summary>
    /// The first 64-byte aligned address in a pinned buffer. AVX-512 loads and stores that cross a cache line cost two,
    /// and tiles that meet inside a cache line would make threads fight over it.
    /// </summary>
    private static float* Aligned(float[] buffer) => (float*)(((nint)Unsafe.AsPointer(ref buffer[0]) + 63) & ~(nint)63);
}
