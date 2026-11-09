using System.Runtime.CompilerServices;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// Where a value of one run holds equal columns: for each batch item n, every slice at positions Start[n] to End[n] - 1
/// along <see cref="Axis"/> is the same, bit for bit, at every index of the other axes. The recogniser pads each batch
/// with zeros to its widest line, and a short line in a batch of long ones is mostly padding: up to the first layer that
/// mixes a whole line (squeeze-excitation, attention), the columns of that padding beyond the reach of the line's own
/// pixels and of the right edge all hold the same values, and past it they still do, because those layers treat every
/// column alike. A kernel whose every output value is computed by the same operations wherever it lies (all of this
/// runtime's kernels: see <see cref="Kernels.GemmKernel"/>) may compute the first column of a run and copy it to the
/// others, which gives every bit of the full computation.
/// </summary>
/// <remarks>
/// The runs are found anew for each input (<see cref="ExecutionPlan.FindRuns"/>), unless the context turns them off
/// (<see cref="InferenceContext.SkipsEqualColumns"/>): the input's trailing columns that are
/// equal to its last one, then each layer's <see cref="Layer.FindRuns"/> from those of its inputs. A layer that does not
/// know how its columns map to its inputs' finds none, and so do all layers after it, so the runs are only ever smaller
/// than what is really equal.
/// </remarks>
internal sealed class ColumnRuns
{
    public ColumnRuns(int batch)
    {
        Start = new int[batch];
        End = new int[batch];
    }

    /// <summary>The axis the runs lie along, or -1 when nothing is known to be equal.</summary>
    public int Axis { get; private set; } = -1;

    public int[] Start { get; }

    public int[] End { get; }

    public int Batch => Start.Length;

    /// <summary>Whether item n has a run of at least two columns, the only kind there is something to skip in.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool HasRun(int n) => Axis >= 0 && End[n] - Start[n] >= 2;

    /// <summary>Whether any item has a run of at least <paramref name="length"/> columns along <paramref name="axis"/>.</summary>
    public bool Any(int axis, int length = 2)
    {
        if (Axis != axis) return false;
        for (int n = 0; n < Start.Length; n++)
            if (End[n] - Start[n] >= length) return true;
        return false;
    }

    public void Clear() => Axis = -1;

    /// <summary>Starts runs along an axis, every item's empty.</summary>
    public void Reset(int axis)
    {
        Axis = axis;
        Array.Clear(Start);
        Array.Clear(End);
    }

    /// <summary>Sets item n's run, empty when end is not after start.</summary>
    public void Set(int n, int start, int end)
    {
        Start[n] = start;
        End[n] = Math.Max(start, end);
    }

    /// <summary>The runs of <paramref name="source"/>, along <paramref name="axis"/> of this value.</summary>
    public void CopyFrom(ColumnRuns source, int axis)
    {
        if (source.Axis < 0 || axis < 0 || source.Batch != Batch)
        {
            Clear();
            return;
        }

        Axis = axis;
        source.Start.CopyTo(Start, 0);
        source.End.CopyTo(End, 0);
    }

    /// <summary>Keeps only the columns that are also in a run of <paramref name="other"/>, along the same axis.</summary>
    public void Intersect(ColumnRuns other)
    {
        if (Axis < 0) return;
        if (other.Axis != Axis || other.Batch != Batch)
        {
            Clear();
            return;
        }

        for (int n = 0; n < Start.Length; n++)
            Set(n, Math.Max(Start[n], other.Start[n]), Math.Min(End[n], other.End[n]));
    }

    /// <summary>
    /// Copies each item's first column of its run over the rest of it, in a value whose runs lie along its last axis
    /// (a [N, ..., W] tensor, one row of W values per index of the axes between): the columns a kernel skipped get the
    /// values it would have computed; runs of fewer than <paramref name="minimum"/> columns after the first were
    /// computed in full and are left alone. The rows go in tiles on the context's threads.
    /// </summary>
    public unsafe void FillLastAxis(float* data, int[] shape, InferenceContext context, int minimum = 1)
    {
        if (!Any(shape.Length - 1, minimum + 1)) return;
        var fill = new Fill(this, data, shape, minimum);
        context.For(fill.Tiles, fill, &FillTile);
    }

    private static unsafe void FillTile(object state, int tile, int worker)
    {
        var fill = (Fill)state;
        int n = tile / fill.TilesPerItem;
        var runs = fill.Runs;
        int s = runs.Start[n], e = runs.End[n], w = fill.Width;
        if (e - s - 1 < fill.Minimum) return;
        long first = (long)(tile % fill.TilesPerItem) * fill.RowsPerTile;
        long last = Math.Min(fill.RowsPerItem, first + fill.RowsPerTile);
        float* row = fill.Data + ((long)n * fill.RowsPerItem + first) * w;
        for (long r = first; r < last; r++, row += w) new Span<float>(row + s + 1, e - s - 1).Fill(row[s]);
    }

    /// <summary>One fill's tensor and tiles, handed to the threads.</summary>
    private sealed unsafe class Fill
    {
        public Fill(ColumnRuns runs, float* data, int[] shape, int minimum)
        {
            Runs = runs;
            Minimum = Math.Max(1, minimum);
            Data = data;
            Width = shape[^1];
            long rows = 1;
            for (int d = 1; d < shape.Length - 1; d++) rows *= shape[d];
            RowsPerItem = rows;
            RowsPerTile = Math.Max(1, (int)Math.Min(rows, 16384 / Math.Max(1, Width)));
            TilesPerItem = (int)((rows + RowsPerTile - 1) / RowsPerTile);
            Tiles = TilesPerItem * runs.Batch;
        }

        public ColumnRuns Runs { get; }

        public int Minimum { get; }

        public float* Data { get; }

        public int Width { get; }

        public long RowsPerItem { get; }

        public int RowsPerTile { get; }

        public int TilesPerItem { get; }

        public int Tiles { get; }
    }

    /// <summary>
    /// The runs of the input to a window of <paramref name="kernel"/> taps <paramref name="dilation"/> apart, moved by
    /// <paramref name="stride"/> and starting <paramref name="padStart"/> before the input: output column x reads input
    /// columns x * stride - padStart + i * dilation, and is in a run when all of them are in the input's run. Windows
    /// that reach into the padding are never in one, since the padding is not the run's value.
    /// </summary>
    public void FromWindow(ColumnRuns input, int axis, int kernel, int stride, int dilation, int padStart, int outputLength)
    {
        if (input.Axis != axis || input.Batch != Batch)
        {
            Clear();
            return;
        }

        Axis = axis;
        int reach = dilation * (kernel - 1);
        for (int n = 0; n < Start.Length; n++)
        {
            int s = input.Start[n], e = input.End[n];
            // The first x with x * stride - padStart >= s, and one past the last with x * stride - padStart + reach <= e - 1.
            int first = CeilDiv(s + padStart, stride);
            int limit = e - 1 + padStart - reach;
            int end = limit < 0 ? 0 : (limit / stride) + 1;
            Set(n, Math.Clamp(first, 0, outputLength), Math.Clamp(end, 0, outputLength));
        }
    }

    private static int CeilDiv(int a, int b) => a <= 0 ? -(-a / b) : ((a - 1) / b) + 1;
}
