using System.Globalization;

namespace PdfPecker.Ocr.Inference;

/// <summary>
/// A dense float32 tensor in row-major order (NCHW for images). It is a shape over a block of memory, which may be the
/// caller's array or storage that an <see cref="InferenceContext"/> owns and reuses.
/// </summary>
internal sealed class Tensor
{
    /// <summary>Creates a zero-filled tensor of the given shape.</summary>
    public Tensor(params int[] shape)
        : this(new float[ElementCount(shape)], shape)
    {
    }

    /// <summary>Wraps memory the caller owns; its length must match the shape.</summary>
    public Tensor(Memory<float> data, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (data.Length != ElementCount(shape))
            throw new ArgumentException($"The buffer holds {data.Length} values, the shape needs {ElementCount(shape)}.", nameof(data));
        Shape = (int[])shape.Clone();
        Data = data;
    }

    /// <summary>The dimensions, outermost first.</summary>
    public int[] Shape { get; }

    /// <summary>The values in row-major order.</summary>
    public Memory<float> Data { get; }

    /// <summary>The values in row-major order.</summary>
    public Span<float> Span => Data.Span;

    /// <summary>The number of values.</summary>
    public int Length => Data.Length;

    /// <summary>The number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>Wraps a float buffer the caller filled, without copying it.</summary>
    public static Tensor FromBuffer(float[] buffer, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        return new Tensor(buffer.AsMemory(), shape);
    }

    /// <summary>A copy that owns its storage, for keeping a result after the context that produced it runs again.</summary>
    public Tensor Clone() => new(Data.ToArray().AsMemory(), Shape);

    /// <summary>The value at a full index, for tests and diagnostics.</summary>
    public float this[params int[] index]
    {
        get
        {
            if (index.Length != Shape.Length) throw new ArgumentException("The index rank does not match the tensor.", nameof(index));
            int offset = 0;
            for (int i = 0; i < index.Length; i++)
            {
                if ((uint)index[i] >= (uint)Shape[i]) throw new ArgumentOutOfRangeException(nameof(index));
                offset = offset * Shape[i] + index[i];
            }
            return Data.Span[offset];
        }
    }

    /// <summary>The number of values a shape holds; negative dimensions or overflow throw.</summary>
    public static int ElementCount(ReadOnlySpan<int> shape)
    {
        long count = 1;
        foreach (int d in shape)
        {
            if (d < 0) throw new ArgumentException("Tensor dimensions cannot be negative.", nameof(shape));
            count *= d;
            if (count > Array.MaxLength) throw new ArgumentException("The tensor is too large.", nameof(shape));
        }
        return (int)count;
    }

    /// <inheritdoc/>
    public override string ToString() => "Tensor[" + string.Join(", ", Shape.Select(d => d.ToString(CultureInfo.InvariantCulture))) + "]";
}
