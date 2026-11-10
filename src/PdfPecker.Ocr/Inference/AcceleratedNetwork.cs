namespace PdfPecker.Ocr.Inference;

/// <summary>
/// A network of an <see cref="OcrAccelerator"/> behind the pipeline's <see cref="INetwork"/> seam. The pipeline builds
/// each input in <see cref="InputBuffer"/>, a buffer kept between runs and grown to the largest input, so a page
/// allocates no input tensors once its worker is warm.
/// </summary>
internal sealed class AcceleratedNetwork : INetwork, IDisposable
{
    private readonly IOcrNetwork _network;
    private float[] _input = [];

    public AcceleratedNetwork(IOcrNetwork network)
    {
        ArgumentNullException.ThrowIfNull(network);
        _network = network;
    }

    public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
    {
        ReadOnlySpan<float> output = _network.Run(input, shape, out outputShape);

        // The stages check the shape against their input before reading, but an accelerator is code outside the package,
        // so a null shape or one that claims more values than were returned fails here with the stages' exception.
        if (outputShape is null || !Holds(output.Length, outputShape))
        {
            throw new InvalidOperationException("The accelerator's network returned fewer values than its output shape holds.");
        }

        return output;
    }

    /// <summary>Whether <paramref name="length"/> values are enough for <paramref name="shape"/>, which may be anything.</summary>
    private static bool Holds(int length, int[] shape)
    {
        long count = 1;
        foreach (int d in shape)
        {
            if (d < 0) return false;
            count *= d;
            if (count > length) return false;
        }

        return true;
    }

    public Span<float> InputBuffer(ReadOnlySpan<int> shape)
    {
        int count = Tensor.ElementCount(shape);
        if (_input.Length < count)
        {
            _input = new float[count];
        }

        return _input.AsSpan(0, count);
    }

    public void Dispose() => _network.Dispose();
}
