namespace PdfPecker.Ocr;

/// <summary>
/// One of the networks of <see cref="OcrNetworkKind"/>, run by an <see cref="OcrAccelerator"/>: a float tensor in, a
/// float tensor out.
/// </summary>
/// <remarks>
/// An instance is used by one thread at a time, and only for the kind it was created for. The returned span must stay
/// valid until the next call to <see cref="Run"/> on the same instance, so an implementation can hand out a buffer it
/// reuses instead of allocating one per call. The input span is only valid during the call. The pipeline disposes the
/// instance when it no longer needs it.
/// </remarks>
public interface IOcrNetwork : IDisposable
{
    /// <summary>Runs the network on an input of the given NCHW shape and returns the output and its shape.</summary>
    /// <param name="input">The input tensor, with as many values as the shape holds.</param>
    /// <param name="shape">The input's shape, as <see cref="OcrNetworkKind"/> describes for the network's kind.</param>
    /// <param name="outputShape">The output's shape.</param>
    /// <returns>The output tensor, with at least as many values as <paramref name="outputShape"/> holds.</returns>
    ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape);
}
