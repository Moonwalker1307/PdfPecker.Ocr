namespace PdfPecker.Ocr;

/// <summary>
/// One of the three neural networks (detection, recognition, orientation) as the pipeline sees it: a float tensor in,
/// a float tensor out. The pipeline around the networks depends only on this seam, so it can be tested against outputs
/// recorded from ONNX Runtime, and the runtime in <c>Inference/</c> plugs in behind it with a small adapter.
/// </summary>
/// <remarks>
/// An instance is used by one thread at a time. The returned span stays valid until the next call on the same
/// instance, which lets an adapter hand out its own pooled output buffer instead of copying it.
/// </remarks>
internal interface INetwork
{
    /// <summary>Runs the network on an NCHW input of the given shape and returns the output and its shape.</summary>
    ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape);
}
