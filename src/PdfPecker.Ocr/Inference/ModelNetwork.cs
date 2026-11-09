namespace PdfPecker.Ocr.Inference;

/// <summary>
/// A loaded model behind the pipeline's <see cref="INetwork"/> seam, run in a context that the worker's other networks
/// share. The input span is read where it is, and the output is the context's own buffer for this model, so nothing is
/// copied either way; that buffer is overwritten only when the same model runs in the same context again, which is
/// the next call on this adapter, as the seam promises.
/// </summary>
/// <remarks>
/// The networks of one worker run one after another on the worker's thread, so they can share a context: its arena
/// is sized for the largest of them and reused by all three, and each model keeps its own output buffer and plan.
/// </remarks>
internal sealed class ModelNetwork : INetwork
{
    private readonly OcrModel _model;
    private readonly InferenceContext _context;

    public ModelNetwork(OcrModel model, InferenceContext context)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(context);
        _model = model;
        _context = context;
    }

    public OcrModel Model => _model;

    public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
        => _model.Run(input, shape, _context, out outputShape);
}
