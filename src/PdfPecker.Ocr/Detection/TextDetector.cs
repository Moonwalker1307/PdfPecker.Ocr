using System.Buffers;
using System.Numerics;

namespace PdfPecker.Ocr.Detection;

/// <summary>
/// Finds text lines on a page: builds the detection network's input, runs the network, and turns the probability
/// map into boxes in page pixels. One instance per worker thread; it reuses its buffers between pages.
/// </summary>
internal sealed class TextDetector
{
    private readonly INetwork _network;
    private readonly DbPostProcessor _postProcessor = new();

    public TextDetector(INetwork network, DetectionOptions? options = null)
    {
        _network = network;
        Options = options ?? new DetectionOptions();
    }

    public DetectionOptions Options { get; }

    /// <summary>Detects the text regions of <paramref name="image"/>, as quads in its pixel coordinates.</summary>
    public List<DetectedQuad> Detect(PixelView image, CancellationToken cancellationToken = default)
    {
        DetectionInput.Layout layout = DetectionInput.Plan(image.Width, image.Height, Options);
        int dw = layout.InputWidth, dh = layout.InputHeight;
        int length = 3 * dw * dh;
        float[] input = ArrayPool<float>.Shared.Rent(length);
        try
        {
            DetectionInput.Fill(image, layout, input.AsSpan(0, length));
            cancellationToken.ThrowIfCancellationRequested();
            ReadOnlySpan<float> map = _network.Run(input.AsSpan(0, length), [1, 3, dh, dw], out int[] shape);
            if (shape.Length != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] != dh || shape[3] != dw || map.Length < dw * dh)
            {
                throw new InvalidOperationException("The detection network returned an unexpected shape.");
            }

            cancellationToken.ThrowIfCancellationRequested();
            List<DetectedQuad> boxes = _postProcessor.Process(map[..(dw * dh)], dw, dh, layout.PaddedWidth, layout.PaddedHeight, Options, cancellationToken);
            return layout.PaddingTop == 0 ? boxes : RemovePadding(boxes, layout.PaddingTop, image.Width, image.Height);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(input);
        }
    }

    /// <summary>
    /// Moves boxes found on the padded image back onto the image, clipped to it, as rapidocr's
    /// <c>map_boxes_to_original</c> does.
    /// </summary>
    private static List<DetectedQuad> RemovePadding(List<DetectedQuad> boxes, int top, int width, int height)
    {
        for (int i = 0; i < boxes.Count; i++)
        {
            Quad q = boxes[i].Quad;
            boxes[i] = boxes[i] with { Quad = new Quad(Shift(q.P0), Shift(q.P1), Shift(q.P2), Shift(q.P3)) };
        }

        return boxes;

        Vector2 Shift(Vector2 p) => new(Math.Clamp(p.X, 0, width), Math.Clamp(p.Y - top, 0, height));
    }
}
