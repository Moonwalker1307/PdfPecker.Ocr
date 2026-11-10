namespace PdfPecker.Ocr;

/// <summary>
/// The three neural networks OCR runs, which an <see cref="OcrAccelerator"/> can run instead of the built-in runtime.
/// Each one computes what its ONNX model in the rapidocr 3.10.0 wheel on PyPI computes, which is the model the built-in
/// runtime was converted from; the file names and hashes are in THIRD-PARTY-NOTICES.md. Tensors are 32-bit floats in
/// NCHW order, with the normalisation rapidocr applies already done by the pipeline.
/// </summary>
public enum OcrNetworkKind
{
    /// <summary>
    /// The text detector, PP-OCRv6 small (<c>PP-OCRv6_det_small.onnx</c>). Input [1, 3, H, W] with H and W multiples of
    /// 32; output [1, 1, H, W], the probability that each pixel is text.
    /// </summary>
    Detection,

    /// <summary>
    /// The line recognizer, PP-OCRv6 small (<c>PP-OCRv6_rec_small.onnx</c>). Input [N, 3, 48, W]; output [N, W / 8, C],
    /// a softmax over the model's C classes at each time step. An output of [N, W / 8, 2], the index of the most
    /// probable class (as a float) and its probability, is read the same way and is far smaller to copy back from a
    /// device.
    /// </summary>
    Recognition,

    /// <summary>
    /// The 0/180 degree line orientation classifier, PP-OCR mobile v2.0 (<c>ch_ppocr_mobile_v2.0_cls_mobile.onnx</c>).
    /// Input [N, 3, 48, 192]; output [N, 2], the probabilities of upright and upside down.
    /// </summary>
    Orientation,
}
