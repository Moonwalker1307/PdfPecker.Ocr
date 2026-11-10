namespace PdfPecker.Ocr;

/// <summary>
/// Runs some or all of OCR's neural networks somewhere other than the built-in CPU runtime, such as on a GPU through a
/// separate package. Set it as <see cref="OcrOptions.Accelerator"/>; everything around the networks (decoding, layout,
/// the text layer) stays the same.
/// </summary>
/// <remarks>
/// <para>
/// The pipeline asks for each network once per worker and keeps it while the worker is pooled, so creating one may be
/// slow but running one should not be. Workers are kept apart per accelerator instance: share one instance between
/// calls instead of creating one per call, or every call builds its networks again.
/// </para>
/// <para>
/// A network the accelerator does not provide (<see cref="CreateNetwork"/> returns null) runs on the built-in runtime,
/// which lets an accelerator split the work, for example detection on a GPU and recognition on the CPU, or fall back to
/// the CPU when no device is there. Exceptions the accelerator or its networks throw reach the caller as they are.
/// </para>
/// </remarks>
public abstract class OcrAccelerator
{
    /// <summary>A short name for logs and diagnostics, such as "CUDA" or "DirectML".</summary>
    public abstract string Name { get; }

    /// <summary>
    /// Creates the network of the given kind, or returns null to run that network on the built-in runtime. Called from
    /// any thread, possibly from several at once.
    /// </summary>
    /// <param name="kind">The network to create.</param>
    /// <returns>The network, which the pipeline disposes, or null.</returns>
    public abstract IOcrNetwork? CreateNetwork(OcrNetworkKind kind);
}
