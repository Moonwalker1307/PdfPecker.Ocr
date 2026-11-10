namespace PdfPecker.Ocr.Inference;

/// <summary>
/// Callbacks from inside the runs of an inference context, for tests only. They cancel a token at a known operator of a
/// known run, which a delay cannot do on a machine of unknown speed, and see whether any operator or run follows it.
/// Outside tests there is none, and a run pays one null check for it.
/// </summary>
internal sealed class RunObserver
{
    /// <summary>
    /// Called with the model as a run begins, before its input is prepared and before the token is first checked, so a
    /// test sees a run that was started after the token was cancelled even when the run itself then stops at once.
    /// </summary>
    public Action<OcrModel>? RunStarting { get; init; }

    /// <summary>Called after each step of a run with the model and the step's index, on the thread that runs the model.</summary>
    public Action<OcrModel, int>? StepDone { get; init; }
}
