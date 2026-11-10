namespace PdfPecker.Ocr.Inference;

/// <summary>
/// Callbacks from inside the runs of an inference context, for tests only. They cancel a token at a known operator of a
/// known run, which a delay cannot do on a machine of unknown speed, and see whether any operator, run or stage of the
/// work between runs follows it. Outside tests there is none, and the work pays a null check for each callback.
/// </summary>
internal sealed class RunObserver
{
    /// <summary>
    /// Called with the model as a run begins, before its input is prepared and before the token is first checked, so a
    /// test sees a run that was started after the token was cancelled even when the run itself then stops at once.
    /// </summary>
    public Action<OcrModel>? RunStarting { get; init; }

    /// <summary>
    /// Called as each step of a run begins, after the token was checked, with the model and the step's index, so a test
    /// sees a step that starts after the token was cancelled even when the token is checked again before the step ends.
    /// </summary>
    public Action<OcrModel, int>? StepStarting { get; init; }

    /// <summary>Called after each step of a run with the model and the step's index, on the thread that runs the model.</summary>
    public Action<OcrModel, int>? StepDone { get; init; }

    /// <summary>
    /// Called with its name as a stage of the work between the networks begins, after the token was checked, so a test
    /// sees that stage start after the token was cancelled even when a later loop checks the token before any run.
    /// </summary>
    public Action<string>? StageStarting { get; init; }
}
