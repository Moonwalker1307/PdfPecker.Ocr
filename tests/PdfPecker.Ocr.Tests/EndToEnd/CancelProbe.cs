using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>
/// Cancels a token from inside the engine's network runs at a chosen operator, and counts the operators, runs and stages
/// of the work between runs that start after it. It observes the pages read in the calling flow of execution (see
/// <see cref="OcrEngine.Observe"/>) until it is disposed, through every public entry point, and tests that run at the
/// same time on other flows do not see it. A probe that never cancels counts the operators of each run, for a test that
/// then cancels at the last one.
/// </summary>
internal sealed class CancelProbe : IDisposable
{
    private readonly CancellationTokenSource _cancel = new();
    private readonly Func<Step, bool> _cancelAt;
    private readonly IDisposable _scope;
    private readonly Dictionary<OcrModel, int> _runs = [];
    private readonly Dictionary<(OcrModel Model, int Run), int> _steps = [];

    /// <summary>A probe that cancels after the first operator for which <paramref name="cancelAt"/> is true.</summary>
    public CancelProbe(Func<Step, bool> cancelAt)
    {
        _cancelAt = cancelAt;
        _scope = OcrEngine.Observe(new RunObserver { RunStarting = Starting, StepStarting = StepStarting, StepDone = Done, StageStarting = StageStarting });
    }

    /// <summary>An operator that has just run: its model, the run of that model (from 0), its index in the run, and how many operators of any run came before it.</summary>
    internal readonly record struct Step(OcrModel Model, int Run, int Index, int Before)
    {
        public override string ToString() => $"operator {Index} of run {Run} of {Model.Name} ({Before} operators before it)";
    }

    /// <summary>The token to hand to the call.</summary>
    public CancellationToken Token => _cancel.Token;

    /// <summary>Whether the probe has cancelled the token.</summary>
    public bool Cancelled => _cancel.IsCancellationRequested;

    /// <summary>The operator after which the token was cancelled.</summary>
    public Step? CancelledAt { get; private set; }

    /// <summary>The operators that ran before the token was cancelled, or all of them.</summary>
    public int Steps { get; private set; }

    /// <summary>The operators that started after the token was cancelled, also when they stopped before their end.</summary>
    public int StepsAfter { get; private set; }

    /// <summary>The stages of the work between runs that started after the token was cancelled, by name.</summary>
    public List<string> StagesAfter { get; } = [];

    /// <summary>The runs that started after the token was cancelled, also when they stopped before their first operator.</summary>
    public int RunsAfter { get; private set; }

    /// <summary>The runs of a model that started before the token was cancelled.</summary>
    public int Runs(OcrModel model) => _runs.GetValueOrDefault(model);

    /// <summary>The index of the last operator of a run that ran before the token was cancelled.</summary>
    public int LastStep(OcrModel model, int run) => _steps[(model, run)] - 1;

    /// <summary>
    /// Checks that the token was cancelled and that the work stopped right there: not one more operator started, and
    /// no run or stage of the work between runs. The call is expected to have thrown already.
    /// </summary>
    public void AssertStoppedAtOnce()
    {
        Assert.True(Cancelled, $"The work ended after {Steps} operators, before the point of cancellation.");
        Assert.True(StepsAfter == 0, $"Cancelled after {CancelledAt}, and {StepsAfter} more operators started.");
        Assert.True(RunsAfter == 0, $"Cancelled after {CancelledAt}, and {RunsAfter} more network runs started.");
        Assert.True(StagesAfter.Count == 0, $"Cancelled after {CancelledAt}, and then the {string.Join(", ", StagesAfter)} started.");
    }

    public void Dispose()
    {
        _scope.Dispose();
        _cancel.Dispose();
    }

    private void Starting(OcrModel model)
    {
        if (Cancelled)
        {
            RunsAfter++;
            return;
        }

        _runs[model] = Runs(model) + 1;
    }

    private void StepStarting(OcrModel model, int index)
    {
        if (Cancelled)
        {
            StepsAfter++;
        }
    }

    private void StageStarting(string stage)
    {
        if (Cancelled)
        {
            StagesAfter.Add(stage);
        }
    }

    private void Done(OcrModel model, int index)
    {
        // An operator that ends after the token was cancelled also started after it, and was counted then.
        if (Cancelled)
        {
            return;
        }

        int run = _runs[model] - 1;
        _steps[(model, run)] = index + 1;
        var step = new Step(model, run, index, Steps);
        Steps++;
        if (_cancelAt(step))
        {
            CancelledAt = step;
            _cancel.Cancel();
        }
    }

    /// <summary>Whether the model is one of the detection networks, which one depending on the page's channels.</summary>
    public static bool IsDetection(OcrModel model) => model == ModelStore.Detection || model == ModelStore.DetectionGray;
}
