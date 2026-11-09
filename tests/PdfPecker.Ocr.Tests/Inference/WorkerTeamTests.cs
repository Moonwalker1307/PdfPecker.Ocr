using PdfPecker.Ocr.Inference;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>The threads an inference context keeps between runs.</summary>
public sealed unsafe class WorkerTeamTests
{
    private static readonly AsyncLocal<string?> s_ambient = new();

    [Fact]
    public void ThreadsDoNotKeepTheFirstCallersExecutionContext()
    {
        // A pooled context outlives the call that started its threads; they must not hold that call's async locals.
        var seen = new Seen();
        using (var team = new WorkerTeam(4))
        {
            s_ambient.Value = "first caller";
            try
            {
                team.For(64, seen, &Record);
            }
            finally
            {
                s_ambient.Value = null;
            }
        }

        Assert.True(seen.Others > 0, "No item ran on the team's own threads.");
        Assert.Equal(0, seen.OthersWithAmbient);
    }

    private static void Record(object state, int item, int worker)
    {
        var seen = (Seen)state;
        // Slow items, so that the team's threads wake up and take some of them.
        Thread.Sleep(2);
        if (worker == 0) return;
        Interlocked.Increment(ref seen.Others);
        if (s_ambient.Value is not null) Interlocked.Increment(ref seen.OthersWithAmbient);
    }

    private sealed class Seen
    {
        public int Others;
        public int OthersWithAmbient;
    }
}
