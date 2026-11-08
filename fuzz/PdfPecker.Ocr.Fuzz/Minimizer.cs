using System.Diagnostics;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Shrinks a failing input by removing pieces of it while it keeps failing the same way: whole tokens first for the
/// text formats, then runs of bytes, from large to small (a simple delta-debugging pass). A time and trial budget keeps
/// slow findings, where every trial costs the full timeout, from taking over the run.
/// </summary>
internal static class Minimizer
{
    public static byte[] Minimize(byte[] input, bool text, Func<byte[], bool> reproduces, TimeSpan budget, int maxTrials = 600)
    {
        var clock = Stopwatch.StartNew();
        int trials = 0;
        bool Try(byte[] candidate)
        {
            if (clock.Elapsed > budget || trials >= maxTrials) return false;
            trials++;
            return reproduces(candidate);
        }

        var current = input;
        if (text)
        {
            var tokens = Mutator.Tokenize(current);
            tokens = Reduce(tokens, units => Try(Join(units)), clock, budget);
            current = Join(tokens);
        }
        var bytes = current.Select(b => new[] { b }).ToList();
        bytes = Reduce(bytes, units => Try(Join(units)), clock, budget);
        return Join(bytes);
    }

    private static List<byte[]> Reduce(List<byte[]> units, Func<List<byte[]>, bool> reproduces, Stopwatch clock, TimeSpan budget)
    {
        for (int chunk = Math.Max(1, units.Count / 2); chunk >= 1 && clock.Elapsed < budget; chunk /= 2)
        {
            int at = 0;
            while (at < units.Count && clock.Elapsed < budget)
            {
                var candidate = new List<byte[]>(units.Count);
                candidate.AddRange(units.Take(at));
                candidate.AddRange(units.Skip(at + chunk));
                if (candidate.Count < units.Count && reproduces(candidate)) units = candidate; // the same offset now holds new units
                else at += chunk;
            }
            if (chunk == 1) break;
        }
        return units;
    }

    private static byte[] Join(List<byte[]> units)
    {
        var result = new byte[units.Sum(u => u.Length)];
        int at = 0;
        foreach (var u in units)
        {
            Buffer.BlockCopy(u, 0, result, at, u.Length);
            at += u.Length;
        }
        return result;
    }
}
