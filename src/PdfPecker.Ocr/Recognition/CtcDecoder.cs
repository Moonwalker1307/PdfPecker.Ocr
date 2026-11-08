using System.Numerics;
using System.Text;

namespace PdfPecker.Ocr.Recognition;

/// <summary>One line as the recogniser read it, before it is split into words.</summary>
internal sealed class DecodedLine
{
    public DecodedLine(string text, float score, int[] classes, int[] columns, float[] probabilities, int steps)
    {
        Text = text;
        Score = score;
        Classes = classes;
        Columns = columns;
        Probabilities = probabilities;
        Steps = steps;
    }

    /// <summary>The characters joined, as rapidocr returns the line's text.</summary>
    public string Text { get; }

    /// <summary>rapidocr's confidence: the mean of the kept time steps' probabilities, each rounded to 5 decimals.</summary>
    public float Score { get; }

    /// <summary>The class of each kept character.</summary>
    public int[] Classes { get; }

    /// <summary>The time step at which each kept character was emitted, the position its box is built from.</summary>
    public int[] Columns { get; }

    /// <summary>The probability of each kept character.</summary>
    public float[] Probabilities { get; }

    /// <summary>The number of time steps of the network output the line was decoded from.</summary>
    public int Steps { get; }
}

/// <summary>
/// Greedy CTC decoding as rapidocr's <c>CTCLabelDecode</c> does it (Apache-2.0, https://github.com/RapidAI/RapidOCR):
/// the most probable class at each time step, repeats collapsed, blanks dropped, and the confidence the mean of the
/// kept probabilities. It also keeps the time step of each character, from which word boxes are placed.
/// </summary>
internal static class CtcDecoder
{
    /// <summary>
    /// Decodes one line from <paramref name="probabilities"/>, <paramref name="steps"/> rows of
    /// <paramref name="classCount"/> class probabilities. With <paramref name="subset"/> (ascending class indices that
    /// include the blank), only those classes compete at each step, which also reads far less memory.
    /// </summary>
    public static DecodedLine Decode(ReadOnlySpan<float> probabilities, int steps, int classCount, CharacterSet characters, int[]? subset = null)
    {
        if (classCount != characters.Count)
        {
            throw new InvalidOperationException($"The recognition network has {classCount} classes, the character list {characters.Count}.");
        }

        var text = new StringBuilder();
        var classes = new List<int>();
        var columns = new List<int>();
        var probs = new List<float>();
        double sum = 0;
        int previous = -1;
        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> row = probabilities.Slice(t * classCount, classCount);
            (int index, float p) = subset is null ? ArgMax(row) : ArgMax(row, subset);
            if (index != previous && index != 0)
            {
                text.Append(characters[index]);
                classes.Add(index);
                columns.Add(t);
                probs.Add(p);
                sum += Math.Round((double)p, 5, MidpointRounding.ToEven);
            }

            previous = index;
        }

        // rapidocr averages [0] when nothing was kept, which gives an empty line a score of 0.
        float score = classes.Count == 0 ? 0 : (float)Math.Round(sum / classes.Count, 5, MidpointRounding.ToEven);
        return new DecodedLine(text.ToString(), score, [.. classes], [.. columns], [.. probs], steps);
    }

    /// <summary>The first index of the largest value, as numpy's argmax, and the value. NaN never wins.</summary>
    internal static (int Index, float Value) ArgMax(ReadOnlySpan<float> row)
    {
        float max = float.NegativeInfinity;
        int i = 0;
        if (Vector.IsHardwareAccelerated && row.Length >= 2 * Vector<float>.Count)
        {
            var best = new Vector<float>(float.NegativeInfinity);
            for (; i <= row.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                best = Vector.Max(best, new Vector<float>(row.Slice(i)));
            }

            for (int k = 0; k < Vector<float>.Count; k++)
            {
                max = MathF.Max(max, best[k]);
            }
        }

        for (; i < row.Length; i++)
        {
            if (row[i] > max)
            {
                max = row[i];
            }
        }

        int index = row.IndexOf(max);
        return index < 0 ? (0, row.Length > 0 ? row[0] : 0) : (index, max);
    }

    /// <summary>The argmax over the classes in <paramref name="subset"/> only.</summary>
    internal static (int Index, float Value) ArgMax(ReadOnlySpan<float> row, int[] subset)
    {
        int index = subset[0];
        float max = row[index];
        for (int k = 1; k < subset.Length; k++)
        {
            float v = row[subset[k]];
            if (v > max)
            {
                max = v;
                index = subset[k];
            }
        }

        return (index, max);
    }
}
