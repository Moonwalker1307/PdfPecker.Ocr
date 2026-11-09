using System.Text;

namespace TesseractComparison;

/// <summary>
/// The accuracy metrics. Text is normalised first (see <see cref="Normalize"/>), then compared as one string for the
/// character error rate and as a word sequence for the word error rate, so line breaks, paragraph breaks and the
/// number of spaces never count as errors, but the order of the text does.
/// </summary>
static class Metrics
{
    /// <summary>
    /// The normalisation applied to ground truth and engine output alike: Latin letters with combining marks composed (Unicode form C for the marks above), typographic quotes, dashes and the
    /// ellipsis folded to their ASCII forms, the no-break and narrow spaces treated as spaces, every run of whitespace
    /// (including line breaks) collapsed to one space, and the ends trimmed. Case, accents and punctuation are kept.
    /// </summary>
    public static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        bool space = true;
        foreach (char ch in Compose(text))
        {
            char c = ch switch
            {
                '‘' or '’' or '‚' or '′' => '\'',
                '“' or '”' or '„' or '″' => '"',
                '‐' or '‑' or '‒' or '–' or '—' or '−' => '-',
                _ => ch,
            };
            if (char.IsWhiteSpace(c) || c == '​' || c == '﻿')
            {
                if (!space) sb.Append(' ');
                space = true;
            }
            else
            {
                sb.Append(c);
                space = false;
            }
        }
        if (sb.Length > 0 && sb[^1] == ' ') sb.Length--;
        return sb.ToString();
    }

    // string.Normalize does nothing for non-ASCII text when globalization is invariant (the app is, so that it needs no ICU), so
    // the combining marks Latin text uses are composed here: grave, acute, circumflex, tilde, diaeresis, ring and cedilla.
    static readonly (char Mark, string Bases, string Composed)[] Compositions =
    [
        ('\u0300', "AEIOUaeiou", "ÀÈÌÒÙàèìòù"),
        ('\u0301', "AEIOUYaeiouy", "ÁÉÍÓÚÝáéíóúý"),
        ('\u0302', "AEIOUaeiou", "ÂÊÎÔÛâêîôû"),
        ('\u0303', "ANOano", "ÃÑÕãñõ"),
        ('\u0308', "AEIOUaeiouy", "ÄËÏÖÜäëïöüÿ"),
        ('\u030A', "Aa", "Åå"),
        ('\u0327', "Cc", "Çç"),
    ];

    static string Compose(string text)
    {
        if (text.AsSpan().IndexOfAnyInRange('\u0300', '\u036F') < 0) return text;
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (sb.Length > 0 && c >= '\u0300' && c <= '\u036F')
            {
                bool done = false;
                foreach (var (mark, bases, composed) in Compositions)
                {
                    int i = mark == c ? bases.IndexOf(sb[^1]) : -1;
                    if (i < 0) continue;
                    sb[^1] = composed[i];
                    done = true;
                    break;
                }
                if (done) continue;
            }
            sb.Append(c);
        }
        return sb.ToString();
    }

    public static string[] Words(string normalized) => normalized.Length == 0 ? [] : normalized.Split(' ');

    /// <summary>The Levenshtein distance between two sequences (insertions, deletions and substitutions cost 1).</summary>
    public static int Distance<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : IEquatable<T>
    {
        if (a.Length == 0) return b.Length;
        if (b.Length == 0) return a.Length;
        var prev = new int[b.Length + 1];
        var cur = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) prev[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            cur[0] = i;
            var ai = a[i - 1];
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = ai.Equals(b[j - 1]) ? 0 : 1;
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
            }
            (prev, cur) = (cur, prev);
        }
        return prev[b.Length];
    }

    /// <summary>
    /// The F1 score of the two word multisets, which ignores order: it separates a recognition error from a reading order
    /// difference (columns, or a page that is upside down) that the error rates count as errors.
    /// </summary>
    public static double BagF1(string[] reference, string[] hypothesis)
    {
        if (reference.Length == 0 && hypothesis.Length == 0) return 1;
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var w in reference) counts[w] = counts.GetValueOrDefault(w) + 1;
        int overlap = 0;
        foreach (var w in hypothesis)
            if (counts.TryGetValue(w, out int n) && n > 0) { counts[w] = n - 1; overlap++; }
        if (overlap == 0) return 0;
        double precision = (double)overlap / hypothesis.Length, recall = (double)overlap / reference.Length;
        return 2 * precision * recall / (precision + recall);
    }

    /// <summary>The precision and recall of the word multisets behind <see cref="BagF1"/>.</summary>
    public static (double Precision, double Recall) BagPrecisionRecall(string[] reference, string[] hypothesis)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var w in reference) counts[w] = counts.GetValueOrDefault(w) + 1;
        int overlap = 0;
        foreach (var w in hypothesis)
            if (counts.TryGetValue(w, out int n) && n > 0) { counts[w] = n - 1; overlap++; }
        return (hypothesis.Length == 0 ? (reference.Length == 0 ? 1 : 0) : (double)overlap / hypothesis.Length,
            reference.Length == 0 ? 1 : (double)overlap / reference.Length);
    }

    /// <summary>
    /// The words sorted ordinally and joined by spaces. The character distance between two such strings is a character error
    /// count that ignores reading order entirely; a misread first letter can move a word in the sort and cost up to twice its
    /// length, so it errs on the high side, but a column read in another order costs nothing.
    /// </summary>
    public static string SortedWords(string[] words)
    {
        var copy = (string[])words.Clone();
        Array.Sort(copy, StringComparer.Ordinal);
        return string.Join(' ', copy);
    }

    public static double Iou(Box a, Box b)
    {
        double x0 = Math.Max(a.X, b.X), y0 = Math.Max(a.Y, b.Y);
        double x1 = Math.Min(a.X + a.Width, b.X + b.Width), y1 = Math.Min(a.Y + a.Height, b.Y + b.Height);
        double inter = Math.Max(0, x1 - x0) * Math.Max(0, y1 - y0);
        double union = a.Width * a.Height + b.Width * b.Height - inter;
        return union <= 0 ? 0 : inter / union;
    }

    /// <summary>
    /// Matches each ground truth word to the unused engine word with the highest IoU (greedy, in ground truth order). A match
    /// needs an IoU of at least 0.5. Recall is matched ground truth words over all of them, precision matched engine words over
    /// all of them, and the mean IoU is over the matches only.
    /// </summary>
    public static (double Recall, double Precision, double MeanIou) MatchBoxes(IReadOnlyList<Box> truth, IReadOnlyList<Box> found)
    {
        if (truth.Count == 0) return (double.NaN, double.NaN, double.NaN);
        var used = new bool[found.Count];
        int matched = 0;
        double sum = 0;
        foreach (var t in truth)
        {
            int best = -1;
            double bestIou = 0.5;
            for (int j = 0; j < found.Count; j++)
            {
                if (used[j]) continue;
                double iou = Iou(t, found[j]);
                if (iou >= bestIou) { bestIou = iou; best = j; }
            }
            if (best < 0) continue;
            used[best] = true;
            matched++;
            sum += bestIou;
        }
        return ((double)matched / truth.Count, found.Count == 0 ? 0 : (double)matched / found.Count, matched == 0 ? double.NaN : sum / matched);
    }
}
