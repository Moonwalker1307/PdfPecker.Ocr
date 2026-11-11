using System.Numerics;
using System.Text;
using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// Counts runs of dots on the page, which the recogniser cannot: it emits at most one character per time step of
/// 8 pixels at a line height of 48, and CTC needs a blank step between two equal characters, so a run of dots set
/// closer than 16 pixels apart comes out short or as an ellipsis. The detector does not see a dot leader as text at
/// all, so a table of contents line is read as its title and its page number. Nothing here guesses a letter: dots
/// are only counted where the recogniser read dots in a line, or in the gap between two lines on the same row, and a
/// run is only taken when every mark in it is a round dot on the baseline with nothing above or below it, at an even
/// pitch.
/// </summary>
internal static class DotRuns
{
    /// <summary>Lines of horizontal text tilted more than this, as a slope, are left as the recogniser read them.</summary>
    private const float MaxSlope = 0.05f;

    /// <summary>The most pixels one band may hold, so a page of huge lines cannot make the counting slow.</summary>
    private const long MaxBandPixels = 4_000_000;

    /// <summary>Reads quads of the page as lines, as <see cref="TextRecognizer.Recognize"/> does.</summary>
    public delegate List<RecognizedLine> LineReader(PixelView page, IReadOnlyList<DetectedQuad> quads, CancellationToken cancellationToken);

    /// <summary>
    /// Returns <paramref name="lines"/> with the dot runs counted: lines whose text has runs of dots get as many dots
    /// as the page shows, and two lines on one row with a dot leader between them become one line. A line with a
    /// leader is read again without it by <paramref name="read"/>, when given, because the dots also cost the
    /// recogniser the characters next to them ("41" read as "4").
    /// </summary>
    public static List<RecognizedLine> Apply(PixelView page, List<RecognizedLine> lines, LineReader? read = null, CancellationToken cancellationToken = default)
    {
        var leaders = new Dictionary<RecognizedLine, List<PageRow>>(ReferenceEqualityComparer.Instance);
        var result = new List<RecognizedLine>(lines.Count);
        foreach (RecognizedLine line in lines)
        {
            cancellationToken.ThrowIfCancellationRequested();
            result.Add(Horizontal(line) && HasTextRun(line.Text) ? CountInLine(page, line, leaders) ?? line : line);
        }

        result = JoinLeaders(page, result, leaders, read, cancellationToken);
        return read is null || leaders.Count == 0 ? result : ReadAgain(page, result, leaders, read, cancellationToken);
    }

    /// <summary>A counted run of dots in page pixels, and whether the text has a space before and after it.</summary>
    private readonly record struct PageRow(int Count, float Left, float Right, bool SpaceBefore, bool SpaceAfter);

    /// <summary>The fewest dots in a run that is read as a leader, whose line is read again without it.</summary>
    private const int LeaderDots = 5;

    /// <summary>
    /// Reads the text between the runs of dots of each line with a leader again, every piece on its own, and puts the
    /// line together from the pieces and the counted dots. A line keeps its first reading when a piece reads as
    /// nothing.
    /// </summary>
    private static List<RecognizedLine> ReadAgain(PixelView page, List<RecognizedLine> lines, Dictionary<RecognizedLine, List<PageRow>> leaders, LineReader read, CancellationToken cancellationToken)
    {
        var quads = new List<DetectedQuad>();
        var pieces = new Dictionary<RecognizedLine, (int First, int Count)>(ReferenceEqualityComparer.Instance);
        foreach (RecognizedLine line in lines)
        {
            if (!leaders.TryGetValue(line, out List<PageRow>? rows))
            {
                continue;
            }

            var (left, top, right, bottom) = line.Quad.Bounds();
            float pad = 0.1f * (bottom - top);
            int first = quads.Count;
            float x = left;
            foreach (PageRow row in rows)
            {
                quads.Add(new DetectedQuad(Rectangle(x, top, MathF.Max(x, row.Left - pad), bottom), 1));
                x = row.Right + pad;
            }

            quads.Add(new DetectedQuad(Rectangle(MathF.Min(x, right), top, right, bottom), 1));
            pieces[line] = (first, quads.Count - first);
        }

        List<RecognizedLine> readings = read(page, quads, cancellationToken);
        var byQuad = new RecognizedLine?[quads.Count];
        foreach (RecognizedLine reading in readings)
        {
            byQuad[reading.DetectionIndex] = reading;
        }

        var result = new List<RecognizedLine>(lines.Count);
        foreach (RecognizedLine line in lines)
        {
            if (!pieces.TryGetValue(line, out var at))
            {
                result.Add(line);
                continue;
            }

            List<PageRow> rows = leaders[line];
            var (_, top, _, bottom) = line.Quad.Bounds();
            float minWidth = 0.3f * (bottom - top);
            var text = new StringBuilder();
            var words = new List<RecognizedWord>();
            double confidence = 0, weight = 0;
            bool complete = true;
            for (int k = 0; k < at.Count && complete; k++)
            {
                RecognizedLine? piece = byQuad[at.First + k];
                var (pl, _, pr, _) = quads[at.First + k].Quad.Bounds();
                if (piece is null)
                {
                    // A piece of blank paper reads as nothing; one with room for a character must read as something.
                    complete = pr - pl < minWidth || (k > 0 && k < at.Count - 1);
                }
                else
                {
                    if (text.Length > 0 && (k == 0 || rows[k - 1].SpaceAfter))
                    {
                        text.Append(' ');
                    }

                    text.Append(piece.Text);
                    words.AddRange(piece.Words);
                    confidence += piece.Confidence * piece.Text.Length;
                    weight += piece.Text.Length;
                }

                if (k < rows.Count)
                {
                    PageRow row = rows[k];
                    if (text.Length > 0 && row.SpaceBefore)
                    {
                        text.Append(' ');
                    }

                    text.Append('.', row.Count);
                    words.Add(new RecognizedWord(new string('.', row.Count), line.Confidence, Rectangle(row.Left, top, row.Right, bottom)));
                    confidence += line.Confidence * row.Count;
                    weight += row.Count;
                }
            }

            result.Add(complete ? new RecognizedLine(text.ToString(), (float)(confidence / weight), line.Quad, Joined(words, text.ToString()), line.DetectionIndex, line.UpsideDown) : line);
        }

        return result;
    }

    /// <summary>
    /// The words of a line put together from pieces: a run of dots with no space before or after it joins the word
    /// next to it, so the words still spell the line's text.
    /// </summary>
    private static List<RecognizedWord> Joined(List<RecognizedWord> words, string text)
    {
        string[] tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length == words.Count)
        {
            return words;
        }

        var result = new List<RecognizedWord>(tokens.Length);
        int w = 0;
        foreach (string token in tokens)
        {
            if (w >= words.Count)
            {
                break;
            }

            RecognizedWord word = words[w++];
            var (l, t, r, b) = word.Quad.Bounds();
            float confidence = word.Confidence;
            string spelled = word.Text;
            while (spelled.Length < token.Length && w < words.Count)
            {
                RecognizedWord next = words[w++];
                spelled += next.Text;
                var (nl, nt, nr, nb) = next.Quad.Bounds();
                (l, t, r, b) = (MathF.Min(l, nl), MathF.Min(t, nt), MathF.Max(r, nr), MathF.Max(b, nb));
                confidence = MathF.Min(confidence, next.Confidence);
            }

            result.Add(new RecognizedWord(token, confidence, Rectangle(l, t, r, b)));
        }

        return result;
    }

    /// <summary>Whether a line reads left to right without much tilt, the only lines whose dots are counted.</summary>
    private static bool Horizontal(RecognizedLine line)
    {
        Quad q = line.Quad;
        float dx = q.P1.X - q.P0.X;
        return !line.UpsideDown && dx > 0 && MathF.Abs(q.P1.Y - q.P0.Y) <= MaxSlope * dx && MathF.Abs(q.P3.X - q.P0.X) <= MaxSlope * q.Height * 4;
    }

    private static bool HasTextRun(string text)
    {
        foreach ((int start, int end) in TextRuns(text))
        {
            return start < end;
        }

        return false;
    }

    /// <summary>
    /// The runs of dots in a line's text, as start and end (exclusive) indices: two or more full stops, or an ellipsis,
    /// with spaces between them allowed.
    /// </summary>
    internal static List<(int Start, int End)> TextRuns(string text)
    {
        var runs = new List<(int, int)>();
        int i = 0;
        while (i < text.Length)
        {
            if (text[i] is not ('.' or '…'))
            {
                i++;
                continue;
            }

            int start = i, end = i, units = 0;
            for (int j = i; j < text.Length; j++)
            {
                if (text[j] == '.')
                {
                    units++;
                    end = j + 1;
                }
                else if (text[j] == '…')
                {
                    units += 3;
                    end = j + 1;
                }
                else if (text[j] != ' ')
                {
                    break;
                }
            }

            // A lone full stop between spaces is a leader the recogniser read as one dot, or nothing the page will
            // confirm as a run.
            bool lone = units == 1 && (start == 0 || text[start - 1] == ' ') && (end == text.Length || text[end] == ' ') && start > 0;
            if (units >= 2 || lone)
            {
                runs.Add((start, end));
            }

            i = end;
        }

        return runs;
    }

    /// <summary>
    /// Whether a run of text has spaces between some of its dots but not others, which the evenly spaced dots on the
    /// page it was matched with do not have.
    /// </summary>
    private static bool MixedSpacing(string text, int start, int end)
    {
        bool spaced = false, joined = false;
        for (int i = start + 1; i < end; i++)
        {
            if (text[i] == ' ')
            {
                spaced = true;
            }
            else if (text[i - 1] != ' ')
            {
                joined = true;
            }
        }

        return spaced && joined;
    }

    /// <summary>The number of dots a run of text stands for, an ellipsis counting three.</summary>
    private static int Units(string text, int start, int end)
    {
        int units = 0;
        for (int i = start; i < end; i++)
        {
            units += text[i] == '.' ? 1 : text[i] == '…' ? 3 : 0;
        }

        return units;
    }

    /// <summary>
    /// Counts the dots of each run in a line's text on the page. The line is left as it is (null) unless the page shows
    /// exactly as many runs of dots as the text has, in the same order.
    /// </summary>
    private static RecognizedLine? CountInLine(PixelView page, RecognizedLine line, Dictionary<RecognizedLine, List<PageRow>> leaders)
    {
        List<(int Start, int End)> runs = TextRuns(line.Text);
        var (left, top, right, bottom) = line.Quad.Bounds();
        Band? band = Band.Read(page, left, top, right, bottom);
        if (band is null || !band.FindLetters(out float letterHeight, out float baseline))
        {
            return null;
        }

        List<DotRow> rows = band.DotRows(letterHeight, baseline);
        if (rows.Count != runs.Count)
        {
            return null;
        }

        string text = line.Text;
        var sb = new StringBuilder(text.Length + 16);
        bool changed = false;
        int at = 0;
        for (int r = 0; r < runs.Count; r++)
        {
            var (start, end) = runs[r];
            DotRow row = rows[r];
            sb.Append(text, at, start - at);
            if (Units(text, start, end) == row.Count && !MixedSpacing(text, start, end))
            {
                sb.Append(text, start, end - start);
            }
            else
            {
                // The spaces around the run follow the page, since the recogniser often loses the gap before or after
                // a run it could not count.
                while (sb.Length > 0 && sb[^1] == ' ')
                {
                    sb.Length--;
                }

                if (sb.Length > 0 && row.SpaceBefore)
                {
                    sb.Append(' ');
                }

                sb.Append('.', row.Count);
                int next = end;
                while (next < text.Length && text[next] == ' ')
                {
                    next++;
                }

                if (next < text.Length && row.SpaceAfter)
                {
                    sb.Append(' ');
                }

                end = next;
                changed = true;
            }

            at = end;
        }

        if (!changed)
        {
            return null;
        }

        sb.Append(text, at, text.Length - at);
        string counted = sb.ToString();
        var result = new RecognizedLine(counted, line.Confidence, line.Quad, Rewords(line, counted, rows, band), line.DetectionIndex, line.UpsideDown);
        if (rows.Exists(r => r.Count >= LeaderDots))
        {
            leaders[result] = rows.ConvertAll(r => new PageRow(r.Count, band.Left + r.Left, band.Left + r.Right, r.SpaceBefore, r.SpaceAfter));
        }

        return result;
    }

    /// <summary>
    /// The words of a line whose dot runs were counted: words without dots keep their places, and every word with a
    /// counted run spans the recogniser's word and the dots on the page.
    /// </summary>
    private static List<RecognizedWord> Rewords(RecognizedLine line, string text, List<DotRow> rows, Band band)
    {
        var (_, top, _, bottom) = line.Quad.Bounds();
        string[] tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var words = new List<RecognizedWord>(tokens.Length);
        var old = new Queue<RecognizedWord>(line.Words);
        int row = 0;
        foreach (string token in tokens)
        {
            if (token.Contains('.', StringComparison.Ordinal) && TextRuns(token).Count > 0 && row < rows.Count)
            {
                // A token with a run takes the old words it covers: those with dots or ellipses, and a word joined to
                // the run on either side.
                float x0 = band.Left + rows[row].Left, x1 = band.Left + rows[row].Right;
                float confidence = line.Confidence;
                row += TextRuns(token).Count;
                while (old.Count > 0 && Overlaps(old.Peek(), x0, x1, token))
                {
                    RecognizedWord w = old.Dequeue();
                    var (l, _, r, _) = w.Quad.Bounds();
                    x0 = MathF.Min(x0, l);
                    x1 = MathF.Max(x1, r);
                    confidence = MathF.Min(confidence, w.Confidence);
                }

                words.Add(new RecognizedWord(token, confidence, Rectangle(x0, top, x1, bottom)));
            }
            else if (old.Count > 0)
            {
                RecognizedWord w = old.Dequeue();
                words.Add(w with { Text = token });
            }
            else
            {
                words.Add(new RecognizedWord(token, line.Confidence, line.Quad));
            }
        }

        return words;

        static bool Overlaps(RecognizedWord word, float x0, float x1, string token)
        {
            var (l, _, r, _) = word.Quad.Bounds();
            return word.Text.AsSpan().IndexOfAny('.', '…') >= 0 ? r > x0 - 1 && l < x1 + 1 : r > x0 && l < x1 && token.Length > 0;
        }
    }

    /// <summary>
    /// Joins two lines on one row with a dot leader between them, which the detector does not see as text, into one
    /// line with the dots counted.
    /// </summary>
    private static List<RecognizedLine> JoinLeaders(PixelView page, List<RecognizedLine> lines, Dictionary<RecognizedLine, List<PageRow>> leaders, LineReader? read, CancellationToken cancellationToken)
    {
        var used = new bool[lines.Count];
        var joined = new Dictionary<int, RecognizedLine>();
        for (int a = 0; a < lines.Count; a++)
        {
            if (used[a] || !Horizontal(lines[a]))
            {
                continue;
            }

            RecognizedLine current = lines[a];
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int b = RightNeighbour(lines, used, current, a);
                if (b < 0)
                {
                    if (NothingToTheRight(lines, used, current, a))
                    {
                        current = TrailingLeader(page, current, leaders, read, cancellationToken) ?? current;
                    }

                    break;
                }

                RecognizedLine? merged = JoinIfLeader(page, current, lines[b], leaders);
                if (merged is null)
                {
                    break;
                }

                used[b] = true;
                current = merged;
            }

            if (!ReferenceEquals(current, lines[a]))
            {
                joined[a] = current;
            }
        }

        var result = new List<RecognizedLine>(lines.Count);
        for (int i = 0; i < lines.Count; i++)
        {
            if (!used[i])
            {
                result.Add(joined.TryGetValue(i, out RecognizedLine? line) ? line : lines[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// The nearest horizontal line to the right of <paramref name="line"/> on its row, of about its height, with no
    /// other line in the gap between them; -1 when there is none.
    /// </summary>
    private static int RightNeighbour(List<RecognizedLine> lines, bool[] used, RecognizedLine line, int self)
    {
        var (_, top, right, bottom) = line.Quad.Bounds();
        float height = bottom - top;
        int best = -1;
        float bestLeft = float.MaxValue;
        for (int i = 0; i < lines.Count; i++)
        {
            if (i == self || used[i] || !Horizontal(lines[i]))
            {
                continue;
            }

            var (l, t, r, b) = lines[i].Quad.Bounds();
            float overlap = MathF.Min(bottom, b) - MathF.Max(top, t);
            float h = b - t;
            // A lone page number gets a lower box than a title with ascenders and descenders, and the recogniser's line
            // may end inside the leader where the next box starts.
            if (l > right - (0.3f * height) && (l + r) / 2 > right && overlap >= 0.6f * MathF.Min(height, h) && h < 2.2f * height && height < 2.2f * h && l < bestLeft)
            {
                best = i;
                bestLeft = l;
            }
        }

        if (best < 0)
        {
            return -1;
        }

        // Any other line reaching into the gap means the two are not a title and its page number.
        var (_, bt, _, bb) = lines[best].Quad.Bounds();
        float gapTop = MathF.Min(top, bt), gapBottom = MathF.Max(bottom, bb);
        for (int i = 0; i < lines.Count; i++)
        {
            if (i == self || i == best || used[i])
            {
                continue;
            }

            var (l, t, r, b) = lines[i].Quad.Bounds();
            // Detected boxes are grown around the text, so the lines above and below reach into the band; only a line
            // whose middle is on this row counts.
            float middle = (t + b) / 2;
            if (r > right && l < bestLeft && middle > gapTop && middle < gapBottom)
            {
                return -1;
            }
        }

        return best;
    }

    /// <summary>Whether no other line has its middle on this line's row to the right of it.</summary>
    private static bool NothingToTheRight(List<RecognizedLine> lines, bool[] used, RecognizedLine line, int self)
    {
        var (_, top, right, bottom) = line.Quad.Bounds();
        for (int i = 0; i < lines.Count; i++)
        {
            if (i == self || used[i])
            {
                continue;
            }

            var (l, t, r, b) = lines[i].Quad.Bounds();
            float middle = (t + b) / 2;
            if (r > right && middle > top && middle < bottom)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// A dot leader after the last line on its row, which the detector did not see, and the page number after it,
    /// which the detector can miss as well on a poor scan: the number is read on its own by <paramref name="read"/>.
    /// A leader with nothing after it ends the line. Null when there is no leader.
    /// </summary>
    private static RecognizedLine? TrailingLeader(PixelView page, RecognizedLine line, Dictionary<RecognizedLine, List<PageRow>> leaders, LineReader? read, CancellationToken cancellationToken)
    {
        var (ll, top, lr, bottom) = line.Quad.Bounds();
        float height = bottom - top;
        if (!AnyInk(page, lr, top, lr + (3 * height), bottom))
        {
            return null;
        }

        Band? band = Band.Read(page, ll, top, page.Width, bottom);
        if (band is null || !band.FindLetters(out float letterHeight, out float baseline))
        {
            return null;
        }

        float end = lr - band.Left;
        DotRow? found = null;
        foreach (DotRow candidate in band.DotRows(letterHeight, baseline))
        {
            if (candidate.Count >= 3 && candidate.Right > end + letterHeight && candidate.Left <= end + (3 * letterHeight))
            {
                found = candidate;
                break;
            }
        }

        if (found is not { } row)
        {
            return null;
        }

        // The marks right after the leader, close together, are its page number.
        RecognizedLine? number = null;
        if (read is not null && band.MarksAfter(row.Right, 2.5f * letterHeight, 0.8f * letterHeight, baseline, letterHeight) is { } cluster)
        {
            float pad = 0.25f * height;
            var quad = Rectangle(band.Left + cluster.Left - pad, top, band.Left + cluster.Right + pad, bottom);
            List<RecognizedLine> readings = read(page, [new DetectedQuad(quad, 1)], cancellationToken);
            if (readings.Count == 0)
            {
                return null;
            }

            number = readings[0];
        }

        var (leftText, leftWords) = WithoutDots(line, band.Left + row.Left, atEnd: true);
        if (leftText.Length == 0)
        {
            return null;
        }

        var dots = new string('.', row.Count);
        float leaderLeft = band.Left + row.Left, leaderRight = band.Left + row.Right;
        var words = new List<RecognizedWord>(leftWords) { new(dots, line.Confidence, Rectangle(leaderLeft, top, leaderRight, bottom)) };
        string text = leftText + " " + dots;
        float right = leaderRight, confidence = line.Confidence;
        if (number is not null)
        {
            words.AddRange(number.Words);
            text += " " + number.Text;
            right = MathF.Max(right, number.Quad.Bounds().Right);
            confidence = ((line.Confidence * (leftText.Length + dots.Length)) + (number.Confidence * number.Text.Length)) / (leftText.Length + dots.Length + number.Text.Length);
        }

        var result = new RecognizedLine(text, confidence, Rectangle(ll, top, right, bottom), words, line.DetectionIndex, line.UpsideDown);
        var rows = new List<PageRow>();
        if (leaders.Remove(line, out List<PageRow>? before))
        {
            rows.AddRange(before.FindAll(r => r.Right < leaderLeft));
        }

        rows.Add(new PageRow(row.Count, leaderLeft, leaderRight, true, number is not null));
        leaders[result] = rows;
        return result;
    }

    /// <summary>Whether a rectangle of the page has any dark pixel, a quick test before the marks are found.</summary>
    private static bool AnyInk(PixelView page, float left, float top, float right, float bottom)
    {
        int x0 = Math.Clamp((int)left, 0, page.Width), x1 = Math.Clamp((int)right, 0, page.Width);
        int y0 = Math.Clamp((int)top, 0, page.Height), y1 = Math.Clamp((int)bottom, 0, page.Height);
        ReadOnlySpan<byte> data = page.Data;
        int channels = page.Channels;
        for (int y = y0; y < y1; y++)
        {
            for (int x = x0; x < x1; x++)
            {
                // The green channel (or gray) is enough to tell ink from paper.
                int p = ((y * page.Width) + x) * channels;
                if (data[channels == 1 ? p : p + 1] < 112)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static RecognizedLine? JoinIfLeader(PixelView page, RecognizedLine left, RecognizedLine right, Dictionary<RecognizedLine, List<PageRow>> leaders)
    {
        var (ll, lt, lr, lb) = left.Quad.Bounds();
        var (rl, rt, rr, rb) = right.Quad.Bounds();
        float top = MathF.Min(lt, rt), bottom = MathF.Max(lb, rb);

        // The gap alone first, which is cheap: most gaps between two lines on a row hold no row of small marks. Boxes
        // that touch or overlap, which happens when the recogniser's line ends inside the leader, go straight on.
        float height = bottom - top;
        if (rl - lr >= 0.5f * height && (Band.Read(page, lr, top, rl, bottom) is not { } gap || gap.SmallMarks(0.3f * height) < 2))
        {
            return null;
        }

        Band? band = Band.Read(page, ll, top, rr, bottom);
        if (band is null || !band.FindLetters(out float letterHeight, out float baseline))
        {
            return null;
        }

        // The leader is the run of dots across the meeting point of the two boxes, wherever it starts and ends: the
        // detector often keeps a few of its dots at the end of the title or the start of the page number.
        DotRow? leader = band.Leader(letterHeight, baseline, ((lr + rl) / 2) - band.Left);
        if (leader is not { } row)
        {
            return null;
        }

        var (leftText, leftWords) = WithoutDots(left, band.Left + row.Left, atEnd: true);
        var (rightText, rightWords) = WithoutDots(right, band.Left + row.Right, atEnd: false);
        if (leftText.Length == 0 || rightText.Length == 0)
        {
            return null;
        }

        var dots = new string('.', row.Count);
        var words = new List<RecognizedWord>(leftWords.Count + rightWords.Count + 1);
        words.AddRange(leftWords);
        words.Add(new RecognizedWord(dots, (left.Confidence + right.Confidence) / 2, Rectangle(band.Left + row.Left, top, band.Left + row.Right, bottom)));
        words.AddRange(rightWords);
        string text = leftText + " " + dots + " " + rightText;
        float confidence = ((left.Confidence * leftText.Length) + (right.Confidence * rightText.Length) + ((left.Confidence + right.Confidence) / 2 * dots.Length)) / text.Length;
        var joined = new RecognizedLine(text, confidence, Rectangle(ll, top, rr, bottom), words, left.DetectionIndex, upsideDown: false);
        float leaderLeft = band.Left + row.Left, leaderRight = band.Left + row.Right;
        var rows = new List<PageRow>();
        if (leaders.Remove(left, out List<PageRow>? before))
        {
            rows.AddRange(before.FindAll(r => r.Right < leaderLeft));
        }

        rows.Add(new PageRow(row.Count, leaderLeft, leaderRight, true, true));
        if (leaders.Remove(right, out List<PageRow>? after))
        {
            rows.AddRange(after.FindAll(r => r.Left > leaderRight));
        }

        leaders[joined] = rows;
        return joined;
    }

    /// <summary>
    /// A line's text and words without the dots at its end (or start) that lie past <paramref name="edge"/>, where
    /// the leader begins (or ends): words of nothing but dots there are dropped, and dots are trimmed off a word they
    /// are joined to.
    /// </summary>
    private static (string Text, List<RecognizedWord> Words) WithoutDots(RecognizedLine line, float edge, bool atEnd)
    {
        var words = new List<RecognizedWord>(line.Words);
        while (words.Count > 0)
        {
            int i = atEnd ? words.Count - 1 : 0;
            RecognizedWord w = words[i];
            var (l, _, r, _) = w.Quad.Bounds();
            bool past = atEnd ? r > edge : l < edge;
            string trimmed = atEnd ? w.Text.TrimEnd('.', '…') : w.Text.TrimStart('.', '…');
            if (trimmed.Length == w.Text.Length || (!past && trimmed.Length > 0))
            {
                break;
            }

            if (trimmed.Length == 0)
            {
                words.RemoveAt(i);
                continue;
            }

            words[i] = w with { Text = trimmed };
            break;
        }

        var sb = new StringBuilder();
        foreach (RecognizedWord w in words)
        {
            if (sb.Length > 0)
            {
                sb.Append(' ');
            }

            sb.Append(w.Text);
        }

        return (words.Count == line.Words.Count && sb.ToString() == line.Text.Replace("  ", " ", StringComparison.Ordinal) ? line.Text : sb.ToString(), words);
    }

    private static Quad Rectangle(float left, float top, float right, float bottom) =>
        new(new Vector2(left, top), new Vector2(right, top), new Vector2(right, bottom), new Vector2(left, bottom));

    /// <summary>A run of dots found on the page, in band pixels.</summary>
    private readonly record struct DotRow(int Count, float Left, float Right, bool SpaceBefore, bool SpaceAfter);

    /// <summary>A connected mark of ink, in band pixels.</summary>
    private readonly record struct Mark(int Left, int Top, int Right, int Bottom, int Pixels)
    {
        public int Width => Right - Left + 1;

        public int Height => Bottom - Top + 1;
    }

    /// <summary>The ink of a rectangle of the page as connected marks.</summary>
    private sealed class Band
    {
        private readonly List<Mark> _marks;

        private Band(float left, List<Mark> marks)
        {
            Left = left;
            _marks = marks;
        }

        /// <summary>The band's left edge on the page.</summary>
        public float Left { get; }

        /// <summary>
        /// Reads the marks of a rectangle of the page, dark on light at a threshold between the band's darkest and
        /// lightest pixels (Otsu's); null for a band without contrast or one that is too large.
        /// </summary>
        public static Band? Read(PixelView page, float left, float top, float right, float bottom)
        {
            int x0 = Math.Clamp((int)MathF.Floor(left), 0, page.Width), x1 = Math.Clamp((int)MathF.Ceiling(right), 0, page.Width);
            int y0 = Math.Clamp((int)MathF.Floor(top), 0, page.Height), y1 = Math.Clamp((int)MathF.Ceiling(bottom), 0, page.Height);
            int w = x1 - x0, h = y1 - y0;
            if (w < 2 || h < 4 || (long)w * h > MaxBandPixels)
            {
                return null;
            }

            var gray = new byte[w * h];
            Span<int> histogram = stackalloc int[256];
            histogram.Clear();
            ReadOnlySpan<byte> data = page.Data;
            int channels = page.Channels;
            for (int y = 0; y < h; y++)
            {
                int row = ((y0 + y) * page.Width) + x0;
                for (int x = 0; x < w; x++)
                {
                    int p = (row + x) * channels;
                    int v = channels == 1 ? data[p] : ((data[p] * 299) + (data[p + 1] * 587) + (data[p + 2] * 114)) / 1000;
                    gray[(y * w) + x] = (byte)v;
                    histogram[v]++;
                }
            }

            // A 3 x 3 mean first, so the noise and ragged edges of a scan do not break a dot into specks.
            gray = Smooth(gray, w, h);
            histogram.Clear();
            foreach (byte v in gray)
            {
                histogram[v]++;
            }

            int threshold = Otsu(histogram, w * h, out int darkest, out int lightest);
            if (lightest - darkest < 48)
            {
                return null;
            }

            return new Band(x0, Marks(gray, w, h, threshold));
        }

        /// <summary>
        /// The height of the band's letters (the median height of marks of at least a third of the tallest's) and
        /// their baseline (the median bottom of those marks), in band pixels; false for a band without letters.
        /// </summary>
        public bool FindLetters(out float letterHeight, out float baseline)
        {
            letterHeight = baseline = 0;
            int tallest = 0;
            foreach (Mark m in _marks)
            {
                tallest = Math.Max(tallest, m.Height);
            }

            var heights = new List<int>();
            var bottoms = new List<int>();
            foreach (Mark m in _marks)
            {
                if (m.Height * 3 >= tallest && m.Height >= 4)
                {
                    heights.Add(m.Height);
                    bottoms.Add(m.Bottom);
                }
            }

            if (heights.Count < 2)
            {
                return false;
            }

            heights.Sort();
            bottoms.Sort();
            letterHeight = heights[heights.Count / 2];
            baseline = bottoms[bottoms.Count / 2];
            return true;
        }

        /// <summary>The number of marks at most <paramref name="size"/> pixels wide and high, and at least two.</summary>
        public int SmallMarks(float size)
        {
            int count = 0;
            foreach (Mark m in _marks)
            {
                if (m.Width <= size && m.Height <= size && (m.Width >= 2 || m.Height >= 2))
                {
                    count++;
                }
            }

            return count;
        }

        /// <summary>
        /// The marks on the row of letters that start within <paramref name="reach"/> after <paramref name="x"/>, as one
        /// span of marks no more than <paramref name="gap"/> apart; null when there are none.
        /// </summary>
        public (float Left, float Right)? MarksAfter(float x, float reach, float gap, float baseline, float letterHeight)
        {
            float left = float.NaN, right = x;
            foreach (Mark m in Sorted(letterHeight))
            {
                if (m.Left <= x || m.Bottom < baseline - (1.5f * letterHeight) || m.Top > baseline + (0.3f * letterHeight))
                {
                    continue;
                }

                if (float.IsNaN(left))
                {
                    if (m.Left - x > reach)
                    {
                        return null;
                    }

                    left = m.Left;
                    right = m.Right + 1;
                }
                else if (m.Left - right <= gap)
                {
                    right = MathF.Max(right, m.Right + 1);
                }
                else
                {
                    break;
                }
            }

            return float.IsNaN(left) ? null : (left, right);
        }

        /// <summary>The runs of two or more dots in the band, left to right.</summary>
        public List<DotRow> DotRows(float letterHeight, float baseline)
        {
            List<Mark> marks = Sorted(letterHeight);
            bool[] dot = DotLike(marks, letterHeight, baseline);
            var rows = new List<DotRow>();
            int i = 0;
            while (i < marks.Count)
            {
                if (!dot[i])
                {
                    i++;
                    continue;
                }

                int j = i;
                while (j + 1 < marks.Count && dot[j + 1] && Centre(marks[j + 1]) - Centre(marks[j]) <= 3 * letterHeight)
                {
                    j++;
                }

                if (j > i && Even(marks, i, j, letterHeight))
                {
                    bool before = i == 0 || marks[i].Left - marks[i - 1].Right > 0.4f * letterHeight;
                    bool after = j == marks.Count - 1 || marks[j + 1].Left - marks[j].Right > 0.4f * letterHeight;
                    rows.Add(new DotRow(j - i + 1, marks[i].Left, marks[j].Right + 1, before, after));
                }

                i = j + 1;
            }

            return rows;
        }

        /// <summary>
        /// The run of at least three dots that crosses <paramref name="x"/>, with letters on both sides of it; null when
        /// there is none.
        /// </summary>
        public DotRow? Leader(float letterHeight, float baseline, float x)
        {
            foreach (DotRow row in DotRows(letterHeight, baseline))
            {
                if (row.Count >= 3 && row.Left <= x && row.Right >= x && row.Left > 0)
                {
                    return row;
                }
            }

            return null;
        }

        // The marks without specks too small to be a dot, left to right.
        private List<Mark> Sorted(float letterHeight)
        {
            // Specks of noise in a scan are smaller than a dot: marks of less than a quarter of the median small
            // mark's pixels are dropped, as well as anything under a twelfth of the letter height.
            float speck = MathF.Max(2, 0.08f * letterHeight);
            var small = new List<int>();
            foreach (Mark m in _marks)
            {
                if (Math.Max(m.Width, m.Height) <= 0.45f * letterHeight && (m.Width >= speck || m.Height >= speck))
                {
                    small.Add(m.Pixels);
                }
            }

            small.Sort();
            int minPixels = small.Count == 0 ? 0 : small[small.Count / 2] / 4;
            var marks = new List<Mark>(_marks.Count);
            foreach (Mark m in _marks)
            {
                if ((m.Width >= speck || m.Height >= speck) && m.Pixels >= minPixels)
                {
                    marks.Add(m);
                }
            }

            marks.Sort((a, b) => a.Left != b.Left ? a.Left.CompareTo(b.Left) : a.Top.CompareTo(b.Top));
            return marks;
        }

        /// <summary>
        /// Which marks are dots: small, round and solid, sitting on the baseline, with no other mark above or below
        /// them, which rules out the dots of i, j, colons, semicolons and question and exclamation marks.
        /// </summary>
        private static bool[] DotLike(List<Mark> marks, float letterHeight, float baseline)
        {
            var dot = new bool[marks.Count];
            for (int i = 0; i < marks.Count; i++)
            {
                Mark m = marks[i];
                int big = Math.Max(m.Width, m.Height), small = Math.Min(m.Width, m.Height);
                dot[i] = big <= 0.45f * letterHeight && big <= 2 * small && m.Pixels * 2 >= m.Width * m.Height &&
                    MathF.Abs(m.Bottom - baseline) <= 0.25f * letterHeight;
            }

            for (int i = 0; i < marks.Count; i++)
            {
                for (int j = i + 1; j < marks.Count && marks[j].Left <= marks[i].Right; j++)
                {
                    // Marks are sorted by their left edge, so every mark overlapping this one's columns follows it.
                    dot[i] = dot[j] = false;
                }

                for (int j = i - 1; j >= 0; j--)
                {
                    if (marks[j].Right >= marks[i].Left)
                    {
                        dot[i] = dot[j] = false;
                    }
                }
            }

            return dot;
        }

        // Dots at an even pitch, and of about one size.
        private static bool Even(List<Mark> marks, int first, int last, float letterHeight)
        {
            float minPitch = float.MaxValue, maxPitch = 0;
            int minPixels = int.MaxValue, maxPixels = 0;
            for (int k = first; k <= last; k++)
            {
                minPixels = Math.Min(minPixels, marks[k].Pixels);
                maxPixels = Math.Max(maxPixels, marks[k].Pixels);
                if (k > first)
                {
                    float pitch = Centre(marks[k]) - Centre(marks[k - 1]);
                    minPitch = MathF.Min(minPitch, pitch);
                    maxPitch = MathF.Max(maxPitch, pitch);
                }
            }

            if (maxPixels > 3 * minPixels)
            {
                return false;
            }

            return last == first + 1 ? maxPitch <= 1.5f * letterHeight : maxPitch <= 1.6f * minPitch && maxPitch <= 3 * letterHeight;
        }

        private static byte[] Smooth(byte[] gray, int w, int h)
        {
            var result = new byte[gray.Length];
            for (int y = 0; y < h; y++)
            {
                int ya = Math.Max(0, y - 1), yb = Math.Min(h - 1, y + 1);
                for (int x = 0; x < w; x++)
                {
                    int xa = Math.Max(0, x - 1), xb = Math.Min(w - 1, x + 1);
                    int sum = 0;
                    for (int yy = ya; yy <= yb; yy++)
                    {
                        for (int xx = xa; xx <= xb; xx++)
                        {
                            sum += gray[(yy * w) + xx];
                        }
                    }

                    result[(y * w) + x] = (byte)(sum / ((yb - ya + 1) * (xb - xa + 1)));
                }
            }

            return result;
        }

        private static float Centre(Mark m) => (m.Left + m.Right) * 0.5f;

        private static int Otsu(ReadOnlySpan<int> histogram, int total, out int darkest, out int lightest)
        {
            darkest = 0;
            while (darkest < 255 && histogram[darkest] == 0)
            {
                darkest++;
            }

            lightest = 255;
            while (lightest > 0 && histogram[lightest] == 0)
            {
                lightest--;
            }

            double sum = 0;
            for (int i = 0; i < 256; i++)
            {
                sum += (double)i * histogram[i];
            }

            double sumBelow = 0, best = -1;
            int below = 0, threshold = (darkest + lightest) / 2;
            for (int t = 0; t < 256; t++)
            {
                below += histogram[t];
                if (below == 0)
                {
                    continue;
                }

                int above = total - below;
                if (above == 0)
                {
                    break;
                }

                sumBelow += (double)t * histogram[t];
                double meanBelow = sumBelow / below, meanAbove = (sum - sumBelow) / above;
                double between = (double)below * above * (meanBelow - meanAbove) * (meanBelow - meanAbove);
                if (between > best)
                {
                    best = between;
                    threshold = t;
                }
            }

            return threshold;
        }

        // The 8-connected marks of pixels at or below the threshold.
        private static List<Mark> Marks(byte[] gray, int w, int h, int threshold)
        {
            var marks = new List<Mark>();
            var seen = new bool[gray.Length];
            var stack = new Stack<int>();
            for (int start = 0; start < gray.Length; start++)
            {
                if (seen[start] || gray[start] > threshold)
                {
                    continue;
                }

                int left = int.MaxValue, top = int.MaxValue, right = -1, bottom = -1, pixels = 0;
                seen[start] = true;
                stack.Push(start);
                while (stack.Count > 0)
                {
                    int p = stack.Pop();
                    int x = p % w, y = p / w;
                    pixels++;
                    left = Math.Min(left, x);
                    right = Math.Max(right, x);
                    top = Math.Min(top, y);
                    bottom = Math.Max(bottom, y);
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int yy = y + dy;
                        if (yy < 0 || yy >= h)
                        {
                            continue;
                        }

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int xx = x + dx;
                            int q = (yy * w) + xx;
                            if (xx >= 0 && xx < w && !seen[q] && gray[q] <= threshold)
                            {
                                seen[q] = true;
                                stack.Push(q);
                            }
                        }
                    }
                }

                marks.Add(new Mark(left, top, right, bottom, pixels));
            }

            return marks;
        }
    }
}
