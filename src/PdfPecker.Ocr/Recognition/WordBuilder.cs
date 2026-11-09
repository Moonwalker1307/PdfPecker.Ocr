using System.Text;
using PdfPecker.Ocr.Detection;

namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// Splits a decoded line into words and places each word on the page, after rapidocr's <c>get_word_info</c> and
/// <c>CalRecBoxes</c> (Apache-2.0, https://github.com/RapidAI/RapidOCR, after PaddleOCR): a word ends at a space, at
/// a change between scripts written with and without spaces, and at a gap of more than
/// <see cref="RecognitionOptions.WordGapSteps"/> time steps. Unlike rapidocr, a gap the recogniser did not mark with
/// a space also gets one in the line's text. Character positions come from the time step each character was emitted
/// at; a word spans from its first character's centre to its last one's, widened by half the line's mean character
/// pitch on either side, across the full height of the line, and is mapped through the crop's perspective onto the
/// page, so words of tilted and rotated lines get tilted and rotated quads.
/// </summary>
internal static class WordBuilder
{
    /// <summary>
    /// The geometry that maps time steps to the strip: <paramref name="inputWidth"/> is the batch's input width the
    /// network saw, <paramref name="resizedWidth"/> the part of it the line filled.
    /// </summary>
    public static (string Text, List<RecognizedWord> Words) Build(DecodedLine line, LineCrop crop, CharacterSet characters, int inputWidth, int resizedWidth, int gapSteps)
    {
        int n = line.Classes.Length;
        var words = new List<RecognizedWord>();
        if (n == 0 || line.Steps == 0)
        {
            return (string.Empty, words);
        }

        // Strip pixels per time step. A line squeezed into the widest input has its steps closer together than at its
        // natural aspect, so its gaps are scaled back before they are compared with the threshold in steps. rapidocr
        // squeezes the widest line of every batch by up to a pixel (int against ceil), which is no squeeze here.
        double stepToStrip = (double)inputWidth / line.Steps * crop.CropWidth / resizedWidth;
        double squeeze = Math.Ceiling(48.0 * crop.CropWidth / crop.CropHeight) / resizedWidth;
        if (squeeze < 1.02)
        {
            squeeze = 1;
        }

        var text = new StringBuilder(n + 8);
        var starts = new List<int>();
        var ends = new List<int>();
        int start = -1, last = -1;
        for (int i = 0; i < n; i++)
        {
            string ch = characters[line.Classes[i]];
            if (string.IsNullOrWhiteSpace(ch))
            {
                Close();
                text.Append(ch);
                continue;
            }

            if (start >= 0)
            {
                string prev = characters[line.Classes[last]];
                bool unspaced = CharacterSet.IsUnspaced(ch), prevUnspaced = CharacterSet.IsUnspaced(prev);
                double gap = (line.Columns[i] - line.Columns[last]) * squeeze;
                if (unspaced || prevUnspaced)
                {
                    // Scripts without spaces give each character a word of its own, as rapidocr's boxes do.
                    Close();
                }
                else if (gap > gapSteps)
                {
                    Close();
                    text.Append(' ');
                }
            }

            if (start < 0)
            {
                start = i;
            }

            last = i;
            text.Append(ch);
        }

        Close();

        // The mean character pitch from words of two or more characters, else the strip's width over the characters,
        // as rapidocr's calc_avg_char_width and calc_all_char_avg_width estimate it.
        double pitchSum = 0;
        int pitchCount = 0;
        for (int w = 0; w < starts.Count; w++)
        {
            int a = starts[w], b = ends[w];
            if (b > a)
            {
                pitchSum += (line.Columns[b] - line.Columns[a]) * stepToStrip / (b - a);
                pitchCount++;
            }
        }

        double pitch = pitchCount > 0 ? pitchSum / pitchCount : crop.CropWidth / (double)Math.Max(1, line.Text.Length);
        var x0 = new double[starts.Count];
        var x1 = new double[starts.Count];
        for (int w = 0; w < starts.Count; w++)
        {
            double first = (line.Columns[starts[w]] + 0.5) * stepToStrip, lastCentre = (line.Columns[ends[w]] + 0.5) * stepToStrip;
            x0[w] = Math.Clamp(first - (pitch / 2), 0, crop.CropWidth);
            x1[w] = Math.Clamp(lastCentre + (pitch / 2), 0, crop.CropWidth);
        }

        // rapidocr's adjust_box_overlap: neighbours that overlap meet halfway.
        for (int w = 0; w + 1 < starts.Count; w++)
        {
            if (x1[w] > x0[w + 1])
            {
                double d = x1[w] - x0[w + 1];
                x1[w] -= d / 2;
                x0[w + 1] += d / 2;
            }
        }

        var sb = new StringBuilder();
        for (int w = 0; w < starts.Count; w++)
        {
            sb.Clear();
            double conf = 0;
            for (int i = starts[w]; i <= ends[w]; i++)
            {
                sb.Append(characters[line.Classes[i]]);
                conf += line.Probabilities[i];
            }

            int h = crop.CropHeight;
            var quad = new Quad(crop.EdgeToPage(x0[w], 0), crop.EdgeToPage(x1[w], 0), crop.EdgeToPage(x1[w], h), crop.EdgeToPage(x0[w], h));
            words.Add(new RecognizedWord(sb.ToString(), (float)(conf / (ends[w] - starts[w] + 1)), quad));
        }

        return (text.ToString(), words);

        void Close()
        {
            if (start >= 0)
            {
                starts.Add(start);
                ends.Add(last);
                start = -1;
            }
        }
    }
}
