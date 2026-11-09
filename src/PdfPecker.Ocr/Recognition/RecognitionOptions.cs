namespace PdfPecker.Ocr.Recognition;

/// <summary>Settings of line orientation, recognition and word splitting.</summary>
internal sealed record RecognitionOptions
{
    /// <summary>Whether lines are checked for being upside down with the orientation classifier.</summary>
    public bool DetectRotation { get; init; } = true;

    /// <summary>A line is turned when its upside-down probability is above this (rapidocr's <c>cls_thresh</c>).</summary>
    public float OrientationThreshold { get; init; } = 0.9f;

    /// <summary>
    /// Whether long lines are classified on three windows at their natural aspect instead of squeezed whole into the
    /// classifier's input as rapidocr does. See <see cref="OrientationClassifier"/>.
    /// </summary>
    public bool OrientationWindows { get; init; } = true;

    /// <summary>Whether the page's majority orientation raises the bar for turning a line against it.</summary>
    public bool PageVote { get; init; } = true;

    /// <summary>
    /// Windows per orientation network run. The classifier is a small CNN with no attention across the batch, so the
    /// batch size does not change its outputs, only the number of runs.
    /// </summary>
    public int OrientationBatchSize { get; init; } = 16;

    /// <summary>
    /// Lines per recognition network run (rapidocr's <c>rec_batch_num</c>). Lines are sorted by aspect ratio and a
    /// batch is padded to its widest line, and the network's attention sees the padding, so the batch size and order
    /// change results slightly; 6 keeps them equal to rapidocr's.
    /// </summary>
    public int BatchSize { get; init; } = 6;

    /// <summary>
    /// The widest recognition input in pixels (at the height of 48). A longer line is squeezed to this width, which
    /// bounds the memory and time of one line however long a detected box is. 3,200 is 67 times the height, more
    /// than an A4 line of 8 point text at 300 dpi needs.
    /// </summary>
    public int MaxInputWidth { get; init; } = 3200;

    /// <summary>Lines whose confidence is below this are dropped (rapidocr's <c>text_score</c>).</summary>
    public float MinConfidence { get; init; } = 0.5f;

    /// <summary>
    /// Where two characters that are not in a script without spaces lie further apart than this many time steps (of
    /// 8 pixels at the input height of 48) with no space between them, the line gets a space and the word ends there.
    /// rapidocr ends a word box at the same distance but keeps the text joined, as in "EUR 1,234.56VAT 25%".
    /// </summary>
    public int WordGapSteps { get; init; } = 5;

    /// <summary>The classes the decoder may choose from, or null for all; see <see cref="CharacterSet.Subset"/>.</summary>
    public int[]? CharacterSubset { get; init; }

    /// <summary>The settings rapidocr 3.10 uses, for comparisons with it.</summary>
    public static RecognitionOptions Rapidocr { get; } = new() { OrientationWindows = false, PageVote = false, WordGapSteps = int.MaxValue };
}
