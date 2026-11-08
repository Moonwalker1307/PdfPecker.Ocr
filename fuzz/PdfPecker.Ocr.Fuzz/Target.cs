namespace PdfPecker.Ocr.Fuzz;

/// <summary>One thing to fuzz: how to run an input through it, where its seeds come from and what it may throw.</summary>
internal sealed class Target
{
    /// <summary>The name given to --target; lowercase words joined by hyphens, such as ocr-image.</summary>
    public required string Name { get; init; }

    /// <summary>The file extension of saved findings, with its dot, so a finding opens in the right viewer.</summary>
    public required string Extension { get; init; }

    /// <summary>Runs one input. It must pass the token on, so that a slow input can be stopped instead of abandoned.</summary>
    public required Action<byte[], CancellationToken> Run { get; init; }

    /// <summary>The inputs mutation starts from: files in the repository, or inputs generated at start-up.</summary>
    public required Func<SeedContext, IEnumerable<(string Name, byte[] Data)>> Seeds { get; init; }

    /// <summary>True for text formats, which also get token-level mutations and token-level minimising.</summary>
    public bool Text { get; init; }

    /// <summary>Inputs are cut to this size, and seeds above it are left out.</summary>
    public int MaxSize { get; init; } = 512 * 1024;

    /// <summary>Words that token-level mutations insert into text inputs.</summary>
    public string[] Dictionary { get; init; } = Array.Empty<string>();

    /// <summary>Opening and closing pairs that token-level mutations nest deeply in text inputs.</summary>
    public (string Open, string Close)[] Nesters { get; init; } = Array.Empty<(string, string)>();

    /// <summary>
    /// A per-input time limit for targets whose legitimate worst case at <see cref="MaxSize"/> is slower than the
    /// default --timeout, such as running the models on a large page. It only raises the limit; a longer --timeout still
    /// wins.
    /// </summary>
    public TimeSpan MinTimeout { get; init; }

    /// <summary>The time limit for one input: --timeout, or <see cref="MinTimeout"/> when that is longer.</summary>
    public TimeSpan TimeoutFor(TimeSpan requested) => requested > MinTimeout ? requested : MinTimeout;

    /// <summary>Further exceptions this target's API documents, such as <see cref="ArgumentException"/> for a pixel buffer
    /// that is smaller than its stated size.</summary>
    public Func<Exception, bool>? ExtraExpected { get; init; }

    /// <summary>
    /// The failure modes the public API documents for bad input. Anything else escaping is a bug: an
    /// <see cref="IndexOutOfRangeException"/>, a <see cref="NullReferenceException"/> or an
    /// <see cref="OverflowException"/> from a damaged image or PDF means a check is missing.
    /// <see cref="PdfPeckerException"/> covers <see cref="PdfLimitException"/> and <see cref="PdfPasswordException"/>,
    /// which the PdfPecker package throws for limits and passwords, and the OCR package's own exceptions, which derive
    /// from it.
    /// </summary>
    public bool IsExpected(Exception ex)
    {
        if (ex is FormatException or InvalidDataException or PdfPeckerException or OperationCanceledException or NotSupportedException)
            return true;
        return ExtraExpected?.Invoke(ex) ?? false;
    }
}
