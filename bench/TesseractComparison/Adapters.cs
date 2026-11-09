namespace TesseractComparison;

enum PixelFormat { Gray8 = 1, Rgb24 = 3, Rgba32 = 4 }

/// <summary>What one recognition returns.</summary>
sealed class OcrResult
{
    /// <summary>The recognised text, lines separated by line feeds, in the engine's reading order.</summary>
    public string Text { get; init; } = "";

    /// <summary>The words with their boxes (page pixels, top-left origin) and confidences in [0, 1].</summary>
    public List<Box> Words { get; init; } = [];

    /// <summary>
    /// Set by engines that run in another process: the CPU time the call took there, and that process's peak resident set
    /// since <see cref="IOcrEngineAdapter.BeginMeasurement"/>, which the harness cannot measure itself.
    /// </summary>
    public double? ExternalCpuMs { get; init; }
    public long? ExternalPeakKb { get; init; }

    /// <summary>For engines in another process: that process's peak resident set over its whole life so far.</summary>
    public long? ExternalProcessPeakKb { get; init; }
}

/// <summary>
/// The narrow interface every engine is measured through, so that Tesseract, rapidocr and PdfPecker.Ocr are timed by the
/// same code. An adapter loads its models in <see cref="Initialize"/> and <see cref="Prepare"/> (neither is timed, and both
/// are reported separately) and then recognises one page per <see cref="Recognize"/> call.
/// </summary>
interface IOcrEngineAdapter : IDisposable
{
    string Name { get; }

    /// <summary>Versions and settings, for the report header.</summary>
    string Description { get; }

    /// <summary>True when the engine runs in this process, so the harness can measure its CPU, memory and allocations.</summary>
    bool InProcess { get; }

    /// <summary>True for an engine that allocates managed memory worth reporting.</summary>
    bool Managed { get; }

    void Initialize(int threads);

    /// <summary>Called before the first recognition of every page: lets an engine switch language or page mode.</summary>
    void Prepare(CorpusPage page);

    /// <summary>Called once before the timed runs of a page: an out-of-process engine resets its peak memory here.</summary>
    void BeginMeasurement() { }

    /// <summary>For engines in another process: its resident set after loading the models, in kilobytes (0 when unknown).</summary>
    long ExternalRssAfterLoadKb => 0;

    OcrResult Recognize(ReadOnlySpan<byte> pixels, int width, int height, PixelFormat format);
}

static class PixelConversion
{
    /// <summary>The page as RGBA, for engines that only take that.</summary>
    public static byte[] ToRgba(ReadOnlySpan<byte> pixels, int width, int height, PixelFormat format)
    {
        var rgba = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            switch (format)
            {
                case PixelFormat.Gray8: rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = pixels[i]; rgba[i * 4 + 3] = 255; break;
                case PixelFormat.Rgb24: rgba[i * 4] = pixels[i * 3]; rgba[i * 4 + 1] = pixels[i * 3 + 1]; rgba[i * 4 + 2] = pixels[i * 3 + 2]; rgba[i * 4 + 3] = 255; break;
                default: pixels.Slice(i * 4, 4).CopyTo(rgba.AsSpan(i * 4)); break;
            }
        }
        return rgba;
    }
}
