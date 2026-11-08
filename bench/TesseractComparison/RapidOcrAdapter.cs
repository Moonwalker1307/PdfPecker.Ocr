using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace TesseractComparison;

/// <summary>
/// rapidocr on ONNX Runtime in a Python child process (rapidocr_server.py), as the reference for accuracy and time of the
/// models PdfPecker.Ocr runs. Pixels go over a pipe as raw bytes, so no image decoding is timed, but the pipe transfer is: it
/// is a few milliseconds per page. CPU time and peak memory are those of the Python process, which it reports itself.
/// </summary>
sealed class RapidOcrAdapter(string python, string script) : IOcrEngineAdapter
{
    Process? _process;
    bool _resetNext;
    long _rssAfterLoadKb;

    public string Name => "rapidocr";
    public bool InProcess => false;
    public bool Managed => false;
    public string Description => "rapidocr 3.10.0 (PP-OCRv6 small models) on ONNX Runtime, defaults of its config.yaml, word boxes on, in a Python child process";

    public long ExternalRssAfterLoadKb => _rssAfterLoadKb;

    public void Initialize(int threads)
    {
        var psi = new ProcessStartInfo(python)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            // The script lives in the harness folder: -I keeps the working directory and user site out of Python's import path.
            ArgumentList = { "-I", script },
        };
        _process = Process.Start(psi) ?? throw new InvalidOperationException("could not start Python");
        _process.ErrorDataReceived += (_, _) => { };
        _process.BeginErrorReadLine();
        Send(Encoding.ASCII.GetBytes($"INIT {threads}\n"));
        string? line = ReadLine();
        if (line is null || !line.StartsWith("READY", StringComparison.Ordinal)) throw new InvalidOperationException($"rapidocr did not start: {line}");
        var ready = line.Split(' ');
        if (ready.Length > 1) _rssAfterLoadKb = long.Parse(ready[1], CultureInfo.InvariantCulture);
    }

    public void Prepare(CorpusPage page) { }

    public void BeginMeasurement() => _resetNext = true;

    public OcrResult Recognize(ReadOnlySpan<byte> pixels, int width, int height, PixelFormat format)
    {
        int channels = format == PixelFormat.Gray8 ? 1 : 3;
        byte[] data;
        if (format == PixelFormat.Rgba32)
        {
            data = new byte[width * height * 3];
            for (int i = 0, j = 0; i < pixels.Length; i += 4, j += 3) { data[j] = pixels[i]; data[j + 1] = pixels[i + 1]; data[j + 2] = pixels[i + 2]; }
            pixels = data;
        }
        Send(Encoding.ASCII.GetBytes($"REQ {width} {height} {channels} {(_resetNext ? 1 : 0)}\n"));
        _resetNext = false;
        Send(pixels);
        var lines = new List<string>();
        var words = new List<Box>();
        double cpu = 0;
        long peak = 0, processPeak = 0;
        while (true)
        {
            string? line = ReadLine() ?? throw new InvalidOperationException("rapidocr ended unexpectedly");
            if (line == "END") break;
            var f = line.Split('\t');
            switch (f[0])
            {
                case "L" when f.Length >= 4:
                    lines.Add(f[3]);
                    break;
                case "W" when f.Length >= 4:
                    var (x, y, w, h) = Bounds(f[2]);
                    words.Add(new Box(f[3], x, y, w, h, double.Parse(f[1], CultureInfo.InvariantCulture)));
                    break;
                case "T":
                    cpu = double.Parse(f[2], CultureInfo.InvariantCulture);
                    peak = long.Parse(f[3], CultureInfo.InvariantCulture);
                    if (f.Length > 4) processPeak = long.Parse(f[4], CultureInfo.InvariantCulture);
                    break;
            }
        }
        return new OcrResult { Text = string.Join('\n', lines), Words = words, ExternalCpuMs = cpu, ExternalPeakKb = peak, ExternalProcessPeakKb = processPeak };
    }

    static (double X, double Y, double W, double H) Bounds(string quad)
    {
        var v = quad.Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        for (int i = 0; i + 1 < v.Length; i += 2)
        {
            minX = Math.Min(minX, v[i]); maxX = Math.Max(maxX, v[i]);
            minY = Math.Min(minY, v[i + 1]); maxY = Math.Max(maxY, v[i + 1]);
        }
        return (minX, minY, maxX - minX, maxY - minY);
    }

    void Send(ReadOnlySpan<byte> bytes)
    {
        var s = _process!.StandardInput.BaseStream;
        s.Write(bytes);
        s.Flush();
    }

    string? ReadLine()
    {
        var s = _process!.StandardOutput.BaseStream;
        var buf = new List<byte>(128);
        int b;
        while ((b = s.ReadByte()) >= 0)
        {
            if (b == '\n') return Encoding.UTF8.GetString(buf.ToArray());
            buf.Add((byte)b);
        }
        return null;
    }

    public void Dispose()
    {
        if (_process is null) return;
        try { Send("QUIT\n"u8); _process.WaitForExit(5000); } catch (IOException) { }
        if (!_process.HasExited) _process.Kill();
        _process.Dispose();
        _process = null;
    }
}
