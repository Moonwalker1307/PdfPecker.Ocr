using System.Buffers.Binary;
using System.Text;
using PdfPecker.Ocr.Pdf;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Random OCR results through the searchable text layer: <see cref="SearchableLayer.Write"/> and then
/// <see cref="PdfStamper.Stamp(byte[], byte[], PdfStampOptions?, CancellationToken)"/> onto a scan, as
/// <see cref="PdfOcr.MakeSearchable(byte[], OcrOptions?, CancellationToken)"/> does. The input is read as a stream of
/// choices: page sizes in pixels and points, resolutions, and lines and words with quads that are degenerate, inverted,
/// rotated, far off the page or not numbers at all, texts with control characters, lone surrogates, marks and
/// right-to-left letters, and confidences outside 0 to 1.
/// </summary>
/// <remarks>
/// The layer writer must take any result a recognizer could hand it, so nothing it throws is expected, and both the
/// layer and the stamped PDF must open again with PdfPecker's text extraction; a failure there is reported as an
/// <see cref="InvalidOperationException"/>, which this target does not expect.
/// </remarks>
internal static class SearchableLayerTarget
{
    /// <summary>Numbers that break arithmetic, picked by index from the input.</summary>
    private static readonly float[] Interesting =
    [
        0, -0f, 1, -1, 0.5f, 1e-30f, -1e-30f, 1e30f, -1e30f, float.MaxValue, float.MinValue, float.Epsilon, float.NaN,
        float.PositiveInfinity, float.NegativeInfinity, 72, 300, 2480, 3508, 14400, 1e9f, 65535, 16777217,
    ];

    /// <summary>Scanned pages of one to three A4 pages, made once, onto which the layers are stamped.</summary>
    private static readonly Lazy<byte[][]> Scans = new(() =>
    {
        var result = new byte[3][];
        for (int n = 1; n <= 3; n++)
        {
            string body = string.Concat(Enumerable.Range(0, n).Select(i => $"<div style=\"page-break-after:always\">Scan page {i + 1}</div>"));
            result[n - 1] = HtmlToPdf.Convert("<body>" + body + "</body>", new PdfOptions { PageSize = PageSize.A4 });
        }

        return result;
    });

    public static Target Target { get; } = new()
    {
        Name = "searchable-layer",
        Extension = ".layer",
        MaxSize = 384 * 1024,
        Seeds = _ => Generate(),
        Run = Run,
    };

    private static void Run(byte[] input, CancellationToken ct)
    {
        var reader = new Reader(input);
        int pageCount = 1 + (reader.Byte() % 3);
        var layer = new LayerPage[pageCount];
        bool scaleByDpi = (reader.Byte() & 1) != 0;
        for (int p = 0; p < pageCount; p++)
        {
            ct.ThrowIfCancellationRequested();
            double widthPt = reader.Number(), heightPt = reader.Number();
            // A page without a result keeps an empty stamp page.
            OcrPage? page = (reader.Byte() & 7) == 0 ? null : ReadPage(ref reader, p);
            layer[p] = new LayerPage(page, widthPt, heightPt, scaleByDpi);
        }

        byte[] stamp = SearchableLayer.Write(layer, ct);
        Reopen(stamp, "layer");
        byte[] stamped = PdfStamper.Stamp(Scans.Value[pageCount - 1], stamp, new PdfStampOptions { MatchPages = true, Underlay = true }, ct);
        Reopen(stamped, "stamped PDF");
    }

    private static OcrPage ReadPage(ref Reader reader, int index)
    {
        int width = reader.Size(), height = reader.Size();
        float dpi = reader.Number();
        int lineCount = reader.Byte() % 24;
        var lines = new List<OcrLine>(lineCount);
        for (int l = 0; l < lineCount && !reader.Done; l++)
        {
            OcrQuad quad = reader.Quad(width, height);
            string text = reader.Text();
            float confidence = reader.Number();
            int wordCount = reader.Byte() % 8;
            var words = new List<OcrWord>(wordCount);
            for (int w = 0; w < wordCount && !reader.Done; w++)
                words.Add(new OcrWord(reader.Text(), reader.Quad(width, height), reader.Number()));
            lines.Add(new OcrLine(text, quad, confidence, words));
        }

        return new OcrPage(index, width, height, dpi, lines);
    }

    /// <summary>Opens a PDF the package wrote with PdfPecker's text extraction, which reads every page's content and font.</summary>
    private static void Reopen(byte[] pdf, string what)
    {
        try
        {
            PdfText.ExtractPages(pdf);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            throw new InvalidOperationException($"The {what} the package wrote does not open again: {ex.GetType().Name}: {ex.Message}", ex);
        }
    }

    /// <summary>Reads choices from the input; past its end every choice is zero, so any input is a complete result.</summary>
    private ref struct Reader(byte[] data)
    {
        private readonly byte[] _data = data;
        private int _at;

        public readonly bool Done => _at >= _data.Length;

        public byte Byte() => _at < _data.Length ? _data[_at++] : (byte)0;

        private ReadOnlySpan<byte> Take(int n)
        {
            if (_at + n > _data.Length)
            {
                _at = _data.Length;
                return new byte[n];
            }

            var span = _data.AsSpan(_at, n);
            _at += n;
            return span;
        }

        /// <summary>A number: one of <see cref="Interesting"/>, a small whole number, or any 32-bit float.</summary>
        public float Number()
        {
            byte kind = Byte();
            return (kind % 4) switch
            {
                0 => Interesting[Byte() % Interesting.Length],
                1 => BinaryPrimitives.ReadInt16LittleEndian(Take(2)),
                2 => BinaryPrimitives.ReadUInt16LittleEndian(Take(2)) / 16f,
                _ => BinaryPrimitives.ReadSingleLittleEndian(Take(4)),
            };
        }

        /// <summary>A page size in pixels: usually a plausible one, sometimes zero, negative or huge.</summary>
        public int Size()
        {
            byte kind = Byte();
            return (kind % 8) switch
            {
                0 => BinaryPrimitives.ReadInt32LittleEndian(Take(4)),
                1 => 0,
                _ => 1 + BinaryPrimitives.ReadUInt16LittleEndian(Take(2)),
            };
        }

        /// <summary>A quad: an upright box inside the page, a turned one, or four corners read freely.</summary>
        public OcrQuad Quad(int width, int height)
        {
            byte kind = Byte();
            switch (kind % 3)
            {
                case 0:
                {
                    float x = Byte() * Math.Max(width, 1) / 255f, y = Byte() * Math.Max(height, 1) / 255f;
                    return OcrGeometry.FromBox(x, y, Number(), Number());
                }

                case 1:
                {
                    // A box turned by any angle about its top-left corner, as a skewed or upside-down line would be.
                    double angle = Byte() * (Math.PI * 2 / 256);
                    float x = Number(), y = Number(), w = Number(), h = Number();
                    float cos = (float)Math.Cos(angle), sin = (float)Math.Sin(angle);
                    return new OcrQuad(new OcrPoint(x, y), new OcrPoint(x + (w * cos), y - (w * sin)),
                        new OcrPoint(x + (w * cos) + (h * sin), y - (w * sin) + (h * cos)), new OcrPoint(x + (h * sin), y + (h * cos)));
                }

                default:
                    return new OcrQuad(new OcrPoint(Number(), Number()), new OcrPoint(Number(), Number()),
                        new OcrPoint(Number(), Number()), new OcrPoint(Number(), Number()));
            }
        }

        /// <summary>
        /// Text as UTF-16 code units straight from the input, so lone surrogates, controls and noncharacters all occur;
        /// a length byte of 255 reads a 16-bit length instead, for long texts with many distinct characters.
        /// </summary>
        public string Text()
        {
            int length = Byte();
            if (length == 255) length = BinaryPrimitives.ReadUInt16LittleEndian(Take(2));
            length = Math.Min(length, (_data.Length - _at) / 2);
            if (length <= 0) return string.Empty;
            var chars = new char[length];
            for (int i = 0; i < length; i++) chars[i] = (char)BinaryPrimitives.ReadUInt16LittleEndian(Take(2));
            return new string(chars);
        }
    }

    private static IEnumerable<(string Name, byte[] Data)> Generate()
    {
        yield return ("generated:invoice", Seed(1, (595, 842), (2480, 3508, 300), [("Invoice 2026-0042", 0), ("Total: 1,234.50 EUR", 0)]));
        yield return ("generated:turned", Seed(1, (842, 595), (3508, 2480, 300), [("Upside down", 128), ("Skewed line", 5)]));
        yield return ("generated:scripts", Seed(1, (612, 792), (1700, 2200, 200), [("Grüße aus Århus", 0), ("שלום עולם مرحبا", 0), ("é‍😀 \uD800 tab\there", 64)]));
        yield return ("generated:two-pages", Seed(2, (595, 842), (1240, 1754, 150), [("Page text", 0)]));
        yield return ("generated:many-characters", ManyCharacters());
    }

    /// <summary>A seed in the reader's layout: pages of one size, each with the lines given, every line one word per space-separated part.</summary>
    private static byte[] Seed(int pages, (short W, short H) points, (int W, int H, short Dpi) pixels, (string Text, byte Angle)[] lines, bool words = true)
    {
        var s = new MemoryStream();
        var w = new BinaryWriter(s, Encoding.Unicode);
        void Short(short v) { w.Write((byte)1); w.Write(v); }
        void Text(string t)
        {
            if (t.Length >= 255) { w.Write((byte)255); w.Write((ushort)t.Length); }
            else w.Write((byte)t.Length);
            foreach (char c in t) w.Write((ushort)c);
        }

        w.Write((byte)(pages - 1));
        w.Write((byte)0);
        for (int p = 0; p < pages; p++)
        {
            Short(points.W);
            Short(points.H);
            w.Write((byte)1);
            w.Write((byte)2);
            w.Write((ushort)(pixels.W - 1));
            w.Write((byte)2);
            w.Write((ushort)(pixels.H - 1));
            Short(pixels.Dpi);
            w.Write((byte)lines.Length);
            for (int l = 0; l < lines.Length; l++)
            {
                short y = (short)(200 + (l * 120));
                w.Write((byte)1);
                w.Write(lines[l].Angle);
                Short(200);
                Short(y);
                Short((short)(40 * lines[l].Text.Length));
                Short(80);
                Text(lines[l].Text);
                Short(1);
                string[] parts = words ? lines[l].Text.Split(' ') : [];
                w.Write((byte)Math.Min(parts.Length, 7));
                short x = 200;
                foreach (string part in parts.Take(7))
                {
                    Text(part);
                    w.Write((byte)0);
                    w.Write((byte)(x * 255 / pixels.W));
                    w.Write((byte)(y * 255 / pixels.H));
                    Short((short)(40 * part.Length));
                    Short(80);
                    Short(1);
                    x += (short)(40 * (part.Length + 1));
                }
            }
        }

        w.Flush();
        return s.ToArray();
    }

    /// <summary>
    /// Lines with 66,400 distinct characters between them, most from the supplementary planes: more than the 65,535
    /// codes the layer's font can hold, so the codes run out part way.
    /// </summary>
    private static byte[] ManyCharacters()
    {
        string Run(int first, int count)
        {
            var text = new StringBuilder(count * 2);
            for (int i = 0; i < count; i++) text.Append(char.ConvertFromUtf32(first + i));
            return text.ToString();
        }

        return Seed(1, (595, 842), (2480, 3508, 300), [(Run(0x20000, 32_700), 0), (Run(0x30000, 32_700), 0), (Run(0x4E00, 1_000), 0)], words: false);
    }
}
