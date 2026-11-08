namespace TesseractComparison;

/// <summary>
/// Checks of the metrics and the PNG decoder that need no engine and no corpus, so a change to the scoring cannot silently
/// change every number in a report. Run with <c>TesseractComparison selftest</c>; the exit code is 1 on a failure.
/// </summary>
static class SelfTest
{
    public static int Run()
    {
        int failures = 0;
        void Check(string name, bool ok)
        {
            if (!ok) { failures++; Console.Error.WriteLine($"FAIL {name}"); }
        }
        void Near(string name, double actual, double expected) => Check($"{name}: {actual} vs {expected}", Math.Abs(actual - expected) < 1e-9);

        Check("distance kitten/sitting", Metrics.Distance<char>("kitten", "sitting") == 3);
        Check("distance empty", Metrics.Distance<char>("", "abc") == 3 && Metrics.Distance<char>("abc", "") == 3 && Metrics.Distance<char>("", "") == 0);
        Check("distance words", Metrics.Distance<string>(["a", "b", "c"], ["a", "c", "d"]) == 2);

        Check("normalize whitespace", Metrics.Normalize("  a \n\n b\t c  ") == "a b c");
        Check("normalize quotes", Metrics.Normalize("“it’s” – ok") == "\"it's\" - ok");
        Check("normalize composed", Metrics.Normalize("é") == "é");
        Check("normalize keeps case and accents", Metrics.Normalize("Æble ÖL") == "Æble ÖL");

        Near("bag f1 identical", Metrics.BagF1(["a", "b"], ["b", "a"]), 1);
        Near("bag f1 disjoint", Metrics.BagF1(["a"], ["b"]), 0);
        Near("bag f1 half", Metrics.BagF1(["a", "b"], ["a", "c"]), 0.5);
        Near("bag f1 counts duplicates", Metrics.BagF1(["a", "a"], ["a"]), 2.0 / 3.0);

        var a = new Box("", 0, 0, 10, 10, 1);
        Near("iou same", Metrics.Iou(a, a), 1);
        Near("iou half", Metrics.Iou(a, new Box("", 5, 0, 10, 10, 1)), 50.0 / 150.0);
        Near("iou apart", Metrics.Iou(a, new Box("", 20, 20, 5, 5, 1)), 0);
        var (recall, precision, mean) = Metrics.MatchBoxes([a, new Box("", 50, 0, 10, 10, 1)], [new Box("", 1, 0, 10, 10, 1), new Box("", 200, 0, 10, 10, 1)]);
        Near("match recall", recall, 0.5);
        Near("match precision", precision, 0.5);
        Check("match mean iou", mean > 0.8 && mean < 1);

        // A 2x2 RGB PNG written by hand: row 0 unfiltered, row 1 with the Sub filter. The decoder does not check CRCs.
        var raw = new byte[] { 0, 255, 0, 0, 0, 255, 0, 1, 10, 20, 30, 5, 5, 5 };
        using (var ms = new MemoryStream())
        {
            using (var z = new System.IO.Compression.ZLibStream(ms, System.IO.Compression.CompressionLevel.Optimal, leaveOpen: true)) z.Write(raw);
            var idat = ms.ToArray();
            using var png = new MemoryStream();
            png.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
            void Chunk(string type, byte[] data)
            {
                png.Write([(byte)(data.Length >> 24), (byte)(data.Length >> 16), (byte)(data.Length >> 8), (byte)data.Length]);
                png.Write(System.Text.Encoding.ASCII.GetBytes(type));
                png.Write(data);
                png.Write([0, 0, 0, 0]);
            }
            Chunk("IHDR", [0, 0, 0, 2, 0, 0, 0, 2, 8, 2, 0, 0, 0]);
            Chunk("IDAT", idat);
            Chunk("IEND", []);
            var (pixels, w, h, c) = PngDecoder.Decode(png.ToArray());
            Check("png size", w == 2 && h == 2 && c == 3);
            Check("png row 0", pixels[0] == 255 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 0 && pixels[4] == 255 && pixels[5] == 0);
            Check("png sub filter", pixels[6] == 10 && pixels[7] == 20 && pixels[8] == 30 && pixels[9] == 15 && pixels[10] == 25 && pixels[11] == 35);
        }

        Console.WriteLine(failures == 0 ? "selftest passed" : $"selftest: {failures} failure(s)");
        return failures == 0 ? 0 : 1;
    }
}
