using System.Buffers.Binary;
using System.IO.Compression;
using PdfPecker.Ocr.Inference;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>
/// An ONNX Runtime reference written by tools/OnnxReference/generate.py: the input pixels, the output's shape and
/// statistics, sampled output values and, for the recogniser and classifier, the argmax of every row.
/// </summary>
internal sealed class ReferenceFixture
{
    public string Model { get; private init; } = "";
    public int[] InputShape { get; private init; } = [];
    public byte[] Pixels { get; private init; } = [];
    public int[] OutputShape { get; private init; } = [];
    public double Mean { get; private init; }
    public double StandardDeviation { get; private init; }
    public float Minimum { get; private init; }
    public float Maximum { get; private init; }
    public (int Index, float Value)[] Samples { get; private init; } = [];
    public (int Index, float Value)[] RowMaxima { get; private init; } = [];

    public static string Folder => Path.Combine(AppContext.BaseDirectory, "Fixtures", "onnx");

    public static ReferenceFixture Load(string name)
    {
        using var file = File.OpenRead(Path.Combine(Folder, name + ".ref.gz"));
        using var gzip = new GZipStream(file, CompressionMode.Decompress);
        using var buffer = new MemoryStream();
        gzip.CopyTo(buffer);
        var data = buffer.ToArray().AsSpan();
        int pos = 0;
        ReadOnlySpan<byte> Take(int count, Span<byte> all)
        {
            var slice = all.Slice(pos, count);
            pos += count;
            return slice;
        }
        int I32(Span<byte> all) => BinaryPrimitives.ReadInt32LittleEndian(Take(4, all));
        float F32(Span<byte> all) => BinaryPrimitives.ReadSingleLittleEndian(Take(4, all));
        int[] Shape(Span<byte> all)
        {
            int rank = Take(1, all)[0];
            var dims = new int[rank];
            for (int i = 0; i < rank; i++) dims[i] = I32(all);
            return dims;
        }
        (int, float)[] Pairs(Span<byte> all)
        {
            var pairs = new (int, float)[I32(all)];
            for (int i = 0; i < pairs.Length; i++) pairs[i] = (I32(all), F32(all));
            return pairs;
        }
        if (!Take(4, data).SequenceEqual("OREF"u8) || I32(data) != 1) throw new InvalidDataException("Not a reference fixture.");
        string model = System.Text.Encoding.ASCII.GetString(Take(Take(1, data)[0], data));
        int[] inputShape = Shape(data);
        byte[] pixels = Take(inputShape[0] * inputShape[2] * inputShape[3] * 3, data).ToArray();
        int[] outputShape = Shape(data);
        double mean = BinaryPrimitives.ReadDoubleLittleEndian(Take(8, data));
        double std = BinaryPrimitives.ReadDoubleLittleEndian(Take(8, data));
        float min = F32(data), max = F32(data);
        var samples = Pairs(data);
        var rows = Pairs(data);
        if (pos != data.Length) throw new InvalidDataException("Trailing fixture data.");
        return new ReferenceFixture
        {
            Model = model,
            InputShape = inputShape,
            Pixels = pixels,
            OutputShape = outputShape,
            Mean = mean,
            StandardDeviation = std,
            Minimum = min,
            Maximum = max,
            Samples = samples,
            RowMaxima = rows,
        };
    }

    /// <summary>The model input: (pixel / 255 - 0.5) / 0.5, channel c from byte c of each pixel, in NCHW.</summary>
    public Tensor Input()
    {
        int n = InputShape[0], h = InputShape[2], w = InputShape[3];
        var data = new float[n * 3 * h * w];
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < 3; c++)
                    {
                        float p = Pixels[((b * h + y) * w + x) * 3 + c];
                        data[((b * 3 + c) * h + y) * w + x] = (p / 255f - 0.5f) / 0.5f;
                    }
        return Tensor.FromBuffer(data, InputShape);
    }
}
