using System.Buffers.Binary;
using System.Text;

namespace PdfPecker.Ocr.Models;

/// <summary>
/// Retrained output weights for some of the recogniser's classes, written by tools/HeadTuner, which the recogniser
/// takes in place of its own when it loads. PP-OCRv6 saw few capital Danish and Norwegian letters in training: it reads
/// Æ as the two letters A and E, Ø as O or a zero, and Å as A. Only the last linear layer's weights of the listed
/// classes change, refitted on the features the unchanged network computes for rendered text, so every other class,
/// and everything before the last layer, is the converted model as it was.
/// </summary>
/// <remarks>
/// The file is "PPOCRHD\0", a 16-bit version (1), the layer's input features K and the number of classes, then for
/// each class its index, its character (a 32-bit byte count and UTF-8, checked against the character list) and K
/// weights and a bias as 32-bit floats, all little-endian.
/// </remarks>
internal sealed class HeadPatch
{
    private static ReadOnlySpan<byte> Magic => "PPOCRHD\0"u8;

    private const int FormatVersion = 1;

    private HeadPatch(int features, Column[] columns)
    {
        Features = features;
        Columns = columns;
    }

    /// <summary>The input features of the layer the columns belong to.</summary>
    public int Features { get; }

    public Column[] Columns { get; }

    /// <summary>One class's retrained weights.</summary>
    internal sealed record Column(int Index, string Character, float[] Weights, float Bias);

    /// <summary>Reads a patch; damaged data throws <see cref="InvalidDataException"/>.</summary>
    public static HeadPatch Read(ReadOnlySpan<byte> data)
    {
        int at = 0;
        ReadOnlySpan<byte> Take(int count, ReadOnlySpan<byte> all)
        {
            if (count < 0 || count > all.Length - at) throw new InvalidDataException("The head patch is truncated.");
            var slice = all.Slice(at, count);
            at += count;
            return slice;
        }

        if (!Take(Magic.Length, data).SequenceEqual(Magic)) throw new InvalidDataException("Not a PdfPecker.Ocr head patch.");
        int version = BinaryPrimitives.ReadUInt16LittleEndian(Take(2, data));
        if (version != FormatVersion) throw new InvalidDataException($"Head patch version {version} is not supported.");
        int features = BinaryPrimitives.ReadInt32LittleEndian(Take(4, data));
        int count = BinaryPrimitives.ReadInt32LittleEndian(Take(4, data));
        if (features <= 0 || features > 1 << 16 || count < 0 || count > 1 << 16) throw new InvalidDataException("The head patch's sizes are out of range.");
        var columns = new Column[count];
        for (int c = 0; c < count; c++)
        {
            int index = BinaryPrimitives.ReadInt32LittleEndian(Take(4, data));
            string character = Encoding.UTF8.GetString(Take(BinaryPrimitives.ReadInt32LittleEndian(Take(4, data)), data));
            var weights = new float[features];
            for (int k = 0; k < features; k++) weights[k] = BinaryPrimitives.ReadSingleLittleEndian(Take(4, data));
            float bias = BinaryPrimitives.ReadSingleLittleEndian(Take(4, data));
            columns[c] = new Column(index, character, weights, bias);
        }

        if (at != data.Length) throw new InvalidDataException("The head patch has data after its last column.");
        return new HeadPatch(features, columns);
    }

    /// <summary>Writes the patch in the format <see cref="Read"/> reads.</summary>
    public byte[] Write()
    {
        using var stream = new MemoryStream();
        Span<byte> buffer = stackalloc byte[4];
        stream.Write(Magic);
        BinaryPrimitives.WriteUInt16LittleEndian(buffer, FormatVersion);
        stream.Write(buffer[..2]);
        void I32(int v, Span<byte> b) { BinaryPrimitives.WriteInt32LittleEndian(b, v); stream.Write(b); }
        void F32(float v, Span<byte> b) { BinaryPrimitives.WriteSingleLittleEndian(b, v); stream.Write(b); }
        I32(Features, buffer);
        I32(Columns.Length, buffer);
        foreach (var column in Columns)
        {
            I32(column.Index, buffer);
            byte[] text = Encoding.UTF8.GetBytes(column.Character);
            I32(text.Length, buffer);
            stream.Write(text);
            foreach (float w in column.Weights) F32(w, buffer);
            F32(column.Bias, buffer);
        }

        return stream.ToArray();
    }

    /// <summary>A patch of the given columns.</summary>
    public static HeadPatch Create(int features, IEnumerable<Column> columns) => new(features, [.. columns]);

    /// <summary>
    /// Writes the columns into the recogniser's last linear layer, which must be followed only by its softmax, after
    /// checking that each class index holds the character the patch names.
    /// </summary>
    public void ApplyTo(ModelDefinition model)
    {
        var (weights, bias) = FinalLayer(model);
        int n = weights.Shape[1];
        if (weights.Shape[0] != Features) throw new InvalidDataException($"The head patch has {Features} features, the recogniser's last layer {weights.Shape[0]}.");
        foreach (var column in Columns)
        {
            if (column.Index <= 0 || column.Index >= n || column.Index >= model.Characters.Length || model.Characters[column.Index] != column.Character)
                throw new InvalidDataException($"The head patch's class {column.Index} ({column.Character}) is not in the recogniser's character list.");
            for (int k = 0; k < Features; k++) weights.Data[(k * n) + column.Index] = column.Weights[k];
            bias.Data[column.Index] = column.Bias;
        }
    }

    /// <summary>The weights [K, N] and bias [N] of a recogniser's last linear layer, which feeds its softmax.</summary>
    internal static (WeightDefinition Weights, WeightDefinition Bias) FinalLayer(ModelDefinition model)
    {
        var nodes = model.Nodes;
        if (nodes.Count < 2 || nodes[^1].Op != OpKind.Softmax || nodes[^2].Op != OpKind.Linear || nodes[^2].Weights.Length < 2)
            throw new InvalidDataException("The recogniser does not end in a linear layer and a softmax.");
        var weights = model.Weights[nodes[^2].Weights[0]];
        var bias = model.Weights[nodes[^2].Weights[1]];
        if (weights.Shape.Length != 2 || bias.Data.Length != weights.Shape[1]) throw new InvalidDataException("The recogniser's last layer has unexpected weights.");
        return (weights, bias);
    }
}
