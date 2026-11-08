using System.Buffers.Binary;
using System.Text;

namespace PdfPecker.Ocr.Models;

// The compact model format that tools/ModelBuilder writes and OcrModel reads. Its layout is documented in
// tools/ModelBuilder/README.md; this file is the single definition of the byte layout and of the operator codes, so the
// builder, the runtime and the tests cannot disagree about them.

/// <summary>
/// The operators of the runtime graph. Each operator reads its integer and float attributes by position; the comment
/// on each member lists them. Values are tensors in NCHW (or row-major for the sequence part of the recogniser).
/// </summary>
internal enum OpKind : byte
{
    /// <summary>
    /// Convolution with bias, a fused activation and an optional residual input added after the activation.
    /// Ints: kh, kw, strideH, strideW, dilationH, dilationW, padTop, padLeft, padBottom, padRight, group, activation,
    /// hasResidual. Floats: activation alpha, beta. Weights: W [Cout, Cin/group, kh, kw], B [Cout].
    /// Inputs: x, then the residual when hasResidual is 1.
    /// </summary>
    Conv = 1,

    /// <summary>
    /// Transposed convolution whose kernel equals its stride, without padding or groups.
    /// Ints: kh, kw, activation. Floats: activation alpha, beta. Weights: W [Cin, Cout, kh, kw], B [Cout].
    /// </summary>
    ConvTranspose = 2,

    /// <summary>Max pooling. Ints: kh, kw, strideH, strideW, padTop, padLeft, padBottom, padRight.</summary>
    MaxPool = 3,

    /// <summary>
    /// Average pooling. Ints: kh, kw, strideH, strideW, padTop, padLeft, padBottom, padRight, countIncludePad.
    /// </summary>
    AveragePool = 4,

    /// <summary>The mean of each channel plane, giving [N, C, 1, 1].</summary>
    GlobalAveragePool = 5,

    /// <summary>
    /// A squeeze-and-excitation block: the channel means go through two 1x1 convolutions, and the result scales the
    /// input per channel (y = x * s), or is added as a scaled copy (y = x + x * s) when addInput is 1.
    /// Ints: activation1, activation2, addInput. Floats: alpha1, beta1, alpha2, beta2.
    /// Weights: W1 [Cmid, C], B1 [Cmid], W2 [C, Cmid], B2 [C].
    /// </summary>
    SqueezeExcite = 6,

    /// <summary>
    /// Elementwise sum with broadcasting. Ints: factorH, factorW; factors above 1 upsample the second input by nearest
    /// neighbour first (the feature pyramid's Resize then Add).
    /// </summary>
    Add = 7,

    /// <summary>Elementwise product with broadcasting. Ints: factorH, factorW, as for Add.</summary>
    Mul = 8,

    /// <summary>A standalone activation. Ints: activation. Floats: alpha, beta.</summary>
    Activation = 9,

    /// <summary>
    /// Concatenation. Ints: axis, then a pair of nearest-neighbour upsampling factors (H, W) for each input, 1 for none.
    /// </summary>
    Concat = 10,

    /// <summary>Layer normalisation over the last axis. Floats: epsilon. Weights: gamma, beta.</summary>
    LayerNorm = 11,

    /// <summary>
    /// A matrix product with constant weights over the last axis, with bias, activation and an optional residual.
    /// Ints: activation, hasResidual. Floats: alpha, beta. Weights: W [K, N], B [N]. Inputs: x, then the residual.
    /// </summary>
    Linear = 12,

    /// <summary>
    /// Multi-head self-attention on a packed query, key and value tensor [N, T, 3 * C], giving [N, T, C]: per head,
    /// softmax(q * scale x k^T) x v. Ints: heads. Floats: scale.
    /// </summary>
    Attention = 13,

    /// <summary>Softmax over the last axis.</summary>
    Softmax = 14,

    /// <summary>Axis permutation. Ints: the permutation.</summary>
    Transpose = 15,

    /// <summary>A view with another shape. Ints: the target shape, where 0 copies the input's dimension and -1 is inferred.</summary>
    Reshape = 16,

    /// <summary>Nearest-neighbour upsampling by integer factors. Ints: factorH, factorW.</summary>
    Upsample = 17,
}

/// <summary>The activations a node can apply to its result.</summary>
internal enum ActivationKind : byte
{
    /// <summary>No activation.</summary>
    None = 0,

    /// <summary>max(x, 0).</summary>
    Relu = 1,

    /// <summary>1 / (1 + exp(-x)).</summary>
    Sigmoid = 2,

    /// <summary>clamp(alpha * x + beta, 0, 1).</summary>
    HardSigmoid = 3,

    /// <summary>x * clamp(alpha * x + beta, 0, 1).</summary>
    HardSwish = 4,

    /// <summary>x * sigmoid(x), also called SiLU.</summary>
    Swish = 5,

    /// <summary>0.5 * x * (1 + erf(x / sqrt(2))), the exact form of GELU.</summary>
    Gelu = 6,
}

/// <summary>How a weight tensor's values are stored in the file.</summary>
internal enum WeightStorage : byte
{
    /// <summary>IEEE single precision, four byte planes.</summary>
    Float32 = 1,

    /// <summary>
    /// Single precision rounded to nearest even at 15 mantissa bits, stored as the three high bytes in three planes.
    /// Its relative error of 2^-16 keeps the models within 1e-3 of the original, which half precision does not.
    /// </summary>
    Float24 = 2,

    /// <summary>IEEE half precision, two byte planes.</summary>
    Float16 = 3,
}

/// <summary>A constant tensor of a model.</summary>
internal sealed class WeightDefinition(int[] shape, float[] data, WeightStorage storage)
{
    public int[] Shape { get; } = shape;

    public float[] Data { get; } = data;

    public WeightStorage Storage { get; } = storage;
}

/// <summary>One operator of a model graph.</summary>
internal sealed class NodeDefinition(OpKind op, int[] inputs, int output, int[] ints, float[] floats, int[] weights)
{
    public OpKind Op { get; } = op;

    /// <summary>The values the node reads; value 0 is the graph input.</summary>
    public int[] Inputs { get; } = inputs;

    /// <summary>The value the node writes. Every node writes a new value, numbered in node order from 1.</summary>
    public int Output { get; } = output;

    public int[] Ints { get; } = ints;

    public float[] Floats { get; } = floats;

    /// <summary>Indices into the model's weight list.</summary>
    public int[] Weights { get; } = weights;
}

/// <summary>A whole model: the graph, its weights and, for the recogniser, the character list.</summary>
internal sealed class ModelDefinition
{
    /// <summary>The magic bytes at the start of every model file.</summary>
    public static ReadOnlySpan<byte> Magic => "PPOCRNN\0"u8;

    /// <summary>The format version this code writes and reads.</summary>
    public const int FormatVersion = 1;

    public string Name { get; set; } = "";

    /// <summary>Attribution and change notice carried inside the file, as the models' Apache 2.0 licence asks.</summary>
    public string Notice { get; set; } = "";

    /// <summary>The input shape, with -1 for a dimension the caller chooses.</summary>
    public int[] InputShape { get; set; } = [];

    public List<WeightDefinition> Weights { get; } = [];

    public List<NodeDefinition> Nodes { get; } = [];

    /// <summary>The recogniser's classes; empty for the other models.</summary>
    public string[] Characters { get; set; } = [];

    /// <summary>The value the model returns: the last node's output.</summary>
    public int OutputValue => Nodes.Count == 0 ? 0 : Nodes[^1].Output;

    /// <summary>Writes the model in the file format, rounding each weight to its storage precision.</summary>
    public byte[] Serialize()
    {
        var w = new FormatWriter();
        w.Bytes(Magic);
        w.U16(FormatVersion);
        w.String(Name);
        w.String(Notice);
        w.IntArray(InputShape);
        w.I32(Weights.Count);
        foreach (var weight in Weights)
        {
            w.U8((byte)weight.Storage);
            w.IntArray(weight.Shape);
            w.I32(weight.Data.Length);
            WriteWeightData(w, weight);
        }
        w.I32(Nodes.Count);
        foreach (var node in Nodes)
        {
            w.U8((byte)node.Op);
            w.IntArray(node.Inputs);
            w.I32(node.Output);
            w.IntArray(node.Ints);
            w.U8(checked((byte)node.Floats.Length));
            foreach (float f in node.Floats) w.F32(f);
            w.IntArray(node.Weights);
        }
        w.I32(Characters.Length);
        w.String(string.Join('\n', Characters));
        return w.ToArray();
    }

    /// <summary>Reads a model; damaged data throws <see cref="InvalidDataException"/>.</summary>
    public static ModelDefinition Deserialize(ReadOnlySpan<byte> data)
    {
        var r = new FormatReader(data);
        if (!r.Bytes(Magic.Length).SequenceEqual(Magic)) throw new InvalidDataException("Not a PdfPecker.Ocr model.");
        int version = r.U16();
        if (version != FormatVersion) throw new InvalidDataException($"Model format version {version} is not supported.");
        var model = new ModelDefinition { Name = r.String(), Notice = r.String(), InputShape = r.IntArray() };
        int weightCount = r.Count();
        for (int i = 0; i < weightCount; i++)
        {
            var storage = (WeightStorage)r.U8();
            int[] shape = r.IntArray();
            int length = r.Count();
            long expected = 1;
            foreach (int d in shape)
            {
                if (d < 0) throw new InvalidDataException("Negative weight dimension.");
                expected *= d;
            }
            if (expected != length) throw new InvalidDataException("Weight size does not match its shape.");
            model.Weights.Add(new WeightDefinition(shape, ReadWeightData(ref r, storage, length), storage));
        }
        int nodeCount = r.Count();
        for (int i = 0; i < nodeCount; i++)
        {
            var op = (OpKind)r.U8();
            int[] inputs = r.IntArray();
            int output = r.I32();
            int[] ints = r.IntArray();
            float[] floats = new float[r.U8()];
            for (int j = 0; j < floats.Length; j++) floats[j] = r.F32();
            int[] weights = r.IntArray();
            if (output != i + 1) throw new InvalidDataException("Node outputs must be numbered in order.");
            foreach (int input in inputs)
                if (input < 0 || input > i) throw new InvalidDataException("A node reads a value that is not yet defined.");
            foreach (int weight in weights)
                if (weight < 0 || weight >= model.Weights.Count) throw new InvalidDataException("A node refers to a missing weight.");
            model.Nodes.Add(new NodeDefinition(op, inputs, output, ints, floats, weights));
        }
        int characterCount = r.Count();
        string characters = r.String();
        model.Characters = characterCount == 0 ? [] : characters.Split('\n');
        if (model.Characters.Length != characterCount) throw new InvalidDataException("Character list is damaged.");
        if (!r.AtEnd) throw new InvalidDataException("Unexpected data after the model.");
        return model;
    }

    /// <summary>Rounds a value to the 24-bit storage precision, to nearest even.</summary>
    public static float RoundToFloat24(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7F800000) == 0x7F800000) return value;
        bits += 0x7Fu + ((bits >> 8) & 1);
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFFFF00u);
    }

    private static void WriteWeightData(FormatWriter w, WeightDefinition weight)
    {
        float[] data = weight.Data;
        // Byte planes put the slowly varying sign and exponent bytes of all values next to each other, which Brotli
        // compresses far better than interleaved values.
        switch (weight.Storage)
        {
            case WeightStorage.Float32:
                for (int shift = 24; shift >= 0; shift -= 8)
                    foreach (float f in data) w.U8((byte)(BitConverter.SingleToUInt32Bits(f) >> shift));
                break;
            case WeightStorage.Float24:
                for (int shift = 24; shift >= 8; shift -= 8)
                    foreach (float f in data) w.U8((byte)(BitConverter.SingleToUInt32Bits(RoundToFloat24(f)) >> shift));
                break;
            case WeightStorage.Float16:
                for (int shift = 8; shift >= 0; shift -= 8)
                    foreach (float f in data) w.U8((byte)(BitConverter.HalfToUInt16Bits((Half)f) >> shift));
                break;
            default:
                throw new ArgumentException("Unknown weight storage.", nameof(weight));
        }
    }

    private static float[] ReadWeightData(ref FormatReader r, WeightStorage storage, int length)
    {
        float[] result;
        switch (storage)
        {
            case WeightStorage.Float32:
            case WeightStorage.Float24:
            {
                int planes = storage == WeightStorage.Float32 ? 4 : 3;
                var bytes = r.Bytes(checked(length * planes));
                result = new float[length];
                for (int i = 0; i < length; i++)
                {
                    uint bits = (uint)bytes[i] << 24 | (uint)bytes[length + i] << 16 | (uint)bytes[2 * length + i] << 8;
                    if (planes == 4) bits |= bytes[3 * length + i];
                    result[i] = BitConverter.UInt32BitsToSingle(bits);
                }
                break;
            }
            case WeightStorage.Float16:
            {
                var bytes = r.Bytes(checked(length * 2));
                result = new float[length];
                for (int i = 0; i < length; i++)
                    result[i] = (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[i] << 8 | bytes[length + i]));
                break;
            }
            default:
                throw new InvalidDataException("Unknown weight storage.");
        }
        return result;
    }

    private sealed class FormatWriter
    {
        private readonly MemoryStream _stream = new();

        public void Bytes(ReadOnlySpan<byte> bytes) => _stream.Write(bytes);

        public void U8(byte value) => _stream.WriteByte(value);

        public void U16(int value)
        {
            Span<byte> b = stackalloc byte[2];
            BinaryPrimitives.WriteUInt16LittleEndian(b, checked((ushort)value));
            _stream.Write(b);
        }

        public void I32(int value)
        {
            Span<byte> b = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(b, value);
            _stream.Write(b);
        }

        public void F32(float value) => I32(BitConverter.SingleToInt32Bits(value));

        public void IntArray(int[] values)
        {
            I32(values.Length);
            foreach (int v in values) I32(v);
        }

        public void String(string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            I32(bytes.Length);
            _stream.Write(bytes);
        }

        public byte[] ToArray() => _stream.ToArray();
    }

    private ref struct FormatReader(ReadOnlySpan<byte> data)
    {
        private readonly ReadOnlySpan<byte> _data = data;
        private int _position;

        public readonly bool AtEnd => _position == _data.Length;

        public ReadOnlySpan<byte> Bytes(int count)
        {
            if (count < 0 || count > _data.Length - _position) throw new InvalidDataException("Model data is truncated.");
            var result = _data.Slice(_position, count);
            _position += count;
            return result;
        }

        public byte U8() => Bytes(1)[0];

        public int U16() => BinaryPrimitives.ReadUInt16LittleEndian(Bytes(2));

        public int I32() => BinaryPrimitives.ReadInt32LittleEndian(Bytes(4));

        public float F32() => BitConverter.Int32BitsToSingle(I32());

        /// <summary>A count, which also bounds how much the reader allocates before it has seen the data.</summary>
        public int Count()
        {
            int count = I32();
            if (count < 0 || count > _data.Length - _position + 64 * 1024 * 1024) throw new InvalidDataException("Bad count in model data.");
            return count;
        }

        public int[] IntArray()
        {
            int count = I32();
            if (count < 0 || count > (_data.Length - _position) / 4) throw new InvalidDataException("Bad array length in model data.");
            var result = new int[count];
            for (int i = 0; i < count; i++) result[i] = I32();
            return result;
        }

        public string String() => Encoding.UTF8.GetString(Bytes(I32()));
    }
}
