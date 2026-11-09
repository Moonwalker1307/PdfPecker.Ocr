using System.Text;

namespace ModelBuilder;

/// <summary>
/// The parts of the ONNX protobuf schema the bundled models use, read with a minimal protobuf decoder so the builder
/// needs no package. Field numbers are those of onnx.proto (ModelProto, GraphProto, NodeProto, AttributeProto,
/// TensorProto, ValueInfoProto).
/// </summary>
internal sealed class OnnxModel
{
    public List<OnnxNode> Nodes { get; } = [];
    public List<OnnxTensor> Initializers { get; } = [];
    public List<OnnxValueInfo> Inputs { get; } = [];
    public List<OnnxValueInfo> Outputs { get; } = [];
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);
    public long OpsetVersion { get; private set; }

    public static OnnxModel Parse(byte[] data)
    {
        var model = new OnnxModel();
        var r = new ProtoReader(data);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 7: ParseGraph(r.Message(), model); break;
                case 8:
                {
                    var opset = r.Message();
                    string domain = "";
                    long version = 0;
                    while (opset.Next(out int f, out int w))
                    {
                        if (f == 1) domain = opset.String();
                        else if (f == 2) version = opset.Varint();
                        else opset.Skip(w);
                    }
                    if (domain.Length == 0) model.OpsetVersion = version;
                    break;
                }
                case 14:
                {
                    var entry = r.Message();
                    string key = "", value = "";
                    while (entry.Next(out int f, out int w))
                    {
                        if (f == 1) key = entry.String();
                        else if (f == 2) value = entry.String();
                        else entry.Skip(w);
                    }
                    model.Metadata[key] = value;
                    break;
                }
                default: r.Skip(wire); break;
            }
        }
        return model;
    }

    private static void ParseGraph(ProtoReader r, OnnxModel model)
    {
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1: model.Nodes.Add(OnnxNode.Parse(r.Message())); break;
                case 5: model.Initializers.Add(OnnxTensor.Parse(r.Message())); break;
                case 11: model.Inputs.Add(OnnxValueInfo.Parse(r.Message())); break;
                case 12: model.Outputs.Add(OnnxValueInfo.Parse(r.Message())); break;
                default: r.Skip(wire); break;
            }
        }
    }
}

internal sealed class OnnxNode
{
    public string OpType { get; set; } = "";
    public string Domain { get; set; } = "";
    public List<string> Inputs { get; } = [];
    public List<string> Outputs { get; } = [];
    public Dictionary<string, OnnxAttribute> Attributes { get; } = new(StringComparer.Ordinal);

    public static OnnxNode Parse(ProtoReader r)
    {
        var node = new OnnxNode();
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1: node.Inputs.Add(r.String()); break;
                case 2: node.Outputs.Add(r.String()); break;
                case 4: node.OpType = r.String(); break;
                case 5:
                {
                    var a = OnnxAttribute.Parse(r.Message());
                    node.Attributes[a.Name] = a;
                    break;
                }
                case 7: node.Domain = r.String(); break;
                default: r.Skip(wire); break;
            }
        }
        if (node.Domain.Length != 0 && node.Domain != "ai.onnx") throw new InvalidDataException($"Operator domain {node.Domain} is not supported.");
        return node;
    }
}

internal sealed class OnnxAttribute
{
    public string Name { get; private set; } = "";
    public float F { get; private set; }
    public long I { get; private set; }
    public byte[] S { get; private set; } = [];
    public OnnxTensor? T { get; private set; }
    public List<float> Floats { get; } = [];
    public List<long> Ints { get; } = [];

    public string Text => Encoding.UTF8.GetString(S);

    public static OnnxAttribute Parse(ProtoReader r)
    {
        var a = new OnnxAttribute();
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1: a.Name = r.String(); break;
                case 2: a.F = r.Fixed32Float(); break;
                case 3: a.I = r.Varint(); break;
                case 4: a.S = r.Bytes().ToArray(); break;
                case 5: a.T = OnnxTensor.Parse(r.Message()); break;
                case 7:
                    if (wire == 2) { var p = r.Message(); while (!p.AtEnd) a.Floats.Add(p.Fixed32Float()); }
                    else a.Floats.Add(r.Fixed32Float());
                    break;
                case 8:
                    if (wire == 2) { var p = r.Message(); while (!p.AtEnd) a.Ints.Add(p.Varint()); }
                    else a.Ints.Add(r.Varint());
                    break;
                default: r.Skip(wire); break;
            }
        }
        return a;
    }
}

/// <summary>A tensor in the two element types the models use: float32 and int64 (int32 and doubles are converted).</summary>
internal sealed class OnnxTensor
{
    public const int Float = 1, Int32 = 6, Int64 = 7, Double = 11;

    public string Name { get; set; } = "";
    public long[] Dims { get; set; } = [];
    public int DataType { get; set; }
    public float[]? FloatData { get; set; }
    public long[]? IntData { get; set; }

    public long Count => Dims.Aggregate(1L, (a, b) => a * b);

    public static OnnxTensor Parse(ProtoReader r)
    {
        var t = new OnnxTensor();
        var dims = new List<long>();
        var floats = new List<float>();
        var ints = new List<long>();
        var doubles = new List<double>();
        ReadOnlyMemory<byte> raw = default;
        bool hasRaw = false;
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1:
                    if (wire == 2) { var p = r.Message(); while (!p.AtEnd) dims.Add(p.Varint()); }
                    else dims.Add(r.Varint());
                    break;
                case 2: t.DataType = (int)r.Varint(); break;
                case 4:
                    if (wire == 2) { var p = r.Message(); while (!p.AtEnd) floats.Add(p.Fixed32Float()); }
                    else floats.Add(r.Fixed32Float());
                    break;
                case 5:
                case 7:
                    if (wire == 2) { var p = r.Message(); while (!p.AtEnd) ints.Add(p.Varint()); }
                    else ints.Add(r.Varint());
                    break;
                case 8: t.Name = r.String(); break;
                case 9: raw = r.Bytes(); hasRaw = true; break;
                case 10:
                    if (wire == 2) { var p = r.Message(); while (!p.AtEnd) doubles.Add(BitConverter.Int64BitsToDouble((long)p.Fixed64())); }
                    else doubles.Add(BitConverter.Int64BitsToDouble((long)r.Fixed64()));
                    break;
                default: r.Skip(wire); break;
            }
        }
        t.Dims = dims.ToArray();
        long count = t.Count;
        byte[] bytes = raw.ToArray();
        switch (t.DataType)
        {
            case Float:
                t.FloatData = hasRaw ? Enumerable.Range(0, (int)count).Select(i => BitConverter.ToSingle(bytes, i * 4)).ToArray() : floats.ToArray();
                break;
            case Double:
                t.FloatData = hasRaw ? Enumerable.Range(0, (int)count).Select(i => (float)BitConverter.ToDouble(bytes, i * 8)).ToArray() : doubles.Select(d => (float)d).ToArray();
                t.DataType = Float;
                break;
            case Int64:
                t.IntData = hasRaw ? Enumerable.Range(0, (int)count).Select(i => BitConverter.ToInt64(bytes, i * 8)).ToArray() : ints.ToArray();
                break;
            case Int32:
                t.IntData = hasRaw ? Enumerable.Range(0, (int)count).Select(i => (long)BitConverter.ToInt32(bytes, i * 4)).ToArray() : ints.ToArray();
                t.DataType = Int64;
                break;
            default:
                throw new InvalidDataException($"Tensor {t.Name} has unsupported element type {t.DataType}.");
        }
        long actual = t.FloatData?.LongLength ?? t.IntData!.LongLength;
        if (actual != count) throw new InvalidDataException($"Tensor {t.Name} holds {actual} values for {count} elements.");
        return t;
    }
}

internal sealed class OnnxValueInfo
{
    public string Name { get; private set; } = "";

    /// <summary>The dimensions, with -1 for symbolic ones.</summary>
    public List<long> Dims { get; } = [];

    public static OnnxValueInfo Parse(ProtoReader r)
    {
        var v = new OnnxValueInfo();
        while (r.Next(out int field, out int wire))
        {
            if (field == 1) v.Name = r.String();
            else if (field == 2)
            {
                var type = r.Message();
                while (type.Next(out int tf, out int tw))
                {
                    if (tf != 1) { type.Skip(tw); continue; }
                    var tensor = type.Message();
                    while (tensor.Next(out int f, out int w))
                    {
                        if (f != 2) { tensor.Skip(w); continue; }
                        var shape = tensor.Message();
                        while (shape.Next(out int sf, out int sw))
                        {
                            if (sf != 1) { shape.Skip(sw); continue; }
                            var dim = shape.Message();
                            long value = -1;
                            while (dim.Next(out int df, out int dw))
                            {
                                if (df == 1) value = dim.Varint();
                                else dim.Skip(dw);
                            }
                            v.Dims.Add(value > 0 ? value : -1);
                        }
                    }
                }
            }
            else r.Skip(wire);
        }
        return v;
    }
}

/// <summary>A minimal protobuf wire-format reader over a byte buffer.</summary>
internal struct ProtoReader(ReadOnlyMemory<byte> data)
{
    private readonly ReadOnlyMemory<byte> _data = data;
    private int _position;

    public readonly bool AtEnd => _position >= _data.Length;

    public bool Next(out int field, out int wire)
    {
        if (AtEnd)
        {
            field = wire = 0;
            return false;
        }
        ulong key = (ulong)Varint();
        field = (int)(key >> 3);
        wire = (int)(key & 7);
        return true;
    }

    public long Varint()
    {
        ulong result = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            if (_position >= _data.Length) throw new InvalidDataException("Truncated protobuf varint.");
            byte b = _data.Span[_position++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return (long)result;
        }
        throw new InvalidDataException("Overlong protobuf varint.");
    }

    public float Fixed32Float()
    {
        var value = BitConverter.ToSingle(Take(4).Span);
        return value;
    }

    public ulong Fixed64() => BitConverter.ToUInt64(Take(8).Span);

    public ReadOnlyMemory<byte> Bytes() => Take(checked((int)Varint()));

    public string String() => Encoding.UTF8.GetString(Bytes().Span);

    public ProtoReader Message() => new(Bytes());

    public void Skip(int wire)
    {
        switch (wire)
        {
            case 0: Varint(); break;
            case 1: Take(8); break;
            case 2: Bytes(); break;
            case 5: Take(4); break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }

    private ReadOnlyMemory<byte> Take(int count)
    {
        if (count < 0 || count > _data.Length - _position) throw new InvalidDataException("Truncated protobuf field.");
        var result = _data.Slice(_position, count);
        _position += count;
        return result;
    }
}
