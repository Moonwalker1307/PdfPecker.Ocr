using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Models;
using static PdfPecker.Ocr.Tests.Inference.TestModels;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>The model file format: round trips, storage precisions, and damaged data failing only with InvalidDataException.</summary>
public sealed class ModelFormatTests
{
    private static ModelDefinition Sample()
    {
        var graph = new GraphBuilder(4);
        int conv = graph.Add(OpKind.Conv, [0], ConvInts(3, 3, pt: 1, pl: 1, pb: 1, pr: 1, act: ActivationKind.Relu), [0, 0],
            new WeightDefinition([8, 3, 3, 3], Random(216, 1), WeightStorage.Float24), new WeightDefinition([8], Random(8, 2), WeightStorage.Float32));
        int pooled = graph.Add(OpKind.GlobalAveragePool, [conv], []);
        int flat = graph.Add(OpKind.Reshape, [pooled], [0, -1]);
        graph.Add(OpKind.Linear, [flat], [0, 0], [0, 0], new WeightDefinition([8, 5], Random(40, 3), WeightStorage.Float16), new WeightDefinition([5], Random(5, 4), WeightStorage.Float32));
        graph.Model.Name = "sample";
        graph.Model.Notice = "Test model; ünïcode notice.";
        graph.Model.Characters = ["", "a", "ß", "😀", " "];
        return graph.Model;
    }

    [Fact]
    public void RoundTripKeepsEverythingAtItsStoragePrecision()
    {
        var model = Sample();
        byte[] data = model.Serialize();
        Assert.Equal(data, Sample().Serialize());
        var read = ModelDefinition.Deserialize(data);
        Assert.Equal(model.Name, read.Name);
        Assert.Equal(model.Notice, read.Notice);
        Assert.Equal(model.InputShape, read.InputShape);
        Assert.Equal(model.Characters, read.Characters);
        Assert.Equal(model.Nodes.Count, read.Nodes.Count);
        for (int i = 0; i < model.Nodes.Count; i++)
        {
            Assert.Equal(model.Nodes[i].Op, read.Nodes[i].Op);
            Assert.Equal(model.Nodes[i].Inputs, read.Nodes[i].Inputs);
            Assert.Equal(model.Nodes[i].Ints, read.Nodes[i].Ints);
            Assert.Equal(model.Nodes[i].Floats, read.Nodes[i].Floats);
            Assert.Equal(model.Nodes[i].Weights, read.Nodes[i].Weights);
        }
        for (int i = 0; i < model.Weights.Count; i++)
        {
            var original = model.Weights[i];
            Assert.Equal(original.Shape, read.Weights[i].Shape);
            Assert.Equal(original.Storage, read.Weights[i].Storage);
            var expected = original.Data.Select(v => original.Storage switch
            {
                WeightStorage.Float24 => ModelDefinition.RoundToFloat24(v),
                WeightStorage.Float16 => (float)(Half)v,
                _ => v,
            });
            Assert.Equal(expected, read.Weights[i].Data);
        }
        Assert.Equal(data, read.Serialize());
        var output = Run(read, RandomTensor(5, 2, 3, 9, 7));
        Assert.Equal([2, 5], output.Shape);
    }

    [Fact]
    public void Float24RoundsToNearestEven()
    {
        Assert.Equal(1f, ModelDefinition.RoundToFloat24(1f));
        Assert.Equal(-0f, ModelDefinition.RoundToFloat24(-0f));
        Assert.Equal(float.PositiveInfinity, ModelDefinition.RoundToFloat24(float.PositiveInfinity));
        Assert.True(float.IsNaN(ModelDefinition.RoundToFloat24(float.NaN)));
        // Below half of the dropped byte rounds down, above rounds up, and a tie goes to the even neighbour.
        Assert.Equal(0x3F800000u, Bits(ModelDefinition.RoundToFloat24(Single(0x3F80007Fu))));
        Assert.Equal(0x3F800100u, Bits(ModelDefinition.RoundToFloat24(Single(0x3F800081u))));
        Assert.Equal(0x3F800000u, Bits(ModelDefinition.RoundToFloat24(Single(0x3F800080u))));
        Assert.Equal(0x3F800200u, Bits(ModelDefinition.RoundToFloat24(Single(0x3F800180u))));
        Assert.Equal(0x3F800100u, Bits(ModelDefinition.RoundToFloat24(Single(0x3F8000FFu))));
        foreach (float v in Random(10000, 9, 100f))
        {
            float r = ModelDefinition.RoundToFloat24(v);
            Assert.Equal(0u, Bits(r) & 0xFF);
            Assert.True(Math.Abs(r - v) <= Math.Abs(v) * Math.Pow(2, -16), $"{v} -> {r}");
        }

        static uint Bits(float f) => BitConverter.SingleToUInt32Bits(f);
        static float Single(uint bits) => BitConverter.UInt32BitsToSingle(bits);
    }

    [Fact]
    public void EveryTruncationIsRejected()
    {
        byte[] data = Sample().Serialize();
        for (int length = 0; length < data.Length; length++)
            Assert.Throws<InvalidDataException>(() => ModelDefinition.Deserialize(data.AsSpan(0, length)));
        byte[] longer = [.. data, 0];
        Assert.Throws<InvalidDataException>(() => ModelDefinition.Deserialize(longer));
    }

    [Fact]
    public void DamagedBytesFailOnlyWithInvalidDataException()
    {
        byte[] data = Sample().Serialize();
        var rng = new Random(42);
        int loaded = 0;
        for (int round = 0; round < 3000; round++)
        {
            byte[] damaged = (byte[])data.Clone();
            for (int i = 0, count = 1 + rng.Next(3); i < count; i++)
            {
                int at = rng.Next(damaged.Length);
                damaged[at] = rng.Next(4) switch
                {
                    0 => 0,
                    1 => 0xFF,
                    2 => (byte)(damaged[at] ^ (1 << rng.Next(8))),
                    _ => (byte)rng.Next(256),
                };
            }
            // Loading builds every layer, which checks the attributes and weight shapes against each other. Running a
            // damaged model is not tried: the bundled models are trusted resources, and a damaged padding can ask for
            // gigabytes of activations, which the runtime would allocate.
            try
            {
                OcrModel.Load(damaged);
                loaded++;
            }
            catch (InvalidDataException)
            {
            }
        }
        Assert.InRange(loaded, 1, 2999);
    }

    [Fact]
    public void BundledModelsRoundTrip()
    {
        foreach (string name in new[] { "cls" })
        {
            byte[] data = ModelStore.Inflate(name).ToArray();
            Assert.Equal(data, ModelDefinition.Deserialize(data).Serialize());
        }
    }
}
