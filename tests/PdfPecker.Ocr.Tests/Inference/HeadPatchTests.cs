using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.Inference;

/// <summary>The recogniser's retrained Nordic columns (<see cref="HeadPatch"/>) and how they are loaded.</summary>
public sealed class HeadPatchTests
{
    /// <summary>The classes tools/HeadTuner retrains; 0, O and o compete in the refit but keep their weights.</summary>
    private static readonly string[] Retrained = ["A", "Å", "Æ", "E", "É", "È", "Ê", "e", "Ø", "ø", "a", "å", "æ", "∅", "Ä", "Ö", "ö", "ä"];

    [Fact]
    public void TheBundledPatchRetrainsOnlyTheNordicLettersAndTheirLookalikes()
    {
        HeadPatch patch = ModelStore.RecognitionHeadPatch;
        Assert.Equal(Retrained, patch.Columns.Select(c => c.Character));
        var definition = Converted();
        var (weights, _) = HeadPatch.FinalLayer(definition);
        Assert.Equal(weights.Shape[0], patch.Features);
        foreach (var column in patch.Columns)
        {
            Assert.Equal(column.Character, definition.Characters[column.Index]);
            Assert.All(column.Weights, w => Assert.True(float.IsFinite(w)));
            Assert.True(float.IsFinite(column.Bias));
        }
    }

    [Fact]
    public void ApplyingThePatchChangesOnlyItsColumns()
    {
        var converted = Converted();
        var patched = Converted();
        HeadPatch patch = ModelStore.RecognitionHeadPatch;
        patch.ApplyTo(patched);
        var (w0, b0) = HeadPatch.FinalLayer(converted);
        var (w1, b1) = HeadPatch.FinalLayer(patched);
        int k = w0.Shape[0], n = w0.Shape[1];
        var changed = patch.Columns.Select(c => c.Index).ToHashSet();
        for (int c = 0; c < n; c++)
        {
            bool same = b0.Data[c] == b1.Data[c];
            for (int i = 0; i < k && same; i++) same = w0.Data[(i * n) + c] == w1.Data[(i * n) + c];
            Assert.True(same != changed.Contains(c), $"class {c} ({converted.Characters[c]})");
        }

        // Every other weight of the network is the converted model's.
        for (int i = 0; i < converted.Weights.Count; i++)
        {
            if (ReferenceEquals(converted.Weights[i], w0) || ReferenceEquals(converted.Weights[i], b0)) continue;
            Assert.True(converted.Weights[i].Data.AsSpan().SequenceEqual(patched.Weights[i].Data), $"weight {i}");
        }
    }

    [Fact]
    public void ThePatchRoundTrips()
    {
        var patch = HeadPatch.Create(3, [new HeadPatch.Column(5, "Å", [1f, -2.5f, 3e-7f], 0.25f), new HeadPatch.Column(9, "æ", [0f, 1f, 2f], -1f)]);
        var read = HeadPatch.Read(patch.Write());
        Assert.Equal(3, read.Features);
        Assert.Equal(2, read.Columns.Length);
        Assert.Equal("Å", read.Columns[0].Character);
        Assert.Equal([1f, -2.5f, 3e-7f], read.Columns[0].Weights);
        Assert.Equal(-1f, read.Columns[1].Bias);
        Assert.Equal(patch.Write(), read.Write());
    }

    [Fact]
    public void DamagedPatchesFailWithInvalidDataException()
    {
        byte[] good = ModelStore.RecognitionHeadPatch.Write();
        Assert.Throws<InvalidDataException>(() => HeadPatch.Read(good.AsSpan(0, good.Length - 1)));
        Assert.Throws<InvalidDataException>(() => HeadPatch.Read([.. good, 0]));
        Assert.Throws<InvalidDataException>(() => HeadPatch.Read([]));
        byte[] magic = (byte[])good.Clone();
        magic[0] = (byte)'X';
        Assert.Throws<InvalidDataException>(() => HeadPatch.Read(magic));
        byte[] version = (byte[])good.Clone();
        version[8] = 2;
        Assert.Throws<InvalidDataException>(() => HeadPatch.Read(version));
        byte[] count = (byte[])good.Clone();
        count[14] = count[15] = count[16] = count[17] = 0xFF;
        Assert.Throws<InvalidDataException>(() => HeadPatch.Read(count));
    }

    [Fact]
    public void APatchForAnotherModelIsRefused()
    {
        var definition = Converted();
        int a = Array.IndexOf(definition.Characters, "A");
        var (weights, _) = HeadPatch.FinalLayer(definition);
        int k = weights.Shape[0];

        // A class that holds another character, a class outside the list, and a layer of another width.
        Assert.Throws<InvalidDataException>(() => HeadPatch.Create(k, [new HeadPatch.Column(a, "B", new float[k], 0)]).ApplyTo(definition));
        Assert.Throws<InvalidDataException>(() => HeadPatch.Create(k, [new HeadPatch.Column(1 << 20, "A", new float[k], 0)]).ApplyTo(definition));
        Assert.Throws<InvalidDataException>(() => HeadPatch.Create(k + 1, [new HeadPatch.Column(a, "A", new float[k + 1], 0)]).ApplyTo(definition));
    }

    private static ModelDefinition Converted()
    {
        using var data = ModelStore.Inflate("rec");
        return ModelDefinition.Deserialize(data.GetBuffer().AsSpan(0, (int)data.Length));
    }
}
