namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>A stand-in network whose output a test computes from the input and its shape.</summary>
internal sealed class FunctionNetwork(Func<float[], int[], (float[] Output, int[] Shape)> run) : INetwork
{
    public int Calls { get; private set; }

    public List<int[]> InputShapes { get; } = [];

    public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
    {
        Calls++;
        int[] s = shape.ToArray();
        InputShapes.Add(s);
        (float[] output, int[] o) = run(input.ToArray(), s);
        outputShape = o;
        return output;
    }
}
