using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Layout;
using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>
/// <see cref="OcrOptions.Accelerator"/>: networks from outside the package behind the pipeline, the bundled runtime for
/// the ones an accelerator leaves out, workers pooled per accelerator, and a network that misbehaves failing cleanly.
/// </summary>
[Collection("EndToEnd")]
public sealed class AcceleratorTests
{
    [Fact]
    public void AnAcceleratorRunningTheBundledModelsGivesTheSameResult()
    {
        RenderedPage page = TestPages.Invoice;
        byte[] gray = page.Gray();
        OcrPage expected = ImageOcr.RecognizeGray(gray, page.Width, page.Height);

        // Every network from the accelerator, with the recognizer's full softmax instead of the bundled most-probable
        // output and the detector on three planes instead of the gray variant, both of which give the same values.
        var accelerator = new BundledAccelerator(OcrNetworkKind.Detection, OcrNetworkKind.Recognition, OcrNetworkKind.Orientation);
        OcrPage actual = ImageOcr.RecognizeGray(gray, page.Width, page.Height, new OcrOptions { Accelerator = accelerator });

        Assert.Contains("Total NOK 17,589.45", actual.Text, StringComparison.Ordinal);
        AssertIdentical(expected, actual);
        Assert.Equal(3, accelerator.Created);
        Assert.True(accelerator.Runs[OcrNetworkKind.Detection] > 0);
        Assert.True(accelerator.Runs[OcrNetworkKind.Recognition] > 0);
        EnginePool.Clear();
    }

    [Fact]
    public void NetworksTheAcceleratorLeavesOutRunOnTheBundledRuntime()
    {
        RenderedPage page = TestPages.Latin;
        OcrPage expected = ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height);

        var detectionOnly = new BundledAccelerator(OcrNetworkKind.Detection);
        AssertIdentical(expected, ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { Accelerator = detectionOnly }));
        Assert.Equal(1, detectionOnly.Created);
        Assert.Equal(0, detectionOnly.Runs[OcrNetworkKind.Recognition]);

        var none = new BundledAccelerator();
        AssertIdentical(expected, ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { Accelerator = none }));
        Assert.Equal(0, none.Created);
        Assert.Equal(3, none.Asked);
        EnginePool.Clear();
    }

    [Fact]
    public void WorkersArePooledPerAcceleratorAndDisposeTheirNetworks()
    {
        EnginePool.Clear();
        var a = new BundledAccelerator(OcrNetworkKind.Orientation);
        var b = new BundledAccelerator(OcrNetworkKind.Orientation);
        EngineWorker plain = EnginePool.Rent(1);
        EngineWorker first = EnginePool.Rent(1, a);
        EngineWorker second = EnginePool.Rent(1, b);
        Assert.Equal(0, plain.AcceleratedCount);
        Assert.Equal(1, first.AcceleratedCount);
        EnginePool.Return(plain);
        EnginePool.Return(first);
        EnginePool.Return(second);

        Assert.Same(first, EnginePool.Rent(1, a));
        Assert.Same(second, EnginePool.Rent(1, b));
        Assert.Same(plain, EnginePool.Rent(1));
        Assert.Equal(1, a.Created);

        first.Dispose();
        second.Dispose();
        plain.Dispose();
        Assert.Equal(1, a.Disposed);
        Assert.Equal(1, b.Disposed);
        EnginePool.Clear();
    }

    [Fact]
    public void ANetworkThatReturnsTooLittleFailsWithInvalidOperation()
    {
        RenderedPage page = TestPages.Latin;
        foreach (OcrNetworkKind kind in Enum.GetValues<OcrNetworkKind>())
        {
            var accelerator = new BundledAccelerator(kind) { Truncate = true };
            Assert.Throws<InvalidOperationException>(() =>
                ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { Accelerator = accelerator }));

            // The worker that failed is not pooled, so its networks are let go at once.
            Assert.Equal(1, accelerator.Disposed);
        }

        EnginePool.Clear();
    }

    [Fact]
    public void AnAcceleratorThatThrowsWhileCreatingANetworkPassesItsException()
    {
        RenderedPage page = TestPages.Latin;
        var accelerator = new ThrowingAccelerator();
        var error = Assert.Throws<NotSupportedException>(() =>
            ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, new OcrOptions { Accelerator = accelerator }));
        Assert.Equal("no device", error.Message);
    }

    private static void AssertIdentical(OcrPage expected, OcrPage actual)
    {
        Assert.Equal(expected.Text, actual.Text);
        Assert.Equal(expected.Lines.Count, actual.Lines.Count);
        for (int i = 0; i < expected.Lines.Count; i++)
        {
            OcrLine a = expected.Lines[i], b = actual.Lines[i];
            Assert.Equal(a.Quad, b.Quad);
            Assert.Equal(a.Confidence, b.Confidence);
            Assert.Equal(a.Words.Select(w => (w.Text, w.Quad, w.Confidence)), b.Words.Select(w => (w.Text, w.Quad, w.Confidence)));
        }
    }

    /// <summary>An accelerator that runs the bundled models in contexts of its own, for the kinds it is given.</summary>
    private sealed class BundledAccelerator(params OcrNetworkKind[] kinds) : OcrAccelerator
    {
        private int _created;
        private int _asked;
        private int _disposed;

        public override string Name => "Bundled";

        public bool Truncate { get; init; }

        public int Created => _created;

        public int Asked => _asked;

        public int Disposed => _disposed;

        public Dictionary<OcrNetworkKind, int> Runs { get; } = Enum.GetValues<OcrNetworkKind>().ToDictionary(k => k, _ => 0);

        public override IOcrNetwork? CreateNetwork(OcrNetworkKind kind)
        {
            Interlocked.Increment(ref _asked);
            if (Array.IndexOf(kinds, kind) < 0)
            {
                return null;
            }

            Interlocked.Increment(ref _created);
            OcrModel model = kind switch
            {
                OcrNetworkKind.Detection => ModelStore.Detection,
                OcrNetworkKind.Recognition => ModelStore.Recognition,
                _ => ModelStore.Classifier,
            };
            return new Network(this, kind, model);
        }

        private sealed class Network(BundledAccelerator owner, OcrNetworkKind kind, OcrModel model) : IOcrNetwork
        {
            private readonly InferenceContext _context = new(1);

            public ReadOnlySpan<float> Run(ReadOnlySpan<float> input, ReadOnlySpan<int> shape, out int[] outputShape)
            {
                lock (owner.Runs)
                {
                    owner.Runs[kind]++;
                }

                ReadOnlySpan<float> output = model.Run(input, shape, _context, out outputShape);
                return owner.Truncate ? output[..(output.Length / 2)] : output;
            }

            public void Dispose()
            {
                Interlocked.Increment(ref owner._disposed);
                _context.Dispose();
            }
        }
    }

    private sealed class ThrowingAccelerator : OcrAccelerator
    {
        public override string Name => "Throwing";

        public override IOcrNetwork? CreateNetwork(OcrNetworkKind kind) => throw new NotSupportedException("no device");
    }
}
