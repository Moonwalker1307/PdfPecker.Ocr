using PdfPecker.Ocr.Models;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>
/// Cancellation through each public entry point, at known operators of the engine's networks rather than after a
/// delay, which a fast machine can beat and a busy one overshoot. Each test cancels at one operator and requires the
/// call to throw with not one more operator run and no network run started, which fails when any layer between the
/// entry point and the networks drops the token or stops checking it.
/// </summary>
[Collection("EndToEnd")]
public sealed class CancellationTests
{
    private static readonly OcrOptions s_options = new() { MaxDegreeOfParallelism = 2 };

    private static readonly Lazy<byte[]> s_twoPagePdf = new(() =>
    {
        RenderedPage invoice = TestPages.Invoice;
        return ImageToPdf.Convert([invoice.Png, invoice.Png], null, new ImageToPdfOptions { FitPageToImage = true, Dpi = 300 });
    });

    /// <summary>The operators of each run when the invoice is read as gray pixels, counted once without cancelling.</summary>
    private static readonly Lazy<CancelProbe> s_counted = new(() =>
    {
        RenderedPage invoice = TestPages.Invoice;
        var probe = new CancelProbe(_ => false);
        try
        {
            ImageOcr.RecognizeGray(invoice.Gray(), invoice.Width, invoice.Height, s_options, probe.Token);
        }
        finally
        {
            // Only its counts are used afterwards; it must not go on observing the flow it was made on.
            probe.Dispose();
        }

        return probe;
    });

    [Theory]
    [InlineData("gray")]
    [InlineData("rgba")]
    [InlineData("png")]
    [InlineData("png stream")]
    public void ImageEntryPointsStopInsideTheDetector(string entry)
    {
        RenderedPage invoice = TestPages.Invoice;
        using var probe = new CancelProbe(s => CancelProbe.IsDetection(s.Model) && s.Index == 5);
        Assert.ThrowsAny<OperationCanceledException>(() => ReadImage(entry, invoice, probe.Token));
        probe.AssertStoppedAtOnce();
    }

    [Fact]
    public void PdfRecognizeStopsInsideAPage()
    {
        // Within the first page's recognition, so a token checked only between pages lets the page finish.
        using var probe = new CancelProbe(s => s.Model == ModelStore.RecognitionArgMax && s.Run == 1 && s.Index == 3);
        Assert.ThrowsAny<OperationCanceledException>(() => PdfOcr.Recognize(s_twoPagePdf.Value, s_options, probe.Token));
        probe.AssertStoppedAtOnce();
        Assert.Equal(1, probe.Runs(ModelStore.DetectionGray) + probe.Runs(ModelStore.Detection));
    }

    [Fact]
    public void MakeSearchableStopsInsideAPage()
    {
        using var probe = new CancelProbe(s => s.Model == ModelStore.RecognitionArgMax && s.Run == 1 && s.Index == 3);
        Assert.ThrowsAny<OperationCanceledException>(() => PdfOcr.MakeSearchable(s_twoPagePdf.Value, s_options, probe.Token));
        probe.AssertStoppedAtOnce();
    }

    [Fact]
    public void ToSearchablePdfStopsInsideAPage()
    {
        RenderedPage invoice = TestPages.Invoice;
        using var probe = new CancelProbe(s => s.Model == ModelStore.RecognitionArgMax && s.Run == 1 && s.Index == 3);
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.ToSearchablePdf([invoice.Png, invoice.Png], s_options, probe.Token));
        probe.AssertStoppedAtOnce();
    }

    [Fact]
    public void OrientationStopsInsideItsNetwork()
    {
        RenderedPage invoice = TestPages.Invoice;
        using var probe = new CancelProbe(s => s.Model == ModelStore.Classifier && s.Run == 0 && s.Index == 3);
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.RecognizeRgba(invoice.Rgba, invoice.Width, invoice.Height, s_options, probe.Token));
        probe.AssertStoppedAtOnce();
        Assert.Equal(0, probe.Runs(ModelStore.RecognitionArgMax));
    }

    [Fact]
    public void OrientationStopsBetweenItsBatches()
    {
        // After the last operator of the first batch of windows, the network has finished and only the classifier's
        // own loop can see the token before the next batch starts.
        CancelProbe counted = s_counted.Value;
        Assert.True(counted.Runs(ModelStore.Classifier) >= 2, "The page has only one batch of orientation windows.");
        int last = counted.LastStep(ModelStore.Classifier, 0);
        AssertStopsAfter(s => s.Model == ModelStore.Classifier && s.Run == 0 && s.Index == last);
    }

    [Fact]
    public void RecognitionStopsBetweenItsBatches()
    {
        CancelProbe counted = s_counted.Value;
        Assert.True(counted.Runs(ModelStore.RecognitionArgMax) >= 2, "The page has only one batch of lines.");
        int last = counted.LastStep(ModelStore.RecognitionArgMax, 0);
        AssertStopsAfter(s => s.Model == ModelStore.RecognitionArgMax && s.Run == 0 && s.Index == last);
    }

    [Fact]
    public void DetectionStopsBeforeItsPostProcessing()
    {
        // After the detection network's last operator, so the post-processing, the line crops and the orientation
        // step have to see the token: a pipeline that hands them no token runs them and starts the next network.
        CancelProbe counted = s_counted.Value;
        int last = counted.LastStep(ModelStore.DetectionGray, 0);
        AssertStopsAfter(s => s.Model == ModelStore.DetectionGray && s.Run == 0 && s.Index == last);

        // The engine is in order after all these cancelled pages.
        RenderedPage small = TestPages.Small;
        Assert.Equal(TestPages.SmallText, ImageOcr.RecognizeRgba(small.Rgba, small.Width, small.Height, s_options).Text);
    }

    [Fact]
    public async Task AProbeSeesOnlyItsOwnFlow()
    {
        // The observer flows with the execution context, so a page read at the same time on a flow of its own, as by a
        // test running in parallel, is neither observed nor cancelled, while the probe's own flow is.
        RenderedPage small = TestPages.Small;
        using var probe = new CancelProbe(_ => true);
        Task<OcrPage> other;
        using (ExecutionContext.SuppressFlow())
        {
            other = Task.Run(() => ImageOcr.RecognizeRgba(small.Rgba, small.Width, small.Height, s_options));
        }

        Assert.Equal(TestPages.SmallText, (await other).Text);
        Assert.False(probe.Cancelled, "The probe saw a page read on another flow.");
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.RecognizeRgba(small.Rgba, small.Width, small.Height, s_options, probe.Token));
        probe.AssertStoppedAtOnce();
    }

    /// <summary>Reads the invoice as gray pixels, cancelling after the operator <paramref name="cancelAt"/> picks.</summary>
    private static void AssertStopsAfter(Func<CancelProbe.Step, bool> cancelAt)
    {
        RenderedPage invoice = TestPages.Invoice;
        byte[] gray = invoice.Gray();
        using var probe = new CancelProbe(cancelAt);
        Assert.ThrowsAny<OperationCanceledException>(() => ImageOcr.RecognizeGray(gray, invoice.Width, invoice.Height, s_options, probe.Token));
        probe.AssertStoppedAtOnce();
    }

    private static OcrPage ReadImage(string entry, RenderedPage page, CancellationToken token) => entry switch
    {
        "gray" => ImageOcr.RecognizeGray(page.Gray(), page.Width, page.Height, s_options, token),
        "rgba" => ImageOcr.RecognizeRgba(page.Rgba, page.Width, page.Height, s_options, token),
        "png" => ImageOcr.Recognize(page.Png, s_options, token),
        "png stream" => ImageOcr.Recognize(new MemoryStream(page.Png), s_options, token),
        _ => throw new ArgumentOutOfRangeException(nameof(entry)),
    };
}
