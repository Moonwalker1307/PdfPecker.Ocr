using PdfPecker.Ocr.Layout;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>The pool of engine workers: reuse between calls, a bound on idle workers, and their arenas and themselves released when idle.</summary>
[Collection("EndToEnd")]
public sealed class EnginePoolTests
{
    [Fact]
    public void WorkersAreReusedByThreadCount()
    {
        EnginePool.Clear();
        EngineWorker one = EnginePool.Rent(1);
        EngineWorker two = EnginePool.Rent(2);
        EnginePool.Return(one);
        EnginePool.Return(two);
        Assert.Same(two, EnginePool.Rent(2));
        Assert.Same(one, EnginePool.Rent(1));
        Assert.NotSame(one, EnginePool.Rent(1));
        EnginePool.Clear();
    }

    [Fact]
    public void IdleWorkersAreBoundedAndReleased()
    {
        // As an earlier test or call leaves the pool: a return arms the timer with the default 30 seconds, and Clear
        // must disarm it so that the returns below arm it with the short timeout.
        EnginePool.Return(EnginePool.Rent(1));
        EnginePool.Clear();
        TimeSpan timeout = EnginePool.IdleTimeout;
        try
        {
            EnginePool.IdleTimeout = TimeSpan.FromMilliseconds(200);
            var workers = Enumerable.Range(0, EnginePool.MaxIdle + 2).Select(_ => EnginePool.Rent(1)).ToList();
            workers.ForEach(EnginePool.Return);
            Assert.Equal(EnginePool.MaxIdle, EnginePool.IdleCount);

            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(20);
            while (EnginePool.IdleCount > 0 && DateTime.UtcNow < deadline)
            {
                Thread.Sleep(50);
            }

            Assert.Equal(0, EnginePool.IdleCount);
        }
        finally
        {
            EnginePool.IdleTimeout = timeout;
            EnginePool.Clear();
        }
    }

    [Fact]
    public void IdleWorkersFreeTheirArenaFirst()
    {
        EnginePool.Return(EnginePool.Rent(1));
        EnginePool.Clear();
        TimeSpan arenaTimeout = EnginePool.ArenaTimeout;
        try
        {
            EnginePool.ArenaTimeout = TimeSpan.FromMilliseconds(500);
            RenderedPage page = TestPages.Numbers;
            byte[] gray = page.Gray();
            var options = new OcrOptions { MaxDegreeOfParallelism = 1 };
            Assert.Equal(TestPages.NumbersText, ImageOcr.RecognizeGray(gray, page.Width, page.Height, options).Text);

            EngineWorker worker = EnginePool.Rent(1);
            Assert.True(worker.ArenaBytes > 0);
            EnginePool.Return(worker);
            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(20);
            while (worker.ArenaBytes > 0 && DateTime.UtcNow < deadline)
            {
                Thread.Sleep(20);
            }

            // The worker stays in the pool without its arena, and the next call allocates one again.
            Assert.Equal(0, worker.ArenaBytes);
            Assert.Equal(1, EnginePool.IdleCount);
            Assert.Equal(TestPages.NumbersText, ImageOcr.RecognizeGray(gray, page.Width, page.Height, options).Text);
            Assert.Same(worker, EnginePool.Rent(1));
            Assert.True(worker.ArenaBytes > 0);
            EnginePool.Return(worker);
        }
        finally
        {
            EnginePool.ArenaTimeout = arenaTimeout;
            EnginePool.Clear();
        }
    }

    [Fact]
    public void ParallelCallsGiveTheSameText()
    {
        RenderedPage page = TestPages.Numbers;
        byte[] gray = page.Gray();
        string[] texts = new string[6];
        Parallel.For(0, texts.Length, new ParallelOptions { MaxDegreeOfParallelism = 3 }, i =>
            texts[i] = ImageOcr.RecognizeGray(gray, page.Width, page.Height, new OcrOptions { MaxDegreeOfParallelism = 1 + (i % 2) }).Text);
        Assert.All(texts, t => Assert.Equal(TestPages.NumbersText, t));
    }
}
