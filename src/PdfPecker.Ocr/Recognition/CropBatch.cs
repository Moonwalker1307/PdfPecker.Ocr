using PdfPecker.Ocr.Detection;
using PdfPecker.Ocr.Inference;

namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// Samples the network input of a batch of line crops, one crop per batch item, on the threads of the inference
/// context when there is one. Sampling a line is pure arithmetic on the page, about a tenth of the recogniser's time
/// at one thread, and ran on one thread while the networks used them all. Each item writes only its own three planes
/// of the batch, so the input is the same on any number of threads.
/// </summary>
internal sealed unsafe class CropBatch
{
    private readonly InferenceContext? _context;
    private Item[] _items = new Item[16];
    private int _count;

    // The batch being sampled, for the workers; set only for the duration of Sample.
    private byte* _pixels;
    private int _pixelLength, _pageWidth, _pageHeight, _channels;
    private float* _tensor;
    private int _planeSize, _stride, _height;

    public CropBatch(InferenceContext? context)
    {
        _context = context;
    }

    /// <summary>Starts a new batch.</summary>
    public void Clear() => _count = 0;

    /// <summary>
    /// Adds the next batch item: the part of <paramref name="crop"/> from <paramref name="windowX"/> that is
    /// <paramref name="windowWidth"/> long, resized to <paramref name="width"/> pixels (see <see cref="LineCrop.Sample"/>).
    /// </summary>
    public void Add(LineCrop crop, int width, double windowX, double windowWidth)
    {
        if (_count == _items.Length) Array.Resize(ref _items, _count * 2);
        _items[_count++] = new Item(crop, width, windowX, windowWidth);
    }

    /// <summary>
    /// Samples every item into <paramref name="tensor"/>, item k into the three planes from k * 3 * planeSize, rows
    /// <paramref name="stride"/> floats apart and <paramref name="height"/> rows high.
    /// </summary>
    public void Sample(PixelView page, Span<float> tensor, int planeSize, int stride, int height)
    {
        if ((long)_count * 3 * planeSize > tensor.Length) throw new ArgumentException("The tensor is smaller than the batch.", nameof(tensor));
        fixed (byte* pixels = page.Data)
        fixed (float* planes = tensor)
        {
            _pixels = pixels;
            _pixelLength = page.Data.Length;
            _pageWidth = page.Width;
            _pageHeight = page.Height;
            _channels = page.Channels;
            _tensor = planes;
            _planeSize = planeSize;
            _stride = stride;
            _height = height;
            try
            {
                if (_context is null)
                {
                    for (int k = 0; k < _count; k++) SampleItem(this, k, 0);
                }
                else
                {
                    _context.For(_count, this, &SampleItem);
                }
            }
            finally
            {
                _pixels = null;
                _tensor = null;
            }
        }
    }

    private static void SampleItem(object state, int k, int worker)
    {
        var batch = (CropBatch)state;
        var page = new PixelView(new ReadOnlySpan<byte>(batch._pixels, batch._pixelLength), batch._pageWidth, batch._pageHeight, batch._channels);
        Item item = batch._items[k];
        var planes = new Span<float>(batch._tensor + ((long)k * 3 * batch._planeSize), 3 * batch._planeSize);
        item.Crop.Sample(page, planes, batch._planeSize, batch._stride, item.Width, batch._height, item.WindowX, item.WindowWidth);
    }

    private readonly record struct Item(LineCrop Crop, int Width, double WindowX, double WindowWidth);
}
