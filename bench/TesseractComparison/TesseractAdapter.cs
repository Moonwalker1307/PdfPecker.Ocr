using System.Runtime.InteropServices;

namespace TesseractComparison;

/// <summary>
/// Tesseract through its C API: the LSTM engine, one TessBaseAPI per adapter, raw pixels through TessBaseAPISetImage (no
/// Leptonica file reading), and the word iterator for boxes and confidences. The thread count is not an API setting: Tesseract's
/// OpenMP build takes it from OMP_THREAD_LIMIT, which the parent process sets in the child's environment.
/// </summary>
sealed unsafe class TesseractAdapter(string tessdata, string languageMode, string psmMode) : IOcrEngineAdapter
{
    IntPtr _api;
    string? _language;
    int _threads;

    public string Name => "tesseract";
    public bool InProcess => true;
    public bool Managed => false;

    public string Description => $"Tesseract {Marshal.PtrToStringUTF8(TesseractNative.TessVersion())}, LSTM engine, models in {tessdata}, " +
        $"language {languageMode}, page segmentation {psmMode}, OMP_THREAD_LIMIT={Environment.GetEnvironmentVariable("OMP_THREAD_LIMIT") ?? "unset"}";

    public void Initialize(int threads) => _threads = threads;

    public void Prepare(CorpusPage page)
    {
        string language = languageMode switch
        {
            "auto" => page.Lang switch { "da" => "dan", "de" => "deu", "fr" => "fra", "multi" => "eng+deu+fra", _ => "eng" },
            "multi" => "eng+dan+deu+fra",
            _ => languageMode,
        };
        if (language != _language)
        {
            if (_api != IntPtr.Zero) { TesseractNative.TessBaseAPIEnd(_api); TesseractNative.TessBaseAPIDelete(_api); }
            _api = TesseractNative.TessBaseAPICreate();
            if (TesseractNative.TessBaseAPIInit2(_api, tessdata, language, TesseractNative.OemLstmOnly) != 0)
                throw new InvalidOperationException($"Tesseract could not load '{language}' from {tessdata}");
            _language = language;
        }
        // Pages that are upside down need the orientation detection of mode 1; "auto" uses it for those pages only, which
        // is what a caller who knows the scan is upside down would do, and mode 3 (the default) for everything else.
        int psm = psmMode switch { "auto" => page.Rotation == 180 ? TesseractNative.PsmAutoOsd : TesseractNative.PsmAuto, _ => int.Parse(psmMode, System.Globalization.CultureInfo.InvariantCulture) };
        TesseractNative.TessBaseAPISetPageSegMode(_api, psm);
        if (page.Dpi > 0) TesseractNative.TessBaseAPISetSourceResolution(_api, page.Dpi);
    }

    public OcrResult Recognize(ReadOnlySpan<byte> pixels, int width, int height, PixelFormat format)
    {
        int bpp = (int)format;
        fixed (byte* p = pixels)
            TesseractNative.TessBaseAPISetImage(_api, p, width, height, bpp, width * bpp);
        if (TesseractNative.TessBaseAPIRecognize(_api, IntPtr.Zero) != 0) throw new InvalidOperationException("Tesseract recognition failed");
        string text = TesseractNative.TakeString(TesseractNative.TessBaseAPIGetUTF8Text(_api));
        var words = new List<Box>();
        IntPtr it = TesseractNative.TessBaseAPIGetIterator(_api);
        if (it != IntPtr.Zero)
        {
            IntPtr pageIt = TesseractNative.TessResultIteratorGetPageIterator(it);
            do
            {
                string word = TesseractNative.TakeString(TesseractNative.TessResultIteratorGetUTF8Text(it, TesseractNative.RilWord));
                if (word.Length == 0) continue;
                int l, t, r, b;
                if (TesseractNative.TessPageIteratorBoundingBox(pageIt, TesseractNative.RilWord, &l, &t, &r, &b) == 0) continue;
                float conf = TesseractNative.TessResultIteratorConfidence(it, TesseractNative.RilWord);
                words.Add(new Box(word.Trim(), l, t, r - l, b - t, conf / 100.0));
            }
            while (TesseractNative.TessResultIteratorNext(it, TesseractNative.RilWord) != 0);
            TesseractNative.TessResultIteratorDelete(it);
        }
        TesseractNative.TessBaseAPIClear(_api);
        return new OcrResult { Text = text, Words = words };
    }

    public void Dispose()
    {
        if (_api == IntPtr.Zero) return;
        TesseractNative.TessBaseAPIEnd(_api);
        TesseractNative.TessBaseAPIDelete(_api);
        _api = IntPtr.Zero;
    }
}
