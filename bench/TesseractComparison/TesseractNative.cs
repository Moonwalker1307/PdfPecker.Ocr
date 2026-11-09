using System.Reflection;
using System.Runtime.InteropServices;

namespace TesseractComparison;

/// <summary>
/// Tesseract's C API (include/tesseract/capi.h of 5.5.4, whose declarations these follow), imported with source-generated
/// LibraryImport so that the call sites need no runtime marshalling code and the app stays Native AOT compatible. BOOL in
/// that header is a C int. The library is found through <see cref="SetLibraryPath"/>, because it is built by
/// tools/Tesseract/build.sh and sits in no system folder.
/// </summary>
static unsafe partial class TesseractNative
{
    const string Lib = "tesseract";

    public const int OemLstmOnly = 1;
    public const int PsmAutoOsd = 1;
    public const int PsmAuto = 3;
    public const int RilTextline = 2;
    public const int RilWord = 3;

    static string? _path;

    public static void SetLibraryPath(string path)
    {
        _path = path;
        NativeLibrary.SetDllImportResolver(typeof(TesseractNative).Assembly, Resolve);
    }

    static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath) =>
        name == Lib && _path is not null ? NativeLibrary.Load(_path) : IntPtr.Zero;

    [LibraryImport(Lib)] public static partial IntPtr TessVersion();
    [LibraryImport(Lib)] public static partial void TessDeleteText(IntPtr text);
    [LibraryImport(Lib)] public static partial IntPtr TessBaseAPICreate();
    [LibraryImport(Lib)] public static partial void TessBaseAPIDelete(IntPtr handle);
    [LibraryImport(Lib)] public static partial void TessBaseAPIEnd(IntPtr handle);
    [LibraryImport(Lib)] public static partial void TessBaseAPIClear(IntPtr handle);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int TessBaseAPIInit2(IntPtr handle, string datapath, string language, int oem);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int TessBaseAPISetVariable(IntPtr handle, string name, string value);

    [LibraryImport(Lib)] public static partial void TessBaseAPISetPageSegMode(IntPtr handle, int mode);
    [LibraryImport(Lib)] public static partial void TessBaseAPISetImage(IntPtr handle, byte* data, int width, int height, int bytesPerPixel, int bytesPerLine);
    [LibraryImport(Lib)] public static partial void TessBaseAPISetSourceResolution(IntPtr handle, int ppi);
    [LibraryImport(Lib)] public static partial int TessBaseAPIRecognize(IntPtr handle, IntPtr monitor);
    [LibraryImport(Lib)] public static partial IntPtr TessBaseAPIGetUTF8Text(IntPtr handle);
    [LibraryImport(Lib)] public static partial IntPtr TessBaseAPIGetIterator(IntPtr handle);

    [LibraryImport(Lib)] public static partial void TessResultIteratorDelete(IntPtr it);
    [LibraryImport(Lib)] public static partial IntPtr TessResultIteratorGetPageIterator(IntPtr it);
    [LibraryImport(Lib)] public static partial int TessResultIteratorNext(IntPtr it, int level);
    [LibraryImport(Lib)] public static partial IntPtr TessResultIteratorGetUTF8Text(IntPtr it, int level);
    [LibraryImport(Lib)] public static partial float TessResultIteratorConfidence(IntPtr it, int level);
    [LibraryImport(Lib)] public static partial int TessPageIteratorBoundingBox(IntPtr pageIterator, int level, int* left, int* top, int* right, int* bottom);

    /// <summary>Reads a C string and frees it with TessDeleteText.</summary>
    public static string TakeString(IntPtr p)
    {
        if (p == IntPtr.Zero) return "";
        var s = Marshal.PtrToStringUTF8(p) ?? "";
        TessDeleteText(p);
        return s;
    }
}
