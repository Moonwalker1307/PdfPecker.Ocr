using PdfPecker;
using PdfPecker.Ocr;

// Usage: PdfPecker.Ocr.Sample [options] <image or PDF file>
//
// Prints the text OCR reads in an image (PNG, JPEG, GIF, WebP, BMP, TIFF) or a scanned PDF. Exit codes follow the
// PdfPecker tools: 0 success, 1 the work failed, 2 a wrong command line.
//
//   --hocr               print hOCR instead of plain text
//   --searchable <file>  write a searchable PDF (the PDF made searchable, or the image as a PDF page) and print the
//                        text of its text layer as PdfPecker extracts it
//   --dpi <n>            the resolution PDF pages are read at, and that of images that record none (default 300)
//   --threads <n>        the most threads a page uses (default: the number of processors)
//
// sample.png next to this file is a page PdfPecker drew from HTML and rendered at 300 dpi; CI publishes this program
// with Native AOT and checks the text it reads there.

const string Usage = "Usage: PdfPecker.Ocr.Sample [--hocr] [--searchable <out.pdf>] [--dpi <n>] [--threads <n>] <image or PDF file>";

string? input = null, searchable = null;
bool hocr = false;
var options = new OcrOptions();
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--hocr":
            hocr = true;
            break;
        case "--searchable" when i + 1 < args.Length:
            searchable = args[++i];
            break;
        case "--dpi" when i + 1 < args.Length && float.TryParse(args[i + 1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float dpi) && dpi is >= 10 and <= 2400:
            options.Dpi = dpi;
            i++;
            break;
        case "--threads" when i + 1 < args.Length && int.TryParse(args[i + 1], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int threads) && threads >= 1:
            options.MaxDegreeOfParallelism = threads;
            i++;
            break;
        case "-h" or "--help":
            Console.WriteLine(Usage);
            return 0;
        default:
            if (args[i].StartsWith('-') || input != null)
            {
                Console.Error.WriteLine(Usage);
                return 2;
            }

            input = args[i];
            break;
    }
}

if (input == null || (hocr && searchable != null))
{
    Console.Error.WriteLine(input == null ? Usage : "--hocr and --searchable cannot be used together.");
    return 2;
}

try
{
    byte[] data = File.ReadAllBytes(input);
    bool pdf = data.AsSpan(0, Math.Min(data.Length, 1024)).IndexOf("%PDF-"u8) >= 0;
    if (searchable != null)
    {
        byte[] result = pdf ? PdfOcr.MakeSearchable(data, options) : ImageOcr.ToSearchablePdf([data], options);
        File.WriteAllBytes(searchable, result);
        Console.WriteLine(PdfText.Extract(result));
    }
    else if (pdf)
    {
        OcrDocument document = PdfOcr.Recognize(data, options);
        Console.WriteLine(hocr ? document.ToHocr() : document.Text);
    }
    else
    {
        OcrPage page = ImageOcr.Recognize(data, options);
        Console.WriteLine(hocr ? page.ToHocr() : page.Text);
    }

    return 0;
}
catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException or PdfPeckerException or InvalidOperationException or NotSupportedException)
{
    Console.Error.WriteLine($"{input}: {e.Message}");
    return 1;
}
