using PdfPecker.Ocr;

// Usage: PdfPecker.Ocr.Sample <image file>
// Prints the text read from the image. Exit codes follow the PdfPecker tools: 0 success, 1 the work failed, 2 a wrong
// command line.
//
// TODO(lead): the recognition API is not in the package yet, so for now the sample only prints the package version and
// checks that the file exists. Once ImageOcr is in, replace the marked block with:
//
//     OcrPage page = ImageOcr.Recognize(File.ReadAllBytes(args[0]));
//     Console.WriteLine(page.Text);
//
// and let the native-aot job in .github/workflows/nuget.yml check the printed text of a fixture image.

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: PdfPecker.Ocr.Sample <image file>");
    return 2;
}

if (!File.Exists(args[0]))
{
    Console.Error.WriteLine($"File not found: {args[0]}");
    return 1;
}

// Marked block: replace with the recognition call above.
Console.WriteLine($"PdfPecker.Ocr {OcrInfo.Version}");
return 0;
