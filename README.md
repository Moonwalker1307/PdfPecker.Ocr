# PdfPecker.Ocr

Text recognition (OCR) for .NET with no native dependencies: reads printed text from images and scanned PDFs, and
makes scanned PDFs searchable by adding an invisible text layer.

PdfPecker.Ocr runs PaddleOCR's PP-OCRv6 small models on a neural network runtime of its own, written in C#. It does not
ship native binaries, call ONNX Runtime or Tesseract, or need a GPU or a network connection. Its only dependency is the
[PdfPecker](https://www.nuget.org/packages/PdfPecker) package, which it uses to decode images, render PDF pages and
write the text layer.

> PdfPecker.Ocr is being built towards its first release, 0.1.0. The API below is the one planned for that release;
> until it is on nuget.org, the package and this page may change.

```csharp
using PdfPecker.Ocr;

OcrPage page = ImageOcr.Recognize(File.ReadAllBytes("receipt.png"));
Console.WriteLine(page.Text);
```

## Install

```
dotnet add package PdfPecker.Ocr
```

The package targets `net8.0`, `net9.0` and `net10.0`, so it works on .NET 8 and later, on Windows, Linux and macOS,
on x64 and Arm64. It is trimmable and works in Native AOT applications.

Native AOT compiles for a baseline processor unless told otherwise, and the network kernels then run without AVX2, FMA
and AVX-512, which on x64 makes a page two and a half to four times slower than under the JIT (which picks them at run
time). Name the processors you deploy to in your application's project, for example
`<IlcInstructionSet>x86-64-v3</IlcInstructionSet>` for AVX2 and FMA, or `native` for the machine that builds it.

## Quick start

```csharp
using PdfPecker.Ocr;

// An image file: PNG, JPEG, TIFF, WebP, BMP or GIF
OcrPage page = ImageOcr.Recognize(File.ReadAllBytes("scan.jpg"));
Console.WriteLine(page.Text);

// Lines and words with their positions in page pixels (origin top left) and a confidence from 0 to 1
foreach (OcrLine line in page.Lines)
{
    Console.WriteLine($"{line.Box.X:0},{line.Box.Y:0} {line.Confidence:P0} {line.Text}");
    foreach (OcrWord word in line.Words)
        Console.WriteLine($"  {word.Text} at {word.Quad.TopLeft}");
}

// Pixels you already have, such as a camera frame or a bitmap from another library
OcrPage frame = ImageOcr.RecognizeRgba(rgba, width, height);
OcrPage gray = ImageOcr.RecognizeGray(luminance, width, height);

// A scanned PDF: the text of every page, and hOCR for tools that read it
OcrDocument document = PdfOcr.Recognize(File.ReadAllBytes("scanned.pdf"));
Console.WriteLine(document.Text);
File.WriteAllText("scanned.hocr", document.ToHocr());

// Make a scanned PDF searchable: the pages look the same, and their text can be selected, searched and copied.
// Pages that already have text are left as they are unless PageMode is Force.
byte[] searchable = PdfOcr.MakeSearchable(File.ReadAllBytes("scanned.pdf"));

// Every byte array method also takes a stream, and has a file version, as in PdfPecker
OcrDocument fromFile = PdfOcr.RecognizeFile("scanned.pdf");
PdfOcr.MakeSearchableFile("scanned.pdf", "searchable.pdf");
OcrPage photo = ImageOcr.RecognizeFile("photo.png");

// Images straight to one searchable PDF, a page per image
byte[] pdf = ImageOcr.ToSearchablePdf(Directory.GetFiles("scans", "*.png").Select(File.ReadAllBytes));

// Options: render PDF pages at 200 dpi, drop lines below 60% confidence, use two threads
var options = new OcrOptions { Dpi = 200, MinConfidence = 0.6f, MaxDegreeOfParallelism = 2 };
OcrDocument report = PdfOcr.Recognize(File.ReadAllBytes("report.pdf"), options);
```

The methods are static and thread-safe, so documents can be recognised in parallel. The models are loaded once, on
first use, and shared. Pages are read one after another, each with up to `MaxDegreeOfParallelism` threads (at most the
number of processors) working inside the networks; the threads and buffers are kept for the next call and released after
30 seconds without one, so a stream of pages allocates little after the first.

## How it works

1. **Input.** Images are decoded and PDF pages rendered (at `OcrOptions.Dpi`, 300 by default) by PdfPecker.
2. **Detection.** The PP-OCRv6 small detection model marks where text is; the probability map it returns is turned
   into one quadrilateral per text line, so slanted lines keep their angle.
3. **Orientation.** Each line is cut out and straightened, and a small classifier turns lines that are upside down
   (`OcrOptions.DetectRotation`).
4. **Recognition.** The PP-OCRv6 small recognition model reads each line, and its output is decoded into characters
   with a confidence and a position along the line, from which the words and their boxes come.
5. **Layout.** Lines are put into reading order, top to bottom and left to right, column by column.
6. **Searchable PDF.** The text is written as invisible text over each word, scaled to the word's width, and stamped
   onto the original pages, so the scan itself is not changed.

The runtime that executes the models is the package's own: a graph executor with SIMD kernels (AVX2, FMA, AVX-512
where present, and Arm64 Advanced SIMD) for convolutions, matrix products and the other operations these models use,
with buffers planned when the model loads, so a steady stream of pages allocates little. Results are the same whatever
`MaxDegreeOfParallelism` is. The model weights are embedded in the assembly as 24-bit floats (32-bit floats rounded to
15 mantissa bits, which keeps the outputs within 1e-3 of ONNX Runtime's), generated from the published ONNX files by
`tools/ModelBuilder`.

## Languages

The recognition model reads Latin script, including the accented letters of Western, Central and Northern European
languages (`æ ø å ä ö ü ß é ç ñ ł ő č ž` and others), Greek, Chinese (simplified and traditional) and Japanese
(kana and kanji), as well as digits, punctuation and common symbols. Its character list has no Cyrillic, Korean,
Arabic, Hebrew, Devanagari or Thai letters, so text in those scripts is not recognised.

## Limitations

- **Printed text only.** The models are trained on printed and screen text. Handwriting is not supported.
- **Accuracy depends on the scan.** Blurred, very low resolution (below about 150 dpi for body text), heavily
  compressed or noisy scans read worse, as do decorative fonts, text on busy backgrounds and very small print. The
  confidence of each line and word says how sure the model was.
- **Rotation.** Lines that are slanted, on their side or upside down are read, so pages turned by 90 or 180 degrees are
  read too; each line's quad says which way its text runs. Vertical writing, such as Japanese in columns, is not
  supported.
- **Layout.** Reading order is worked out from the positions of lines. Tables come out as lines of text, not as cells,
  and complex magazine layouts can come out in a different order than a person would read them. On a scan skewed by
  two degrees or more, a table can come out column by column, because the layout step does not yet correct small
  angles; the words themselves are read correctly.
- **No language model.** Each line is read on its own, without a dictionary, so a misread character is not corrected
  from the words around it.
- **Size.** Pages are limited to `OcrOptions.MaxPixels` (100 million pixels by default), so that a hostile file cannot
  exhaust memory: a PDF page or image file with more pixels is read at a lower resolution, and a raw pixel buffer that
  large is refused with a `PdfLimitException`.

## Performance

Measured with `bench/TesseractComparison` on its synthetic corpus of 65 pages (A4 invoices, letters, two-column
articles, tables, small print, a monospaced log and identifiers in English, Danish, German and French, at 300 and
150 dpi, 27 degraded variants, and Tesseract's two sample images), against Tesseract 5.5.1 called in the same process
and RapidOCR running the same models on ONNX Runtime. The machine was a shared 4-vCPU cloud VM (Xeon at 2.1 GHz with
AVX-512) on which other jobs were running, measured on 2026-11-08; times are the median over pages of each page's median
after a warm-up, with no image decoding.

| | PdfPecker.Ocr | RapidOCR + ONNX Runtime | Tesseract 5.5.1, `tessdata_fast` | Tesseract 5.5.1, `tessdata_best` (English pages) |
| --- | ---: | ---: | ---: | ---: |
| Character error rate, in reading order | 1.7% | 23.3% | 10.4% | 15.2% |
| Character error rate, ignoring reading order | 0.4% | 6.4% | 3.3% | 2.2% |
| Word bag F1 | 0.996 | 0.960 | 0.971 | 0.973 |
| Time per page, 1 thread | 2.35 s | 5.20 s | 0.58 s | 1.35 s |
| Time per page, 4 threads | 0.73 s | 1.91 s | 0.57 s | 0.76 s |
| CPU time per page | 2.4 s | 5.0 s | 0.6 s | 1.6 s |
| Peak memory, one A4 page at 300 dpi in a fresh process | 431 MB | 548 MB | 67 MB | 80 MB |

PdfPecker.Ocr reads the most text correctly, and it reads every page faster than ONNX Runtime does with the same models,
but Tesseract needs about a quarter of its CPU time and a sixth of its memory. A page of long lines takes more time and
memory (7.3 s and 906 MB at one thread for a monospaced log of 36 lines of up to 80 characters). Most of the reading order
penalty of the other engines is their order of table columns (Tesseract) and of article columns (RapidOCR); the method,
the results per page group and the causes of each engine's errors are in `bench/TesseractComparison/README.md`. The
corpus is rendered, not scanned, which flatters every engine. Cloud machines differ too much in processor to compare a
run with this table; to compare two versions, run `bench/compare-versions.sh` on one machine.

**Memory.** Reading an A4 page at 300 dpi takes about 220 to 240 MB beyond the .NET runtime's own: the models (30 MB,
kept once loaded) and the detector's working memory (about 100 MB of native memory, released with the threads after 30
seconds without a call). Recognising pixels allocates under 1 MB per page once warm, so the garbage collector's settings
make no difference there. Rendering PDF pages does allocate: about 117 MB of short-lived pixel buffers per scanned A4
page at 300 dpi, in PdfPecker's renderer. A process that reads PDFs all day keeps less memory resident with
`System.GC.ConserveMemory` set to 5 (in `runtimeconfig.json`, or `DOTNET_GCConserveMemory=5`): on the benchmark machine,
making a six-page scanned PDF searchable three times at four threads peaked at 698 MB instead of 1,086 MB with the
default settings, in the same time. Server GC peaked at 848 MB and was about 10% slower there, so the workstation
collector, the default for console apps, suits this work.

## Licence

PdfPecker.Ocr's code is licensed under the [MIT license](LICENSE). The model weights embedded in the package are
derived from PaddleOCR's models, converted to ONNX by the RapidOCR authors, and are licensed under the Apache License,
Version 2.0; [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) has the attribution, the notice of the changes made to
them and the licence text. Both licences allow commercial use. The package's licence expression is
`MIT AND Apache-2.0`.

## Contributing

Bug reports with a sample image or PDF are the most useful thing; see [CONTRIBUTING.md](CONTRIBUTING.md). Security
problems go through the [security policy](SECURITY.md).
