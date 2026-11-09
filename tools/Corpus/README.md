# Synthetic OCR corpus

Generates pages with known text, so that OCR engines can be scored without hand-labelled data. `PdfPecker.Ocr` is MIT licensed
and so is this generator and everything it writes; the texts are written for the corpus and are not quoted from anywhere. Only
the generator is checked in. The output goes to `bench/TesseractComparison/corpus/` (git-ignored).

```
dotnet run -c Release --project tools/Corpus                              # 63 pages, about 20 seconds
dotnet run -c Release --project tools/Corpus -- --tesseract-samples       # plus Tesseract's two sample images (downloaded)
dotnet run -c Release --project tools/Corpus -- --only invoice-en.150,letter-da --no-degraded --out /tmp/small
```

Options: `--out dir`, `--seed n` (default 20261108), `--dpis 300,150`, `--only a,b` (page ids containing any of the words),
`--no-degraded`, `--tesseract-samples`. The output is deterministic: the pages are drawn by the PdfPecker release in
`Directory.Build.props`, the noise comes from a SplitMix64 generator seeded from the page id, and nothing depends on the clock or the machine.

## How a page is made

1. `Pages.cs` writes the page as HTML with `Doc`. Every line of text is its own block with `white-space: nowrap`, and long
   paragraphs are wrapped by the generator, so the line breaks are the ones written down and the layout engine never re-wraps.
   The ground truth lines are produced by the same calls that write the HTML. A table row is one line (its cells joined by one
   space), and the columns of a two-column article are read left column first.
2. `HtmlToPdf` makes a one-page A4 PDF, which fails if the text spills onto a second page.
3. **A check of the ground truth**: `PdfText` reads the words back from the PDF, and the generator stops if they are not
   exactly the words of the ground truth lines (it also stops if a word runs past the right margin). A template or layout
   error cannot slip into the ground truth.
4. `PdfRenderer` renders the page at 300 and at 150 dpi (not scaled from one another). Pages marked colour are RGB, the others
   8-bit gray.
5. **Word boxes**: `PdfText` gives each word's box in points. Those boxes span the font's ascent and descent, so each is cut
   down to the dark pixels inside it (luma below 140) and converted to pixels: the ground truth hugs the ink, as an OCR engine's
   boxes do, and IoU measures the engine rather than the font metrics.

## The pages

| Id | Content |
| --- | --- |
| `invoice-en`, `-da` (gray), `-de`, `-fr` (colour) | header band, address, a table of five line items with amounts, totals, an IBAN, an email and a URL |
| `letter-en`, `-da`, `-de`, `-fr` | serif 12 pt business letter, right-aligned sender, bold salutation |
| `article-en`, `-de` | headline, byline and two columns of serif 10 pt text |
| `table-en`, `-da` | a bordered numeric table (quarterly sales; monthly consumption) |
| `smallprint-en` (sans), `-fr` (serif) | terms and conditions at 8 pt, a long line of about 110 characters |
| `headings-en`, `-da` | 36, 26 and 18 pt headings with 13 pt body text |
| `mono-en` | a service log in Liberation Mono 9.5 pt: timestamps, ids, email addresses |
| `identifiers-en` | phones, emails, URLs, dates in four formats, amounts in four number formats, VAT numbers |

The fonts are PdfPecker's bundled Liberation Sans, Serif and Mono. English, Danish (æøå), German (äöüß) and French (accents,
`«` is not used; apostrophes are ASCII).

Each id is written as `<page>.<dpi>`, for example `letter-da.300`. Three pages (`invoice-en`, `letter-de`, `article-en`) also get
degraded variants at 300 dpi, `<page>.300.<variant>`, made on raw gray pixels in `Degrade.cs` with the seeded generator:

| Variant | Degradation |
| --- | --- |
| `noise` | additive Gaussian noise, sigma 28 gray levels |
| `blur` | Gaussian blur, sigma 1.6 px |
| `skew1`, `skew2`, `skew3` | rotation by +1, -2 and +3 degrees (bilinear, white fill) |
| `jpeg30` | what a baseline JPEG at quality 30 does to a gray page (8x8 DCT, IJG luminance table scaled for q30, rounding, inverse DCT). The entropy coding is lossless, so these are the pixels a decoder returns; it avoids a JPEG decoder (PdfPecker's is internal) and resampling |
| `lowcontrast` | gray range squeezed to 120..175 plus noise of sigma 3, like a faint photocopy |
| `rot180` | the page turned upside down; the ground truth stays the text in its normal reading order |
| `scan` | 1.5 degree skew, blur 0.8, noise 14 and JPEG quality 40 together |

Ground truth boxes of the skewed pages are the axis-aligned boxes of the rotated ink boxes, so they are looser than the ink
(IoU against a perfect engine is below 1 for those pages).

## Files

For every page `<id>`: `<id>.png` (8-bit gray or RGB), `<id>.txt` (ground truth lines, UTF-8), `<id>.boxes.tsv`
(`x y width height text`, pixels, origin top-left), and one `manifest.tsv` with the columns `id file width height channels dpi
lang category font degradation rotation text boxes`. Other workstreams can use the PNGs directly (`ImageOcr.Recognize(byte[])`)
and compare with the `.txt` files.

## Tesseract's sample images

`--tesseract-samples` fetches `testing/phototest.tif` and `testing/eurotext.tif` with their `.txt` ground truth from
[tesseract-ocr/test](https://github.com/tesseract-ocr/test) (**Apache-2.0**, the repository's licence), checks their SHA-256, decodes
the TIFFs through PdfPecker and adds them as `tesseract-phototest` and `tesseract-eurotext` (category `tesseract-sample`, no boxes,
no resolution; `eurotext` is multilingual, so Tesseract runs it with `eng+deu+fra`). They are never checked in or redistributed;
the licence matters only if someone copies them out. `eurotext.txt` keeps the `„ ”` and `« »` quotes, which is part of the test.
