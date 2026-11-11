# HeadTuner

Writes `src/PdfPecker.Ocr/Models/rec-head.bin`: retrained output weights for 18 of the recogniser's classes, which the
recogniser takes in place of the converted ones when it loads (`HeadPatch` in `src/PdfPecker.Ocr/Models`). The file is
generated data; change this tool and run it again, never edit the file.

```
dotnet run -c Release --project tools/HeadTuner -- [--pages 800] [--out path]
```

A run takes about seven minutes on four cores.

## Why

PP-OCRv6 saw few capital Danish and Norwegian letters in training. It reads Æ as the two letters A and E, emitted on
neighbouring time steps inside the one glyph, Ø as O or a zero (in fonts with a slashed zero), and Å as A, the more so
in capitals, monospaced fonts and poor scans. The crops and the decoding are not the cause: the network gives Å about
0.3% where it reads A on a degraded line, ring and all. Rewriting the text after decoding ("AE" to Æ, a zero inside a
word to Ø) would guess at what the page says; retraining the output layer lets the network read it.

## What it does

1. Generates lines of text with a seeded generator (`TrainingText.cs`): Danish words with and without æ, ø and å,
   Danish place and person names, English (with capital AE, which must stay two letters), German, numbers, codes with
   zeros next to letters, and punctuation; a fifth of the lines in capitals.
2. Renders 800 pages of 12 lines with PdfPecker in its bundled Liberation Serif, Sans and Mono, at 8 to 20 points,
   regular, bold and italic, at 200 and 300 dpi, four in ten clean and the rest blurred, noisy and low in contrast
   (`Pages.cs`). Only the bundled fonts are used, so every machine renders the same pages.
3. Reads each page with the pipeline and the converted recogniser, and keeps for each line, per time step, the
   softmax output and the 120 features that the last linear layer reads (`Samples.cs`). A line whose text is not
   within a fifth of a known line is skipped.
4. Aligns the known text with the output by CTC forced alignment, which says what each time step should be: a blank,
   or a character at the step where the network emits it.
5. Refits the last layer's weights and biases of the tuned classes (`Fit.cs`): the cross-entropy of the full softmax
   over 18,710 classes, with every other class's logit fixed, plus an L2 pull towards the original weights; Adam, 8
   passes, in double precision and a fixed order. 0, O and o take part in the softmax, so the other letters learn not
   to take their places, but keep their own weights: a zero or an O next to letters reads as before. Steps where no
   tuned class is in play are kept at one in 25, weighted to match.
6. Writes the weights of the 18 classes that changed, with each class's character so the loader can check that the
   file belongs to the character list.

Pages are read in parallel, each in an inference context of its own, and their samples joined in page order, so the
number of threads does not change the result. The last bits of the features can differ between processors whose
SIMD paths differ, so a run elsewhere writes weights that differ in the last bits, not in what they read.

## Checking a new file

The golden tests show what changed: `nordic` and `degraded-nordic` are pages of Danish capitals, monospaced text,
English capitals with AE, codes with zeros and punctuation; `letter-da` is a Danish letter. Before committing a new
file, score the corpus of `bench/TesseractComparison` with the old and the new file: the Danish pages must not get
worse, and the others must not change.
