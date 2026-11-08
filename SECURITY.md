# Security policy

PdfPecker.Ocr reads input that may come from untrusted sources: image files, PDF files and raw pixel buffers with
their sizes. A crash, hang, runaway memory use or unexpected exception type on crafted input is a security bug, as is
a searchable PDF whose text layer lets recognised text change the document's structure.

## Supported versions

Security fixes go into the latest release on [nuget.org](https://www.nuget.org/packages/PdfPecker.Ocr). Please upgrade
to it before reporting; older versions do not get patch releases.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub instead:
[Report a vulnerability](https://github.com/Moonwalker1307/PdfPecker.Ocr/security/advisories/new) (the **Security**
tab, then **Report a vulnerability**).

Include the PdfPecker.Ocr and PdfPecker versions, the .NET runtime, the processor, and the smallest input that shows
the problem, attached as a file. Inputs found with the fuzzer in `fuzz/PdfPecker.Ocr.Fuzz` are welcome as they are.

You can expect an acknowledgement within a few days. Once a fix is released, the advisory is published and you are
credited unless you prefer not to be.

## Out of scope

- Problems in image decoding, PDF parsing or PDF rendering that happen with PdfPecker alone, without OCR. Report those
  to [PdfPecker](https://github.com/Moonwalker1307/PdfPecker/security/advisories/new); the fix reaches this package
  through a PdfPecker release.
- Wrong or missing text. OCR is a best effort on what a scan shows, and misreading a crafted image is not a
  vulnerability unless it crashes, hangs or exhausts memory.
- Images or pages larger than a caller raised `OcrOptions.MaxPixels` to allow: that limit is the caller's choice.
