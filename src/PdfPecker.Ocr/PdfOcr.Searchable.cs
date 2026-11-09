using PdfPecker.Ocr.Input;
using PdfPecker.Ocr.Output;
using PdfPecker.Ocr.Pdf;

namespace PdfPecker.Ocr;

/// <summary>Reads the text of scanned PDFs and makes them searchable. All methods are static and thread-safe.</summary>
public static partial class PdfOcr
{
    /// <summary>
    /// Makes a scanned PDF searchable: reads the pages that have no text (see <see cref="OcrOptions.PageMode"/>) and
    /// puts what it reads on them as invisible text, which viewers select, copy and search, and text extraction reads.
    /// Nothing on the pages changes how they look.
    /// </summary>
    /// <remarks>
    /// The text is drawn beneath each page's content, word by word at the place OCR found it, following the page's
    /// rotation and crop box. Everything else in the document is kept: annotations, form fields, bookmarks, links,
    /// attachments, metadata and the structure tree, in which the text layer is marked as an artifact. The font of the
    /// layer is embedded with a ToUnicode map, so a PDF/A file stays valid. The result is written in full, not as an
    /// incremental update, and it is not encrypted, also when the input was: encrypt it again with
    /// <see cref="PdfProtector"/> if needed. When no page needs reading, the result is a copy of the input.
    /// </remarks>
    /// <param name="pdf">The PDF.</param>
    /// <param name="options">Resolution, pages, password and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <returns>The searchable PDF.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="pdf"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of its range, or <see cref="OcrOptions.Pages"/> names a page the PDF does not have.</exception>
    /// <exception cref="PdfPasswordException">The PDF is encrypted and <see cref="OcrOptions.Password"/> does not open it.</exception>
    /// <exception cref="InvalidDataException">The PDF is not readable, or a page cannot be rendered.</exception>
    /// <exception cref="InvalidOperationException">The PDF is digitally signed: adding text would break the signature.</exception>
    /// <exception cref="NotSupportedException">The PDF uses an encryption handler other than the standard one.</exception>
    /// <exception cref="PdfLimitException">
    /// The PDF is larger than PdfPecker can rewrite, or a page needs more than 2 GB of pixels at the resolution it is read at.
    /// </exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled.</exception>
    public static byte[] MakeSearchable(byte[] pdf, OcrOptions? options = null, CancellationToken cancellationToken = default)
        => SearchablePdf.MakeSearchable(pdf, options, PageRecognizer.Default, cancellationToken);

    /// <summary>Makes a scanned PDF read from a stream searchable and writes the result to <paramref name="output"/>.</summary>
    /// <remarks>The input is read from its current position to its end; neither stream is disposed.</remarks>
    /// <param name="pdf">The PDF.</param>
    /// <param name="output">Where the searchable PDF is written.</param>
    /// <param name="options">Resolution, pages, password and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <exception cref="ArgumentNullException"><paramref name="pdf"/> or <paramref name="output"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="pdf"/> is not readable, or <paramref name="output"/> is not writable.</exception>
    /// <exception cref="IOException">Reading or writing a stream fails.</exception>
    /// <inheritdoc cref="MakeSearchable(byte[], OcrOptions?, CancellationToken)"/>
    public static void MakeSearchable(Stream pdf, Stream output, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pdf);
        ArgumentNullException.ThrowIfNull(output);
        if (!pdf.CanRead) throw new ArgumentException("The stream is not readable.", nameof(pdf));
        if (!output.CanWrite) throw new ArgumentException("The output stream is not writable.", nameof(output));
        var result = MakeSearchable(ReadAll(pdf, cancellationToken), options, cancellationToken);
        output.Write(result);
        output.Flush();
    }

    /// <summary>
    /// Makes a scanned PDF file searchable and writes the result to <paramref name="outputPath"/>, which may be the same
    /// path. The output file appears only once it is complete, so a failure leaves any earlier file at that path as it was.
    /// A file it replaces keeps its Unix permissions, and a symbolic link at the output path is followed.
    /// </summary>
    /// <param name="inputPath">The PDF file.</param>
    /// <param name="outputPath">The searchable PDF file to write.</param>
    /// <param name="options">Resolution, pages, password and limits; null for the defaults.</param>
    /// <param name="cancellationToken">Stops the work.</param>
    /// <exception cref="ArgumentNullException">A path is null.</exception>
    /// <exception cref="IOException">The input file cannot be read or the output file cannot be written.</exception>
    /// <exception cref="UnauthorizedAccessException">The process may not read the input file or write the output file.</exception>
    /// <inheritdoc cref="MakeSearchable(byte[], OcrOptions?, CancellationToken)"/>
    public static void MakeSearchableFile(string inputPath, string outputPath, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputPath);
        ArgumentNullException.ThrowIfNull(outputPath);
        var result = MakeSearchable(File.ReadAllBytes(inputPath), options, cancellationToken);
        OutputFile.WriteAllBytes(outputPath, result);
    }

    /// <summary>The stream from its position to its end, read in chunks with the token checked between them.</summary>
    internal static byte[] ReadAll(Stream stream, CancellationToken cancellationToken)
    {
        if (stream is MemoryStream ms && ms.TryGetBuffer(out var buffer))
        {
            // A position past the end is allowed, and the stream reads nothing from there.
            int start = (int)Math.Min(ms.Position, ms.Length);
            var data = buffer.AsSpan(start).ToArray();
            if (start < ms.Length) ms.Position = ms.Length;
            return data;
        }
        var copy = new MemoryStream();
        var chunk = new byte[81920];
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int read = stream.Read(chunk, 0, chunk.Length);
            if (read == 0) break;
            copy.Write(chunk, 0, read);
        }
        return copy.ToArray();
    }
}
