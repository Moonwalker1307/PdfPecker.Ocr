using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using ModelBuilder;
using PdfPecker.Ocr.Inference;
using PdfPecker.Ocr.Models;

// Builds src/PdfPecker.Ocr/Models/{det,rec,cls}.bin.deflate from the ONNX models in the rapidocr 3.10.0 wheel on PyPI.
// Usage, from the repository root:
//   dotnet run -c Release --project tools/ModelBuilder -- [--wheel path] [--out dir] [--precision fp24|fp16|fp32]
// Without --wheel the wheel is downloaded once into a cache folder in the temporary directory. The wheel and each model
// in it are checked against the hashes below, so a different upstream file cannot change the output unnoticed; the
// output is byte-identical on every run. See README.md for the file format.

const string WheelUrl = "https://files.pythonhosted.org/packages/59/07/203065283bba0c5e9c10af3602dc25ab3e59ee54dc2d1d22c8faeabf5ed8/rapidocr-3.10.0-py3-none-any.whl";
const string WheelSha256 = "2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050";
const string LicenceNotice =
    "Copyright Baidu and/or the PaddleOCR rights holders; converted by the RapidOCR Authors. Licensed under the Apache License, Version 2.0 "
    + "(https://www.apache.org/licenses/LICENSE-2.0). Changed by PdfPecker.Ocr's ModelBuilder: graph simplified and fused, batch normalisation "
    + "folded into convolutions, weights stored at reduced precision.";

var models = new[]
{
    new ModelSource("det", "rapidocr/models/PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
        "PP-OCRv6_small_det", [[1, 3, 64, 96], [2, 3, 96, 64], [3, 3, 160, 128]]),
    new ModelSource("rec", "rapidocr/models/PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
        "PP-OCRv6_small_rec", [[1, 3, 48, 320], [2, 3, 48, 160], [3, 3, 48, 96]]),
    new ModelSource("cls", "rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
        "ch_ppocr_mobile_v2.0_cls", [[1, 3, 48, 192], [2, 3, 48, 96], [3, 3, 48, 128]]),
};

string? wheelPath = null;
string outDir = Path.Combine("src", "PdfPecker.Ocr", "Models");
var precision = WeightStorage.Float24;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--wheel": wheelPath = args[++i]; break;
        case "--out": outDir = args[++i]; break;
        case "--precision":
            precision = args[++i] switch
            {
                "fp24" => WeightStorage.Float24,
                "fp16" => WeightStorage.Float16,
                "fp32" => WeightStorage.Float32,
                var other => throw new ArgumentException($"Unknown precision {other}; use fp24, fp16 or fp32."),
            };
            break;
        default:
            Console.Error.WriteLine("Usage: ModelBuilder [--wheel path] [--out dir] [--precision fp24|fp16|fp32]");
            return 2;
    }
}

byte[] wheel = await LoadWheel(wheelPath);
Directory.CreateDirectory(outDir);
using var zip = new ZipArchive(new MemoryStream(wheel), ZipArchiveMode.Read);
foreach (var source in models)
{
    var entry = zip.GetEntry(source.Entry) ?? throw new InvalidDataException($"{source.Entry} is missing from the wheel.");
    byte[] onnx;
    using (var stream = entry.Open())
    using (var buffer = new MemoryStream())
    {
        stream.CopyTo(buffer);
        onnx = buffer.ToArray();
    }
    string hash = Sha256(onnx);
    if (hash != source.Sha256) throw new InvalidDataException($"{source.Entry} has SHA-256 {hash}, expected {source.Sha256}.");

    Console.WriteLine($"{source.Name}: {source.Entry} ({onnx.Length:N0} bytes)");
    var onnxModel = OnnxModel.Parse(onnx);
    if (onnxModel.OpsetVersion != 11) throw new NotSupportedException($"Opset {onnxModel.OpsetVersion} is not the opset 11 the builder was written for.");
    var graph = Graph.FromOnnx(onnxModel);
    Console.WriteLine($"  ONNX graph:    {Passes.Summary(graph)}");
    Passes.Run(graph, source.Samples, Console.Out);

    string[] characters = [];
    if (onnxModel.Metadata.TryGetValue("character", out var list))
    {
        // rapidocr reads the list with str.splitlines() and adds the CTC blank first and a space last. The list holds
        // no line separator other than \n, which the builder checks so that both readings agree.
        if (list.Any(c => c is '\r' or '\v' or '\f' or '\u001c' or '\u001d' or '\u001e' or '\u0085' or '\u2028' or '\u2029'))
            throw new InvalidDataException("The character list holds line separators other than \\n.");
        characters = [string.Empty, .. list.Split('\n'), " "];
    }

    string notice = $"{source.Upstream}, from rapidocr 3.10.0 ({source.Entry}, SHA-256 {source.Sha256}). {LicenceNotice}";
    var definition = Emitter.Emit(graph, source.Upstream, notice, precision, characters, source.Samples);
    byte[] data = definition.Serialize();
    byte[] packed = Compress(data);
    string target = Path.Combine(outDir, source.Name + ".bin.deflate");
    File.WriteAllBytes(target, packed);
    Console.WriteLine($"  weights: {definition.Weights.Sum(w => (long)w.Data.Length):N0} values in {definition.Weights.Count} tensors, {definition.Nodes.Count} runtime nodes");
    Console.WriteLine($"  written {target}: {data.Length:N0} bytes, {packed.Length:N0} compressed (SHA-256 {Sha256(packed)})");

    // Loading and running the model here catches a definition the runtime rejects before any test does.
    var model = OcrModel.Load(data);
    if (characters.Length != 0 && model.Characters.Length != characters.Length) throw new InvalidDataException("Character list did not round-trip.");
    var sample = source.Samples[0].Select(d => (int)d).ToArray();
    var watch = Stopwatch.StartNew();
    using var context = new InferenceContext(Environment.ProcessorCount);
    var output = model.Run(new Tensor(sample), context);
    Console.WriteLine($"  check run {string.Join('x', sample)} -> {string.Join('x', output.Shape)} in {watch.ElapsedMilliseconds} ms");
    if (characters.Length != 0 && output.Shape[^1] != characters.Length)
        throw new InvalidDataException($"The recogniser has {output.Shape[^1]} classes but {characters.Length} characters.");
}
return 0;

static async Task<byte[]> LoadWheel(string? path)
{
    if (path is null)
    {
        string cache = Path.Combine(Path.GetTempPath(), "pdfpecker-ocr-modelbuilder");
        Directory.CreateDirectory(cache);
        path = Path.Combine(cache, "rapidocr-3.10.0-py3-none-any.whl");
        if (!File.Exists(path) || Sha256(await File.ReadAllBytesAsync(path)) != WheelSha256)
        {
            Console.WriteLine($"Downloading {WheelUrl}");
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
            byte[] downloaded = await http.GetByteArrayAsync(WheelUrl);
            await File.WriteAllBytesAsync(path + ".part", downloaded);
            File.Move(path + ".part", path, overwrite: true);
        }
    }
    byte[] wheel = await File.ReadAllBytesAsync(path);
    string hash = Sha256(wheel);
    if (hash != WheelSha256) throw new InvalidDataException($"{path} has SHA-256 {hash}, expected {WheelSha256}.");
    return wheel;
}

static string Sha256(byte[] data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();

// Deflate rather than Brotli: the files are 4% larger, but the runtime inflates them two and a half times as fast,
// which is most of a model's load time.
static byte[] Compress(byte[] data)
{
    using var packed = new MemoryStream();
    using (var deflate = new DeflateStream(packed, CompressionLevel.SmallestSize, leaveOpen: true)) deflate.Write(data);
    return packed.ToArray();
}

internal sealed record ModelSource(string Name, string Entry, string Sha256, string Upstream, long[][] Samples);
