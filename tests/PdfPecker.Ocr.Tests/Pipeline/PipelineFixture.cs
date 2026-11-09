using System.IO.Compression;
using System.Text.Json;

namespace PdfPecker.Ocr.Tests.Pipeline;

/// <summary>
/// One test case recorded from rapidocr 3.10 by tools/PipelineReference/capture.py: the rendered image, the detection
/// probability map and rapidocr's boxes, the line crops, and the classifier and recogniser outputs with rapidocr's
/// decoded texts. JSON is read with <see cref="JsonDocument"/>, which needs no reflection.
/// </summary>
internal sealed class PipelineFixture : IDisposable
{
    private static readonly string Folder = Path.Combine(AppContext.BaseDirectory, "Fixtures", "pipeline");
    private static readonly Lazy<string[]> CharacterList = new(LoadCharacters);

    private PipelineFixture(string name, JsonDocument json)
    {
        Name = name;
        Json = json;
        JsonElement root = json.RootElement;
        Width = root.GetProperty("width").GetInt32();
        Height = root.GetProperty("height").GetInt32();
        Channels = root.GetProperty("channels").GetInt32();
        JsonElement pre = root.GetProperty("preprocess");
        DetImageWidth = pre.GetProperty("det_image_width").GetInt32();
        DetImageHeight = pre.GetProperty("det_image_height").GetInt32();
        JsonElement shape = pre.GetProperty("det_input_shape");
        DetInputHeight = shape[2].GetInt32();
        DetInputWidth = shape[3].GetInt32();
    }

    public static string[] Cases { get; } = ["invoice", "small_text", "rotated", "columns", "table", "upside_down", "nordic", "gap"];

    public string Name { get; }

    public JsonDocument Json { get; }

    public JsonElement Root => Json.RootElement;

    public int Width { get; }

    public int Height { get; }

    public int Channels { get; }

    public int DetImageWidth { get; }

    public int DetImageHeight { get; }

    public int DetInputWidth { get; }

    public int DetInputHeight { get; }

    /// <summary>rapidocr's character list with the CTC blank first and the space it appends last.</summary>
    public static string[] Characters => CharacterList.Value;

    public static PipelineFixture Load(string name) => new(name, JsonDocument.Parse(ReadGzip($"{name}.json.gz")));

    public static byte[] ReadGzip(string file)
    {
        using FileStream stream = File.OpenRead(Path.Combine(Folder, file));
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var buffer = new MemoryStream();
        gzip.CopyTo(buffer);
        return buffer.ToArray();
    }

    public byte[] Image() => ReadGzip($"{Name}.image.gz");

    /// <summary>The image rapidocr's detector saw, which is the image itself unless rapidocr padded or resized it.</summary>
    public byte[] DetImage() => Root.GetProperty("det_image_is_image").GetBoolean() ? Image() : ReadGzip($"{Name}.detimage.gz");

    /// <summary>The detection input as pixels: one plane for grey images, else the B, G and R planes.</summary>
    public byte[] DetInput() => ReadGzip($"{Name}.detinput.gz");

    /// <summary>The probability map, dequantised as capture.py dequantised it for rapidocr's own post-processing.</summary>
    public float[] ProbabilityMap()
    {
        byte[] q = ReadGzip($"{Name}.prob.gz");
        var map = new float[q.Length];
        for (int i = 0; i < q.Length; i++)
        {
            // numpy: q.astype(float32) / 255.0 stays in float32.
            map[i] = q[i] / 255f;
        }

        return map;
    }

    public byte[] RecPixels() => ReadGzip($"{Name}.rec.gz");

    /// <summary>Reads a list of 4-point boxes from the JSON.</summary>
    public List<float[]> Boxes(string property)
    {
        var boxes = new List<float[]>();
        foreach (JsonElement box in Root.GetProperty(property).EnumerateArray())
        {
            var b = new float[8];
            int i = 0;
            foreach (JsonElement point in box.EnumerateArray())
            {
                b[i++] = point[0].GetSingle();
                b[i++] = point[1].GetSingle();
            }

            boxes.Add(b);
        }

        return boxes;
    }

    public void Dispose() => Json.Dispose();

    private static string[] LoadCharacters()
    {
        string text = System.Text.Encoding.UTF8.GetString(ReadGzip("characters.txt.gz"));
        string[] lines = text.Split('\n');
        var list = new string[lines.Length + 2];
        list[0] = "blank";
        lines.CopyTo(list, 1);
        list[^1] = " ";
        return list;
    }
}
