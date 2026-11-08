using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// The findings folder of one target: a minimised input and a .txt with the reason and stack trace per distinct
/// signature. Signatures seen in earlier runs are loaded from the .txt files so a restart does not report them again.
/// </summary>
internal sealed class FindingStore
{
    private readonly Dictionary<string, int> _seen = new(StringComparer.Ordinal);

    public string Directory { get; }

    private readonly string _laneSuffix;

    /// <summary>Opens the folder for one lane; lanes share the findings but each keeps its own crash and restart state.</summary>
    public FindingStore(string outRoot, string targetName, int lane = 0)
    {
        _laneSuffix = lane > 0 ? "-" + lane : "";
        Directory = Path.Combine(outRoot, targetName);
        System.IO.Directory.CreateDirectory(Directory);
        foreach (var txt in System.IO.Directory.EnumerateFiles(Directory, "*.txt"))
        {
            var first = File.ReadLines(txt).FirstOrDefault(l => l.StartsWith("signature: ", StringComparison.Ordinal));
            if (first != null) _seen[first["signature: ".Length..]] = 1;
        }
    }

    public string CurrentInputPath => Path.Combine(Directory, ".current" + _laneSuffix);

    /// <summary>How many seeds the workers of this run have started, so a restarted worker can skip them.</summary>
    public int SeedsDone
    {
        get => File.Exists(SeedsPath) && int.TryParse(File.ReadAllText(SeedsPath), NumberStyles.None, CultureInfo.InvariantCulture, out var n) ? n : 0;
        set { if (value > 0) File.WriteAllText(SeedsPath, value.ToString(CultureInfo.InvariantCulture)); else File.Delete(SeedsPath); }
    }

    private string SeedsPath => Path.Combine(Directory, ".seeds-done" + _laneSuffix);

    public int Count => System.IO.Directory.EnumerateFiles(Directory, "*.txt").Count();

    /// <summary>Counts a repeat of a known signature; false when the signature is new.</summary>
    public bool Repeat(string signature)
    {
        if (!_seen.ContainsKey(signature)) return false;
        _seen[signature]++;
        return true;
    }

    public string Save(Target target, Verdict verdict, byte[] input, long originalSize, long seed, long executions)
    {
        _seen[verdict.Signature] = 1;
        var id = verdict.Kind + "-" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(verdict.Signature)))[..8].ToLowerInvariant();
        var inputName = id + target.Extension;
        WriteAtomically(inputName, input);
        var sb = new StringBuilder();
        sb.Append("signature: ").Append(verdict.Signature).Append('\n');
        sb.Append("target: ").Append(target.Name).Append('\n');
        sb.Append("kind: ").Append(verdict.Kind).Append('\n');
        sb.Append("input: ").Append(inputName).Append(" (").Append(input.Length).Append(" bytes");
        if (originalSize != input.Length) sb.Append(", minimised from ").Append(originalSize).Append(" bytes");
        sb.Append(")\n");
        sb.Append("seed: ").Append(seed).Append(", found after ").Append(executions).Append(" executions\n\n");
        sb.Append(verdict.Detail).Append('\n');
        WriteAtomically(id + ".txt", Encoding.UTF8.GetBytes(sb.ToString()));
        return inputName;
    }

    // Two lanes can find the same signature at once; each writes a private file and renames it into place, so the
    // folder always holds one whole finding rather than a mix of both.
    private void WriteAtomically(string name, byte[] data)
    {
        var temp = Path.Combine(Directory, "." + Guid.NewGuid().ToString("N") + ".tmp");
        File.WriteAllBytes(temp, data);
        try { File.Move(temp, Path.Combine(Directory, name), overwrite: true); }
        catch (IOException) { File.Delete(temp); } // the other lane's copy of the same finding stays
    }
}
