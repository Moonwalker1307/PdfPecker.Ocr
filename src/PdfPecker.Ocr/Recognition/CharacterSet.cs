namespace PdfPecker.Ocr.Recognition;

/// <summary>
/// The recognition model's output classes: the CTC blank at index 0, the model's characters, and a space appended
/// last, as rapidocr's <c>CTCLabelDecode</c> builds the list from the <c>character</c> metadata of the ONNX model.
/// </summary>
internal sealed class CharacterSet
{
    private readonly string[] _classes;

    /// <summary>Builds the classes from the model's character list (one entry per line of its metadata).</summary>
    public CharacterSet(IReadOnlyList<string> modelCharacters)
    {
        _classes = new string[modelCharacters.Count + 2];
        _classes[0] = string.Empty;
        for (int i = 0; i < modelCharacters.Count; i++)
        {
            _classes[i + 1] = modelCharacters[i];
        }

        _classes[^1] = " ";
    }

    /// <summary>The number of classes, which the recognition network's last dimension must equal.</summary>
    public int Count => _classes.Length;

    /// <summary>The index of the space class.</summary>
    public int Space => _classes.Length - 1;

    /// <summary>The text of a class; empty for the blank.</summary>
    public string this[int index] => _classes[index];

    /// <summary>
    /// The classes a decoder may choose from when only some characters are wanted: the blank, the space and every
    /// class whose text passes <paramref name="allow"/>, in ascending order.
    /// </summary>
    public int[] Subset(Func<string, bool> allow)
    {
        var list = new List<int> { 0 };
        for (int i = 1; i < _classes.Length - 1; i++)
        {
            if (allow(_classes[i]))
            {
                list.Add(i);
            }
        }

        list.Add(Space);
        return [.. list];
    }

    /// <summary>
    /// Whether every character of a class is in the Latin script or common to all scripts: ASCII, Latin-1, Latin
    /// Extended A and B, general punctuation, currency signs, and the letter-like and number forms. This is the
    /// subset for Western European documents, which lets the decoder skip the other 18,000 classes.
    /// </summary>
    public static bool IsLatin(string text)
    {
        foreach (char c in text)
        {
            bool ok = c is (>= ' ' and <= 'ɏ') or (>= ' ' and <= '⁯') or (>= '₠' and <= '⃏')
                or (>= '℀' and <= '↏') or (>= 'Ḁ' and <= 'ỿ');
            if (!ok)
            {
                return false;
            }
        }

        return text.Length > 0;
    }

    /// <summary>
    /// Whether a character belongs to a script written without spaces between words (Han, kana, CJK punctuation and
    /// full-width forms). rapidocr's word boxes give each such character a box of its own, and no space is inserted
    /// between them for a gap. This extends rapidocr's <c>is_chinese_char</c> with kana and the CJK extension A block.
    /// </summary>
    public static bool IsUnspaced(string text)
    {
        if (text.Length == 0)
        {
            return false;
        }

        char c = text[0];
        return c is (>= '一' and <= '鿿') or (>= '㐀' and <= '䶿') or (>= '　' and <= 'ヿ')
            or (>= '＀' and <= '￯') or (>= '豈' and <= '﫿') || char.IsHighSurrogate(c);
    }
}
