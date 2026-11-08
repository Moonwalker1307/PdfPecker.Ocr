using System.Text;
using System.Text.RegularExpressions;

namespace PdfPecker.Ocr.Fuzz;

/// <summary>
/// Turns a corpus entry into a new input. Without coverage feedback the mutations have to be broad, so every call
/// stacks one to four of them: byte-level ones for every format and token-level ones for the text formats.
/// </summary>
internal sealed class Mutator
{
    private static readonly long[] Interesting = { 0, 1, -1, 2, 127, 128, 255, 256, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000L, 0xffffffffL };

    private static readonly string[] InterestingNumbers =
    {
        "0", "-1", "1", "2147483647", "2147483648", "4294967295", "65535", "65536", "16777216", "99999999999",
        "1e308", "-1e308", "1e-308", "0.0000001", "NaN", "Infinity", "100000", "-100000",
    };

    // Splits text into tags, comments, words, white space and single punctuation characters.
    private static readonly Regex TokenPattern = new(@"<!--.*?-->|</?[A-Za-z][^>]*>|[A-Za-z0-9_\-#.%]+|\s+|.", RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex NumberPattern = new(@"^-?\d+(\.\d+)?$", RegexOptions.Compiled);

    private readonly Random _rng;
    private readonly int _maxSize;
    private readonly string[] _dictionary;
    private readonly (string Open, string Close)[] _nesters;

    public Mutator(Random rng, int maxSize, string[] dictionary, (string Open, string Close)[] nesters)
    {
        _rng = rng;
        _maxSize = maxSize;
        _dictionary = dictionary;
        _nesters = nesters;
    }

    public byte[] Mutate(byte[] input, IReadOnlyList<byte[]> corpus, bool text)
    {
        var data = input;
        int rounds = 1 + _rng.Next(4);
        for (int i = 0; i < rounds; i++)
        {
            data = text && _rng.Next(10) < 6 ? MutateText(data, corpus) : MutateBytes(data, corpus);
            if (data.Length > _maxSize) Array.Resize(ref data, _maxSize);
        }
        return data;
    }

    private byte[] MutateBytes(byte[] d, IReadOnlyList<byte[]> corpus)
    {
        if (d.Length == 0) return RandomBytes(1 + _rng.Next(16));
        switch (_rng.Next(11))
        {
            case 0: // flip a few bits
            {
                var r = (byte[])d.Clone();
                for (int n = 1 + _rng.Next(4); n > 0; n--) r[_rng.Next(r.Length)] ^= (byte)(1 << _rng.Next(8));
                return r;
            }
            case 1: // replace a few bytes
            {
                var r = (byte[])d.Clone();
                for (int n = 1 + _rng.Next(4); n > 0; n--) r[_rng.Next(r.Length)] = (byte)_rng.Next(256);
                return r;
            }
            case 2: // insert random bytes
                return Insert(d, _rng.Next(d.Length + 1), RandomBytes(1 + _rng.Next(32)));
            case 3: // delete a chunk
            {
                int len = Math.Min(d.Length, 1 + _rng.Next(Math.Max(1, d.Length / 4)));
                int at = _rng.Next(d.Length - len + 1);
                return Remove(d, at, len);
            }
            case 4: // duplicate a chunk in place, possibly many times
            {
                int len = Math.Min(d.Length, 1 + _rng.Next(Math.Min(64, d.Length)));
                int at = _rng.Next(d.Length - len + 1);
                int times = _rng.Next(4) == 0 ? 2 + _rng.Next(300) : 1 + _rng.Next(3);
                var chunk = d.AsSpan(at, len).ToArray();
                var r = new List<byte>(d.Length + len * times);
                r.AddRange(d.AsSpan(0, at + len).ToArray());
                for (int t = 0; t < times && r.Count < _maxSize; t++) r.AddRange(chunk);
                r.AddRange(d.AsSpan(at + len).ToArray());
                return r.ToArray();
            }
            case 5: // interesting integers of one, two or four bytes in either byte order
            case 6:
            {
                var r = (byte[])d.Clone();
                int width = 1 << _rng.Next(3);
                if (r.Length < width) return r;
                int at = _rng.Next(r.Length - width + 1);
                long v = Interesting[_rng.Next(Interesting.Length)];
                bool big = _rng.Next(2) == 0;
                for (int b = 0; b < width; b++)
                {
                    int shift = 8 * (big ? width - 1 - b : b);
                    r[at + b] = (byte)(v >> shift);
                }
                return r;
            }
            case 7: // splice a chunk of another corpus entry in
            {
                var other = corpus[_rng.Next(corpus.Count)];
                if (other.Length == 0) return d;
                int len = Math.Min(other.Length, 1 + _rng.Next(Math.Max(1, other.Length / 2)));
                int from = _rng.Next(other.Length - len + 1);
                var chunk = other.AsSpan(from, len).ToArray();
                return _rng.Next(2) == 0 ? Insert(d, _rng.Next(d.Length + 1), chunk) : Overwrite(d, _rng.Next(d.Length), chunk);
            }
            case 8: // keep the head of one entry and the tail of another
            {
                var other = corpus[_rng.Next(corpus.Count)];
                int cutA = _rng.Next(d.Length + 1);
                int cutB = _rng.Next(other.Length + 1);
                return d.AsSpan(0, cutA).ToArray().Concat(other.AsSpan(cutB).ToArray()).ToArray();
            }
            case 9: // zero or fill a region
            {
                var r = (byte[])d.Clone();
                int len = Math.Min(r.Length, 1 + _rng.Next(32));
                int at = _rng.Next(r.Length - len + 1);
                Array.Fill(r, _rng.Next(2) == 0 ? (byte)0 : (byte)0xFF, at, len);
                return r;
            }
            default: // truncate
                return d.AsSpan(0, _rng.Next(d.Length)).ToArray();
        }
    }

    private byte[] MutateText(byte[] bytes, IReadOnlyList<byte[]> corpus)
    {
        var tokens = TokenPattern.Matches(Encoding.UTF8.GetString(bytes)).Select(m => m.Value).ToList();
        if (tokens.Count == 0) return Encoding.UTF8.GetBytes(_dictionary.Length > 0 ? _dictionary[_rng.Next(_dictionary.Length)] : "x");
        int op = _rng.Next(8);
        if (op is 1 or 2 && _nesters.Length == 0) op = 0;
        if (op == 6 && !tokens.Any(t => NumberPattern.IsMatch(t))) op = 5;
        switch (op)
        {
            case 0: // duplicate a run of tokens
            {
                int len = Math.Min(tokens.Count, 1 + _rng.Next(_rng.Next(3) == 0 ? 40 : 4));
                int at = _rng.Next(tokens.Count - len + 1);
                int times = _rng.Next(4) == 0 ? 20 + _rng.Next(2000) : 1 + _rng.Next(5);
                var run = tokens.GetRange(at, len);
                var sb = new StringBuilder();
                for (int t = 0; t < times && sb.Length < _maxSize; t++) foreach (var tok in run) sb.Append(tok);
                tokens.Insert(at + len, sb.ToString());
                break;
            }
            case 1: // nest a run of tokens in deep pairs
            case 2:
            {
                var (open, close) = _nesters[_rng.Next(_nesters.Length)];
                int depth = _rng.Next(3) == 0 ? 100 + _rng.Next(3000) : 1 + _rng.Next(40);
                int len = Math.Min(tokens.Count, _rng.Next(1 + Math.Min(tokens.Count, 30)));
                int at = _rng.Next(tokens.Count - len + 1);
                var pre = new StringBuilder();
                var post = new StringBuilder();
                for (int t = 0; t < depth && pre.Length < _maxSize / 2; t++) { pre.Append(open); post.Append(close); }
                if (_rng.Next(4) == 0) post.Clear(); // an unclosed pile of openers is its own kind of hostile
                tokens.Insert(at + len, post.ToString());
                tokens.Insert(at, pre.ToString());
                break;
            }
            case 3: // delete a run of tokens
            {
                int len = Math.Min(tokens.Count, 1 + _rng.Next(Math.Max(1, tokens.Count / 8)));
                tokens.RemoveRange(_rng.Next(tokens.Count - len + 1), len);
                break;
            }
            case 4: // swap two tokens
            {
                int a = _rng.Next(tokens.Count), b = _rng.Next(tokens.Count);
                (tokens[a], tokens[b]) = (tokens[b], tokens[a]);
                break;
            }
            case 5: // insert a dictionary word, or a token from another entry
            {
                string word;
                if (_dictionary.Length > 0 && _rng.Next(3) != 0) word = _dictionary[_rng.Next(_dictionary.Length)];
                else
                {
                    var other = TokenPattern.Matches(Encoding.UTF8.GetString(corpus[_rng.Next(corpus.Count)])).Select(m => m.Value).ToList();
                    word = other.Count == 0 ? "x" : other[_rng.Next(other.Count)];
                }
                tokens.Insert(_rng.Next(tokens.Count + 1), word);
                break;
            }
            case 6: // replace a number with an interesting one
            {
                var numbered = Enumerable.Range(0, tokens.Count).Where(i => NumberPattern.IsMatch(tokens[i])).ToList();
                tokens[numbered[_rng.Next(numbered.Count)]] = InterestingNumbers[_rng.Next(InterestingNumbers.Length)];
                break;
            }
            default: // repeat one token a very large number of times
            {
                int at = _rng.Next(tokens.Count);
                var tok = tokens[at];
                if (tok.Length == 0) break;
                int times = Math.Min(_maxSize / Math.Max(1, tok.Length), 50 + _rng.Next(20000));
                tokens[at] = new StringBuilder(tok.Length * times).Insert(0, tok, times).ToString();
                break;
            }
        }
        return Encoding.UTF8.GetBytes(string.Concat(tokens));
    }

    /// <summary>The tokens of a text input as byte arrays, for the minimiser to drop whole tokens at a time.</summary>
    public static List<byte[]> Tokenize(byte[] bytes)
        => TokenPattern.Matches(Encoding.UTF8.GetString(bytes)).Select(m => Encoding.UTF8.GetBytes(m.Value)).ToList();

    private byte[] RandomBytes(int n)
    {
        var r = new byte[n];
        _rng.NextBytes(r);
        return r;
    }

    private static byte[] Insert(byte[] d, int at, byte[] chunk)
    {
        var r = new byte[d.Length + chunk.Length];
        Buffer.BlockCopy(d, 0, r, 0, at);
        Buffer.BlockCopy(chunk, 0, r, at, chunk.Length);
        Buffer.BlockCopy(d, at, r, at + chunk.Length, d.Length - at);
        return r;
    }

    private static byte[] Remove(byte[] d, int at, int len)
    {
        var r = new byte[d.Length - len];
        Buffer.BlockCopy(d, 0, r, 0, at);
        Buffer.BlockCopy(d, at + len, r, at, d.Length - at - len);
        return r;
    }

    private static byte[] Overwrite(byte[] d, int at, byte[] chunk)
    {
        var r = (byte[])d.Clone();
        Buffer.BlockCopy(chunk, 0, r, at, Math.Min(chunk.Length, r.Length - at));
        return r;
    }
}
