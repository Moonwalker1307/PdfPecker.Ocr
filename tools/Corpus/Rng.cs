namespace Corpus;

/// <summary>
/// A seeded SplitMix64 generator. The corpus must be identical on every machine and runtime, which System.Random does
/// not promise across versions, so the corpus has its own.
/// </summary>
sealed class Rng
{
    ulong _state;

    public Rng(ulong seed) => _state = seed;

    /// <summary>A seed derived from a base seed and a name, so that each page's noise is independent of the order pages are made in.</summary>
    public static Rng For(ulong seed, string name)
    {
        ulong h = 14695981039346656037UL;
        foreach (char c in name) h = (h ^ c) * 1099511628211UL;
        return new Rng(seed ^ h);
    }

    public ulong NextUInt64()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>An integer in [min, max).</summary>
    public int Next(int min, int max) => min + (int)(NextUInt64() % (ulong)(max - min));

    /// <summary>A double in [0, 1).</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1UL << 53));

    /// <summary>A standard normal sample (Box-Muller).</summary>
    public double NextGaussian()
    {
        double u1 = 1.0 - NextDouble(), u2 = NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    public T Pick<T>(IReadOnlyList<T> items) => items[Next(0, items.Count)];
}
