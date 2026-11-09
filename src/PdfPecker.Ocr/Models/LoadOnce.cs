namespace PdfPecker.Ocr.Models;

/// <summary>
/// A value built on first use and then shared, like <see cref="Lazy{T}"/> in its default mode, except that a factory
/// that throws is tried again on the next use. Lazy keeps the exception, so one out-of-memory failure while the first
/// call inflates the models would fail every later call until the process restarts. The factory runs under a lock, so
/// parallel first calls still build the value once rather than each inflating the models at the same time.
/// </summary>
internal sealed class LoadOnce<T>(Func<T> factory) where T : class
{
    private readonly object _gate = new();
    private T? _value;

    /// <summary>The value, built by the factory if no earlier use has built it.</summary>
    public T Value
    {
        get
        {
            T? value = Volatile.Read(ref _value);
            if (value is not null) return value;
            lock (_gate)
            {
                value = _value;
                if (value is null)
                {
                    value = factory();
                    Volatile.Write(ref _value, value);
                }
                return value;
            }
        }
    }
}
