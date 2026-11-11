namespace HeadTuner;

/// <summary>
/// Refits the last layer's weights of the tuned classes by minimising the cross-entropy of the full softmax over the
/// samples, with every other class's logit fixed, plus an L2 pull towards the original weights. Adam on minibatches, in
/// double precision and a fixed order, so a run is reproducible.
/// </summary>
internal static class Fit
{
    /// <summary>
    /// Returns the weights [class][K + 1] (the bias last) after <paramref name="epochs"/> passes. Rows in
    /// <paramref name="frozen"/> keep their original weights but still compete in the softmax.
    /// </summary>
    public static double[][] Run(List<Sample> samples, double[][] original, ISet<int> frozen, int epochs, double learningRate, double lambda, int seed, TextWriter log)
    {
        int m = original.Length, k = original[0].Length - 1;
        double[][] w = [.. original.Select(r => (double[])r.Clone())];
        var m1 = new double[m][];
        var m2 = new double[m][];
        for (int j = 0; j < m; j++)
        {
            m1[j] = new double[k + 1];
            m2[j] = new double[k + 1];
        }

        int[] order = [.. Enumerable.Range(0, samples.Count)];
        var random = new Random(seed);
        double totalWeight = samples.Sum(s => (double)s.Weight);
        const int Batch = 512;
        int step = 0;
        var logits = new double[m];
        log.WriteLine($"  {samples.Count:N0} samples, loss {Loss(samples, w, logits) / totalWeight:0.00000} before");
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            random.Shuffle(order);
            for (int start = 0; start < order.Length; start += Batch)
            {
                int end = Math.Min(order.Length, start + Batch);
                var gradient = new double[m][];
                for (int j = 0; j < m; j++) gradient[j] = new double[k + 1];
                double batchWeight = 0;
                for (int b = start; b < end; b++)
                {
                    Sample s = samples[order[b]];
                    batchWeight += s.Weight;
                    double lse = Logits(s, w, logits);
                    for (int j = 0; j < m; j++)
                    {
                        if (frozen.Contains(j)) continue;
                        double g = s.Weight * (Math.Exp(logits[j] - lse) - (s.Target == j ? 1 : 0));
                        for (int i = 0; i < k; i++) gradient[j][i] += g * s.Features[i];
                        gradient[j][k] += g;
                    }
                }

                step++;
                double c1 = 1 - Math.Pow(0.9, step), c2 = 1 - Math.Pow(0.999, step);
                for (int j = 0; j < m; j++)
                {
                    if (frozen.Contains(j)) continue;
                    for (int i = 0; i <= k; i++)
                    {
                        double g = (gradient[j][i] / batchWeight) + (2 * lambda * (w[j][i] - original[j][i]));
                        m1[j][i] = (0.9 * m1[j][i]) + (0.1 * g);
                        m2[j][i] = (0.999 * m2[j][i]) + (0.001 * g * g);
                        w[j][i] -= learningRate * (m1[j][i] / c1) / (Math.Sqrt(m2[j][i] / c2) + 1e-8);
                    }
                }
            }

            log.WriteLine($"  epoch {epoch + 1}: loss {Loss(samples, w, logits) / totalWeight:0.00000}");
        }

        return w;
    }

    /// <summary>Fills the tuned classes' logits and returns the log of the softmax's normaliser.</summary>
    private static double Logits(Sample s, double[][] w, double[] logits)
    {
        int k = s.Features.Length;
        double max = s.LogRest;
        for (int j = 0; j < w.Length; j++)
        {
            double v = w[j][k];
            for (int i = 0; i < k; i++) v += w[j][i] * s.Features[i];
            logits[j] = v;
            max = Math.Max(max, v);
        }

        double sum = Math.Exp(s.LogRest - max);
        foreach (double v in logits) sum += Math.Exp(v - max);
        return max + Math.Log(sum);
    }

    private static double Loss(List<Sample> samples, double[][] w, double[] logits)
    {
        double loss = 0;
        foreach (Sample s in samples)
        {
            double lse = Logits(s, w, logits);
            loss += s.Weight * (lse - (s.Target >= 0 ? logits[s.Target] : s.FixedLogit));
        }

        return loss;
    }
}
