using System.Globalization;
using System.Text;

namespace HeadTuner;

/// <summary>
/// Lines of training text from a seeded generator: Danish words with and without æ, ø and å, Danish and Norwegian
/// place and person names, plain Danish, English (with capital AE, which must stay two letters), German, numbers,
/// codes with zeros next to letters, and punctuation. A fifth of the lines are in capitals, where the model is weakest.
/// </summary>
internal static class TrainingText
{
    private static readonly string[] Special = Split(
        "æble æbler ærlig ærme æske hæk kæde mæt næse pæn sæbe tæt væg læge træ kære være lære hære fæle blæse græs " +
        "øl øje øre ønske ønsker øst østlig ølkasse ødelagt øvelse bøf død før føre født grøn grød hør høre køb købe kø køre løn løve mød møde nød rød " +
        "rødgrød sø søn søster sød støj tøj tør vør fløde smør smørrebrød brød hjørne gøre spørge spørgsmål børn bøger bøgeskov skøn føde " +
        "å åben år årlig ål ånd åre båd bål blå blåbær gå går hånd hår kål lå lån må måned måske når på rå så stå tå vår vågen sådan således " +
        "blåbærsyltetøj færge færgen græsplæne mærkværdig rødbrun søndag lørdag fredag torsdag tirsdag ærtesuppe høstfest skærgård åkande");

    private static readonly string[] Names = Split(
        "Ærø Ærøskøbing Æbelø Ærtebjerg Ølstykke Ørsted Østerbro Østergade Øresund Øjvind Ølgod Åbenrå Ålborg Århus Åse Åby Åkirkeby " +
        "Ålsgårde Søren Bjørn Jørgen Køge Næstved Hørsholm Tårnby Sæby Fanø Rønne Mølle Løkken Højer Nørrebro Fåborg Græsted Bøgh " +
        "Sørensen Møller Kjær Grønlund Næsby");

    private static readonly string[] Plain = Split(
        "og i at det en den til er som på de med han af for ikke der var mig sig men et har om vi min havde ham hun nu over da fra du ud " +
        "sin dem os op man hans hvor eller hvad skal selv her alle vil blev kunne ind når være dog noget ville jo deres efter ned skulle " +
        "denne end dette mit også under have dig anden hende mine alt meget sit sine vor mod disse hvis din nogle hos blive mange ad bliver " +
        "hendes været thi jer sådan dag hus by vej land tid vand kaffe kage mad bil tog bus skole kontor faktura beløb moms side bilag stk " +
        "nr dato kunde ordre levering");

    private static readonly string[] English = Split(
        "the quick brown fox jumps over lazy dog and of to in is for on with as by at from that this report invoice total amount payment " +
        "order customer account number date page table figure section chapter results method data value price quantity Michael Rafael " +
        "Caesar Aesop aerial aerospace Aegean Israel Mae Oslo Boston London MICHAEL RAFAEL CAESAR AEROSPACE AEROSOL AEGEAN ISRAEL MAE " +
        "AESOP AERIAL PAELLA MAESTRO SUNDAE CAESAREA café résumé élite");

    private static readonly string[] German = Split("Größe Straße über Brücke Köln München Düsseldorf schön Grüße für Mädchen Ärger Öffnung Übung Bär Höhe");

    private static readonly string[] Punctuation =
        ["...", "…", "...", "…", "....", "........", ". . .", "(...)", "?", "!", ";", ":", ",", ".", "-", "–", "(ja)", "[nej]",
         "»her«", "\"godt\"", "'så'", "/", "%", "§", "&", "+", "="];

    private static readonly string[] Codes =
        ["SK0904", "X0Y", "2.0b", "B10", "A0", "B0", "H2O", "V0.5", "0x1F", "ID 0042", "R2D2", "SE0123", "DK-0450", "Ver. 10.0", "Q0",
         "0O", "ISO 9001", "EN 10204", "CO2", "NO2", "B05", "M0", "T-800", "OB-0710", "A4", "OK", "OO7", "007"];

    /// <summary>The lines of one page, the same for the same seed on every machine and runtime.</summary>
    public static List<string> Lines(int count, int seed)
    {
        var random = new Random(seed);
        var lines = new List<string>(count);
        for (int n = 0; n < count; n++)
        {
            var sb = new StringBuilder();
            int length = random.Next(22, 56);
            int mode = random.Next(10); // 0 and 1: capitals; 2: mostly English
            while (sb.Length < length)
            {
                int k = random.Next(100);
                string word = mode == 2 && k < 60 ? Pick(random, English)
                    : k < 30 ? Pick(random, Special)
                    : k < 42 ? Pick(random, Names)
                    : k < 65 ? Pick(random, Plain)
                    : k < 72 ? Pick(random, English)
                    : k < 76 ? Pick(random, German)
                    : k < 86 ? Number(random)
                    : Pick(random, Punctuation);
                if (k < 42 && random.Next(6) == 0) word = char.ToUpperInvariant(word[0]) + word[1..];
                if (sb.Length > 0)
                {
                    bool attached = word is "?" or "!" or ";" or ":" or "," or "." or "%" || (word is "..." or "…" && random.Next(2) == 0);
                    if (!attached) sb.Append(' ');
                }

                sb.Append(word);
            }

            string line = sb.ToString();
            lines.Add(mode < 2 ? line.ToUpperInvariant() : line);
        }

        return lines;
    }

    private static string Number(Random random) => random.Next(11) switch
    {
        0 => random.Next(1, 999).ToString(CultureInfo.InvariantCulture),
        1 => Invariant($"{random.Next(1, 99)}.{random.Next(100, 999)},{random.Next(10, 99)} kr."),
        2 => Invariant($"{random.Next(1, 28)}.{random.Next(1, 12)}.20{random.Next(10, 30)}"),
        3 => Invariant($"kl. {random.Next(7, 18)}.{random.Next(0, 6)}0"),
        4 => Invariant($"{random.Next(0, 9)},{random.Next(0, 99):00}"),
        5 => Invariant($"A{random.Next(0, 5)}"),
        6 => Invariant($"{random.Next(100, 9999)}0"),
        7 => Invariant($"{random.Next(10, 99)}-{random.Next(1000, 9999)}"),
        _ => Pick(random, Codes),
    };

    private static string Pick(Random random, string[] words) => words[random.Next(words.Length)];

    private static string Invariant(FormattableString value) => value.ToString(CultureInfo.InvariantCulture);

    private static string[] Split(string words) => words.Split(' ', StringSplitOptions.RemoveEmptyEntries);
}
