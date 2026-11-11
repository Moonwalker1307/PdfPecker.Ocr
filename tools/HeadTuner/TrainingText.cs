using System.Globalization;
using System.Text;

namespace HeadTuner;

/// <summary>
/// Lines of training text from a seeded generator, each in one language: Danish, Norwegian, Swedish, Finnish,
/// Icelandic, German, French, Spanish, Portuguese, Italian or English. Each language has words with its accented
/// letters, names, plain words and its own quotation marks, so the refit sees every accented letter next to the plain
/// letters it is confused with. Numbers, codes with zeros next to letters, punctuation and English with capital AE
/// (which must stay two letters) are mixed in, and a fifth of the lines are in capitals, where the model is weakest.
/// </summary>
internal static class TrainingText
{
    private sealed record Language(string[] Special, string[] Names, string[] Plain, string[] Quotes, int Weight);

    private static readonly Language Danish = new(
        Split(
            "æble æbler ærlig ærme æske hæk kæde mæt næse pæn sæbe tæt væg læge træ kære være lære fæle blæse græs " +
            "øl øje øre ønske øst østlig ødelagt øvelse bøf død før føre født grøn grød hør høre køb kø køre løn løve møde nød rød " +
            "rødgrød sø søn søster sød støj tøj tør fløde smør smørrebrød brød hjørne gøre spørge spørgsmål børn bøger bøgeskov " +
            "å åben år årlig ål ånd åre båd bål blå blåbær gå går hånd hår kål lå lån må måned måske når på rå så stå tå vår vågen " +
            "sådan blåbærsyltetøj færge færgen græsplæne mærkværdig søndag lørdag ærtesuppe høstfest skærgård åkande"),
        Split(
            "Ærø Ærøskøbing Æbelø Ærtebjerg Ølstykke Ørsted Østerbro Østergade Øresund Øjvind Ølgod Åbenrå Ålborg Århus Åse Åby " +
            "Søren Bjørn Jørgen Køge Næstved Hørsholm Tårnby Sæby Fanø Rønne Løkken Nørrebro Fåborg Sørensen Møller Kjær Grønlund"),
        Split(
            "og i at det en den til er som de med han af for ikke der var mig sig men et har om vi min havde ham hun nu over da fra " +
            "du ud sin dem os op man hans hvor eller hvad skal selv her alle vil blev kunne ind dog noget ville jo deres efter ned " +
            "denne end dette mit også under have dig anden hende alt meget dag hus by vej land tid vand kaffe kage mad bil tog bus " +
            "skole kontor faktura beløb moms side bilag stk nr dato kunde ordre levering"),
        ["»", "«", "\"", "\""], 20);

    private static readonly Language Norwegian = new(
        Split(
            "ærlig være lære bær øy øya øre øl øst før høst brød grønn snø søster søndag tøy blåbærsyltetøy rømmegrøt fårikål " +
            "på år åpen gå må både båt fjør kjøre møte større første hjørne smøre søppel nærmere særlig vær"),
        Split("Ålesund Bodø Tromsø Kristiansand Lillehammer Mjøsa Trøndelag Østfold Bjørnstjerne Åsmund Sølvi Ørjan Bærum Hønefoss Ås"),
        Split("og i det som på er en til ikke med av for har de var jeg han hun vi ble kan skal fra om eller hva hvor også etter men da nå her alle"),
        ["«", "»", "\"", "\""], 9);

    private static readonly Language Swedish = new(
        Split(
            "åska åtta år åter även ändå ännu ärlig älg ängel öl öga ö över öppen öst höst höra kött köttbullar smörgås " +
            "smörgåsbord gräddsås färja färjan väg väder kväll själv sjö svår svåra både någon något här där sätt möte större " +
            "första tjänst lärare händer säker äntligen fönster sömn drömmar björn räkna"),
        Split("Åsa Örjan Älmhult Växjö Örebro Ängelholm Åre Malmö Göteborg Gävle Västerås Jönköping Umeå Luleå Söderström Åkesson Ström Bäckström Sjöberg Hägglund"),
        Split("och i att det som en på är av för med till den har de inte om ett han men var jag sig från vi så kan man när år sin nu ska"),
        ["”", "”", "’", "’"], 10);

    private static readonly Language Finnish = new(
        Split("äiti isä pöytä äänestys yö työ hyvä päivä jäätelö kävellä tänään sää lähellä hölmö töissä kesä talvi ystävä mökki järvi"),
        Split("Hämeenlinna Mäkelä Äänekoski Jyväskylä Hyvinkää Pyhäjärvi Mänttä Kärkkäinen Häkkinen Räikkönen Seinäjoki Pietarsaari"),
        Split("ja on ei se että hän oli mutta kun niin myös tai nyt sitten jo vain mitä siitä kanssa"),
        ["”", "”", "’", "’"], 5);

    private static readonly Language Icelandic = new(
        Split("þú það þetta þeir þing því sjá hvað maður góður dagur staður ráð fólk bók bær hús lífið veður fjörður jökull"),
        Split("Þórður Guðrún Þingvellir Húsavík Akureyri Ísafjörður Eyjafjörður Þórshöfn Ðorri Sigurður Hafnarfjörður Reykjavík"),
        Split("og að er í á um sem við til en það með ekki var"),
        ["\"", "\"", "'", "'"], 3);

    private static readonly Language German = new(
        Split(
            "Größe Straße über Brücke schön Grüße für Mädchen Ärger Öffnung Übung Bär Höhe Maße Fußgänger überqueren Ärzte prüfen " +
            "Öfen müssen können wäre hätte möchte während natürlich Bücher Häuser Söhne Töchter Gebäude Gemüse Frühstück Glück " +
            "Fräulein Mühle Künstler Lösung Bäckerei Übermut ändern Äpfel Öl übrigens"),
        Split("Köln München Düsseldorf Nürnberg Lübeck Göttingen Würzburg Saarbrücken Müller Schröder Bäcker Jäger Böhm Krüger Gößweinstein"),
        Split("und der die das ist nicht ein eine zu den mit von sich des auf für im dem auch es an werden aus er hat dass sie nach wird bei Frau Herr"),
        ["»", "«", "\"", "\""], 12);

    private static readonly Language French = new(
        Split(
            "été déjà où à là ça être tête fête forêt hôtel hôpital côté côte bientôt plutôt dîner île maïs Noël naïf cœur œuf œuvre " +
            "sœur bœuf mœurs Œdipe crème brûlée garçon français leçon reçu élève après très frère père mère problème deuxième voilà " +
            "écrire école étudiant événement préféré général théâtre âge pâte château goût août sûr mûr flûte"),
        Split("Élodie Hélène François Besançon Orléans Nîmes Évreux Sète Géraldine Jérôme Benoît Loïc Zoë Gaëlle Mégane Valérie Côte-d'Or"),
        Split("le la les de des du un une et est en que qui dans pour pas sur au avec il elle nous vous ils ce se ne plus par son mais ou comme"),
        ["«", "»", "“", "”"], 10);

    private static readonly Language Spanish = new(
        Split(
            "dónde está estación aquí niño niña piña jalapeños mañana árbol camión ñandú pingüino España señora Núñez atención " +
            "cigüeña vergüenza acción canción también así más después teléfono música número página fácil difícil último " +
            "sábado miércoles corazón jardín razón lápiz bebé café sofá rincón ¿qué? ¿cómo? ¿cuándo? ¿por qué?"),
        Split("Málaga Cádiz Córdoba León Logroño Ávila Jaén Peñíscola Gómez Pérez Sánchez Muñoz Ibáñez José María Ángel Inés Óscar Raúl"),
        Split("el la de que y en los se del las un por con no una su para es al lo como pero sus le ya o fue este ha sí porque esta"),
        ["“", "”", "«", "»"], 10);

    private static readonly Language Portuguese = new(
        Split("não são mãe irmão coração ação informação português você está avó avô três pôr pôde também é já há lá até então ótimo"),
        Split("São Paulo João Conceição Brasília Lisboa Évora Gonçalves Antônio Simões Magalhães Guimarães"),
        Split("de a o que e do da em um para é com uma os no se na por mais as dos como mas"),
        ["“", "”", "«", "»"], 4);

    private static readonly Language Italian = new(
        Split("città perché più già però è così università caffè lunedì martedì virtù tribù può Niccolò Forlì gioventù"),
        Split("Niccolò Forlì Cantù Mestrè Rossi Bianchi Colombo Ricci"),
        Split("di e il la che a per un in è non una sono le si con del della"),
        ["“", "”", "«", "»"], 3);

    private static readonly Language English = new(
        Split(
            "the quick brown fox jumps over lazy dog report invoice total amount payment order customer account number date page " +
            "table figure section chapter results method data value price quantity Michael Rafael Caesar Aesop aerial aerospace " +
            "Aegean Israel Mae MICHAEL RAFAEL CAESAR AEROSPACE AEROSOL AEGEAN ISRAEL MAE AESOP AERIAL PAELLA MAESTRO SUNDAE " +
            "PROBLEM PRODUCT DESIGN OPEN OFFICE POLICY DEPOT ODD OBOE DIOD PIPE PEOPLE DEEP POOL café résumé naïve"),
        Split("Oslo Boston London Paris Peter Paul David Daniel Diana Edward Elena Ivan Olivia Pedro Ida Uma Ulrich"),
        Split("and of to in is for on with as by at from that this it be are was have not or an all but they"),
        ["“", "”", "‘", "’"], 10);

    // Capitalised words that start with a Nordic letter, mixed into every Nordic line: Æ, Ø and Å at the start of a word
    // in mixed case are where the model is weakest.
    private static readonly string[] NordicCapitals = Split(
        "Ærlig Ærø Æble Æbler Ære Ærtebjerg Ærinde Øst Øster Østergade Øresund Øl Ølstykke Øjvind Øya Øvre Åben Åbenrå Åby Ålborg " +
        "Århus Åse Ålesund Åsa Åre Älg Älmhult Ängel Ängelholm Äntligen Öl Örebro Örjan Östra Öland Ödeshög Ågatan Åtta");

    private static readonly Language[] Languages = [Danish, Norwegian, Swedish, Finnish, Icelandic, German, French, Spanish, Portuguese, Italian, English];

    private static readonly int TotalWeight = Languages.Sum(l => l.Weight);

    private static readonly string[] Punctuation =
        ["...", "…", "...", "....", ". . .", "(...)", "?", "!", ";", ":", ",", ".", "–", "–", "-", "(ja)", "[nej]", "\"godt\"",
         "'så'", "/", "%", "§ 3", "§§ 12-14", "§ 7, stk. 2", "&", "+", "=", "¿sí?", "1–2", "12–14", "3-4", "9.30-11.00", "kl. 8–16",
         "E-Mail", "e-mail", "T-shirt", "ÊTES-VOUS", "Saint-Denis", "Schleswig-Holstein", "it's—really", "word—word", "2-3", "A-B"];

    private static readonly string[] Codes =
        ["SK0904", "X0Y", "2.0b", "B10", "A0", "B0", "H2O", "V0.5", "0x1F", "ID 0042", "R2D2", "SE0123", "DK-0450", "Ver. 10.0", "Q0",
         "0O", "ISO 9001", "EN 10204", "CO2", "NO2", "B05", "M0", "T-800", "OB-0710", "A4", "OK", "OO7", "007", "PDF", "DPI", "P0", "D0"];

    /// <summary>The lines of one page, the same for the same seed on every machine and runtime.</summary>
    public static List<string> Lines(int count, int seed)
    {
        var random = new Random(seed);
        var lines = new List<string>(count);
        for (int n = 0; n < count; n++)
        {
            Language language = PickLanguage(random);
            var sb = new StringBuilder();
            int length = random.Next(22, 56);
            bool capitals = random.Next(10) < 3;
            while (sb.Length < length)
            {
                int k = random.Next(100);
                string word = k < 34 ? Pick(random, language.Special)
                    : k < 46 ? Pick(random, language.Names)
                    : k < 68 ? Pick(random, language.Plain)
                    : k < 73 ? Pick(random, English.Special)
                    : k < 83 ? Number(random)
                    : k < 88 ? Quoted(random, language)
                    : Pick(random, Punctuation);
                if (k < 46 && random.Next(6) == 0) word = char.ToUpperInvariant(word[0]) + word[1..];
                if (k < 10 && (language == Danish || language == Norwegian || language == Swedish)) word = Pick(random, NordicCapitals);
                if (sb.Length > 0)
                {
                    bool attached = word is "?" or "!" or ";" or ":" or "," or "." or "%" || (word is "..." or "…" && random.Next(2) == 0);
                    if (!attached) sb.Append(' ');
                }

                sb.Append(word);
            }

            string line = sb.ToString();
            lines.Add(capitals ? line.ToUpperInvariant() : line);
        }

        return lines;
    }

    private static Language PickLanguage(Random random)
    {
        int w = random.Next(TotalWeight);
        foreach (Language language in Languages)
        {
            if (w < language.Weight) return language;
            w -= language.Weight;
        }

        return English;
    }

    // A word in the language's double or single quotation marks.
    private static string Quoted(Random random, Language language)
    {
        int pair = random.Next(2) * 2;
        return language.Quotes[pair] + Pick(random, random.Next(2) == 0 ? language.Special : language.Plain) + language.Quotes[pair + 1];
    }

    private static string Number(Random random) => random.Next(12) switch
    {
        0 => random.Next(1, 999).ToString(CultureInfo.InvariantCulture),
        1 => Invariant($"{random.Next(1, 99)}.{random.Next(100, 999)},{random.Next(10, 99)} kr."),
        2 => Invariant($"{random.Next(1, 28)}.{random.Next(1, 12)}.20{random.Next(10, 30)}"),
        3 => Invariant($"kl. {random.Next(7, 18)}.{random.Next(0, 6)}0"),
        4 => Invariant($"{random.Next(0, 9)},{random.Next(0, 99):00}"),
        5 => Invariant($"A{random.Next(0, 5)}"),
        6 => Invariant($"{random.Next(100, 9999)}0"),
        7 => Invariant($"{random.Next(10, 99)}-{random.Next(1000, 9999)}"),
        8 => Invariant($"{random.Next(1, 9)}.º"),
        9 => random.Next(3) switch
        {
            // The degree sign looks much like the ordinal º, and French writes "N°" with it.
            0 => Invariant($"N° {random.Next(1, 999)}"),
            1 => Invariant($"{random.Next(-20, 40)} °C"),
            _ => Invariant($"{random.Next(1, 360)}°"),
        },
        _ => Pick(random, Codes),
    };

    private static string Pick(Random random, string[] words) => words[random.Next(words.Length)];

    private static string Invariant(FormattableString value) => value.ToString(CultureInfo.InvariantCulture);

    private static string[] Split(string words) => words.Split(' ', StringSplitOptions.RemoveEmptyEntries);
}
