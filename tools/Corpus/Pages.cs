using System.Globalization;

namespace Corpus;

/// <summary>One generated page before it is rendered: its HTML, and the ground truth text and the settings the manifest records.</summary>
sealed record PageSpec(string Id, string Category, string Lang, string Font, bool Colour, string Html, IReadOnlyList<string> Lines);

/// <summary>
/// The page templates. The texts are written for this corpus (none is quoted from a book or a website), in English,
/// Danish, German and French, so the corpus has no licence of its own beyond the repository's.
/// </summary>
static class Pages
{
    const string Sans = "'Liberation Sans', sans-serif";
    const string Serif = "'Liberation Serif', serif";
    const string Mono = "'Liberation Mono', monospace";

    sealed record Lang(
        string Code, string Name, string[] Months, string Invoice, string InvoiceDate, string DueDate, string Customer, string CustomerNo,
        string[] Columns, string Subtotal, string Vat, string Total, string Payment, string Currency, bool CurrencyFirst,
        char Thousands, char Decimal, string[] Items, string Company, string[] Address, string Email, string Url, string[] LetterLines,
        string[] LetterBody, string LetterClosing, string[] LetterSender);

    static readonly Lang En = new("en", "English", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        "Invoice", "Invoice date", "Due date", "Customer", "Customer no.", ["Description", "Qty", "Unit price", "Amount"],
        "Subtotal", "VAT 20%", "Total due", "Please pay within 30 days to the account below, quoting the invoice number.", "$", true, ',', '.',
        ["Annual software licence", "On-site installation and training", "Support hours, March", "Replacement power supply unit", "Shipping and handling"],
        "Northgate Services Ltd", ["14 Harbour Road", "Leeds LS1 4AB", "United Kingdom"], "billing@northgate.example", "https://www.northgate.example/pay",
        ["14 March 2026", "Ms Eleanor Carter", "Finance Department", "Oakfield Logistics", "22 Mill Lane, Bristol BS3 5QD"],
        ["Dear Ms Carter,",
         "Thank you for your letter of 14 October about the new agreement for the maintenance of the ventilation system in your office building. We have reviewed the offer and can confirm that the price is unchanged from last year, even though the cost of spare parts has risen.",
         "Our engineers will visit on Thursday 6 November between 8 and 12 o'clock. Please make sure that they can reach the basement and the attic, so that the filters can be replaced and the motors checked. If the time does not suit you, please call us on 020 7946 0123. We need to know about any larger changes a week in advance, because the street will be closed."],
        "Yours sincerely,", ["Jonathan Weir", "Head of Customer Service, Fresh Air Ltd"]);

    static readonly Lang Da = new("da", "Danish", ["januar", "februar", "marts", "april", "maj", "juni", "juli", "august", "september", "oktober", "november", "december"],
        "Faktura", "Fakturadato", "Forfaldsdato", "Kunde", "Kundenr.", ["Beskrivelse", "Antal", "Stykpris", "Beløb"],
        "Subtotal", "Moms 25%", "Total at betale", "Betal venligst inden 30 dage til kontoen nedenfor og angiv fakturanummeret.", "kr.", false, '.', ',',
        ["Årlig softwarelicens", "Installation og oplæring på stedet", "Supporttimer i marts", "Ny strømforsyning til serveren", "Fragt og håndtering"],
        "Grøn Luft A/S", ["Havnegade 14", "8000 Aarhus C", "Danmark"], "faktura@gronluft.example", "https://www.gronluft.example/betal",
        ["14. marts 2026", "Fru Mette Nielsen", "Økonomiafdelingen", "Østerbro Logistik ApS", "Strandvejen 22, 2100 København Ø"],
        ["Kære Mette Nielsen,",
         "Tak for din henvendelse af 14. oktober vedrørende den nye aftale om vedligeholdelse af kontorets ventilationsanlæg. Vi har gennemgået tilbuddet og kan bekræfte, at prisen er uændret i forhold til sidste år, selvom udgifterne til reservedele er steget.",
         "Teknikerne kommer på besøg torsdag den 6. november mellem klokken 8 og 12. Husk at give dem adgang til kælderen og til tagetagen, så de kan udskifte filtrene og kontrollere motorerne. Hvis tidspunktet ikke passer, er du velkommen til at ringe til os på 70 20 30 40. Større ændringer skal vi kende en uge i forvejen, fordi vejen bliver lukket."],
        "Med venlig hilsen", ["Søren Østergaard", "Servicechef, Grøn Luft A/S"]);

    static readonly Lang De = new("de", "German", ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
        "Rechnung", "Rechnungsdatum", "Fälligkeitsdatum", "Kunde", "Kundennr.", ["Beschreibung", "Menge", "Einzelpreis", "Betrag"],
        "Zwischensumme", "MwSt. 19%", "Gesamtbetrag", "Bitte überweisen Sie den Betrag innerhalb von 30 Tagen unter Angabe der Rechnungsnummer.", "€", false, '.', ',',
        ["Jährliche Softwarelizenz", "Installation und Schulung vor Ort", "Supportstunden im März", "Ersatznetzteil für den Server", "Versand und Verpackung"],
        "Frischluft GmbH", ["Hafenstraße 14", "20457 Hamburg", "Deutschland"], "rechnung@frischluft.example", "https://www.frischluft.example/zahlen",
        ["14. März 2026", "Frau Anna Müller", "Buchhaltung", "Süd-Logistik GmbH", "Gartenweg 22, 80331 München"],
        ["Sehr geehrte Frau Müller,",
         "vielen Dank für Ihr Schreiben vom 14. Oktober zur geplanten Wartung der Lüftungsanlage in Ihrem Bürogebäude. Wir haben das Angebot geprüft und können bestätigen, dass der Preis gegenüber dem Vorjahr unverändert bleibt, obwohl die Kosten für Ersatzteile gestiegen sind.",
         "Unsere Techniker kommen am Donnerstag, dem 6. November, zwischen 8 und 12 Uhr vorbei. Bitte sorgen Sie dafür, dass sie Zugang zum Keller und zum Dachboden haben, damit die Filter ausgetauscht und die Motoren überprüft werden können. Falls der Termin nicht passt, rufen Sie uns gerne unter 030 5550 1234 an. Größere Änderungen müssen wir spätestens eine Woche vorher wissen, weil die Straße gesperrt wird."],
        "Mit freundlichen Grüßen", ["Jürgen Weiß", "Leiter Kundendienst, Frischluft GmbH"]);

    static readonly Lang Fr = new("fr", "French", ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
        "Facture", "Date de facture", "Date d'échéance", "Client", "N° client", ["Désignation", "Qté", "Prix unitaire", "Montant"],
        "Sous-total", "TVA 20 %", "Total à payer", "Merci de régler sous 30 jours sur le compte ci-dessous en indiquant le numéro de facture.", "€", false, ' ', ',',
        ["Licence logicielle annuelle", "Installation et formation sur site", "Heures d'assistance en mars", "Alimentation de remplacement", "Frais d'expédition et de manutention"],
        "Air Frais SARL", ["14 quai des Chartrons", "33000 Bordeaux", "France"], "facturation@airfrais.example", "https://www.airfrais.example/payer",
        ["14 mars 2026", "Madame Chloé Lefèvre", "Service comptabilité", "Logistique Côte d'Azur SA", "22 rue des Écoles, 06000 Nice"],
        ["Chère Madame Lefèvre,",
         "Nous vous remercions de votre courrier du 14 octobre concernant le contrat d'entretien du système de ventilation de vos bureaux. Après examen de l'offre, nous pouvons confirmer que le prix reste identique à celui de l'année dernière, malgré la hausse du coût des pièces détachées.",
         "Nos techniciens passeront le jeudi 6 novembre entre 8 h et 12 h. Merci de leur garantir l'accès à la cave et au grenier afin qu'ils puissent remplacer les filtres et vérifier les moteurs. Si ce créneau ne vous convient pas, appelez-nous au 01 55 50 12 34. Étant donné les travaux dans la rue, nous vous prions de nous prévenir au plus tard une semaine à l'avance."],
        "Veuillez agréer, Madame, nos salutations distinguées.", ["François Dubois", "Responsable du service client, Air Frais SARL"]);

    static string Money(decimal v, Lang l, bool symbol = true)
    {
        long cents = (long)Math.Round(v * 100m, MidpointRounding.AwayFromZero);
        long whole = cents / 100;
        var digits = whole.ToString(CultureInfo.InvariantCulture);
        var grouped = new System.Text.StringBuilder();
        for (int i = 0; i < digits.Length; i++)
        {
            if (i > 0 && (digits.Length - i) % 3 == 0) grouped.Append(l.Thousands);
            grouped.Append(digits[i]);
        }
        var s = $"{grouped}{l.Decimal}{cents % 100:00}";
        if (!symbol) return s;
        return l.CurrencyFirst ? l.Currency + s : s + " " + l.Currency;
    }

    static string Date(Lang l, int day, int month, int year) => l.Code switch
    {
        "en" => $"{day} {l.Months[month - 1]} {year}",
        "da" or "de" => $"{day}. {l.Months[month - 1]} {year}",
        _ => $"{day}{(day == 1 ? "er" : "")} {l.Months[month - 1]} {year}",
    };

    public static List<PageSpec> All(ulong seed)
    {
        var pages = new List<PageSpec>
        {
            Invoice("invoice-en", En, seed, colour: false),
            Invoice("invoice-da", Da, seed, colour: false),
            Invoice("invoice-de", De, seed, colour: true),
            Invoice("invoice-fr", Fr, seed, colour: true),
            Letter("letter-en", En), Letter("letter-da", Da), Letter("letter-de", De), Letter("letter-fr", Fr),
            Article("article-en", En, ArticleEn),
            Article("article-de", De, ArticleDe),
            Grid("table-en", En, seed),
            Grid("table-da", Da, seed),
            SmallPrint("smallprint-en", En, SmallEn, Sans),
            SmallPrint("smallprint-fr", Fr, SmallFr, Serif),
            Headings("headings-en", En, "Annual Report 2025", "Highlights of the year", "Outlook for 2026", "Next steps", HeadingEn),
            Headings("headings-da", Da, "Årsrapport 2025", "Årets resultater", "Udsigter for 2026", "De næste skridt", HeadingDa),
            LogPage("mono-en", seed),
            Identifiers("identifiers-en", seed),
        };
        return pages;
    }

    // ------------------------------------------------------------------ invoice

    static PageSpec Invoice(string id, Lang l, ulong seed, bool colour)
    {
        var rng = Rng.For(seed, id);
        var d = new Doc();
        string accent = colour ? "color:#1a3a6b;" : "";
        string band = colour ? "background:#dbe8f7;padding:6pt 8pt;" : "";
        d.Line(l.Company, $"font-size:17pt;font-weight:bold;{accent}{band}");
        foreach (var a in l.Address) d.Line(a);
        d.Gap("10mm");
        int invoiceNo = rng.Next(1000, 9999);
        d.Line($"{l.Invoice} 2026-{invoiceNo}", $"font-size:20pt;font-weight:bold;{accent}");
        d.Gap("4mm");
        int day = rng.Next(1, 28), month = rng.Next(1, 12);
        d.Line($"{l.InvoiceDate}: {Date(l, day, month, 2026)}");
        d.Line($"{l.DueDate}: {Date(l, day, month + 1 > 12 ? 12 : month + 1, 2026)}");
        d.Line($"{l.Customer}: {l.LetterLines[3]}");
        d.Line($"{l.CustomerNo}: {rng.Next(10000, 99999)}");
        d.Gap("8mm");

        var rows = new List<string[]> { l.Columns };
        decimal subtotal = 0;
        foreach (var item in l.Items)
        {
            int qty = rng.Next(1, 12);
            decimal unit = rng.Next(1500, 99900) / 100m;
            decimal amount = qty * unit;
            subtotal += amount;
            rows.Add([item, qty.ToString(CultureInfo.InvariantCulture), Money(unit, l, false), Money(amount, l, false)]);
        }
        d.Table(rows, [false, true, true, true], ["92mm", "16mm", "30mm", "30mm"], headerRow: true,
            tableStyle: "width:168mm",
            headerStyle: colour ? "font-weight:bold;background:#1a3a6b;color:#fff" : "font-weight:bold;border-bottom:1pt solid #000",
            stripe: colour ? "#eef3fa" : "");
        d.Gap("5mm");
        decimal vat = Math.Round(subtotal * (l.Code == "en" ? 0.20m : l.Code == "da" ? 0.25m : l.Code == "de" ? 0.19m : 0.20m), 2);
        var totals = new List<string[]>
        {
            new[] { l.Subtotal, Money(subtotal, l) },
            new[] { l.Vat, Money(vat, l) },
            new[] { l.Total, Money(subtotal + vat, l) },
        };
        d.Table(totals, [false, true], ["110mm", "58mm"], headerRow: false, tableStyle: "width:168mm;font-weight:bold");
        d.Gap("12mm");
        foreach (var line in Doc.Wrap(l.Payment, 78)) d.Line(line);
        d.Line($"IBAN {l.Code.ToUpperInvariant()}{rng.Next(10, 99)} {rng.Next(1000, 9999)} {rng.Next(1000, 9999)} {rng.Next(1000, 9999)} {rng.Next(10, 99)}");
        d.Line($"{l.Email}  |  {l.Url}", "font-size:9pt");
        return new PageSpec(id, "invoice", l.Code, "sans", colour, d.ToHtml(l.Code, Sans, "11pt", "1.4"), d.Lines);
    }

    // ------------------------------------------------------------------ letter

    static PageSpec Letter(string id, Lang l)
    {
        var d = new Doc();
        d.Line(l.LetterSender[0] + " - " + l.Company, "text-align:right");
        foreach (var a in l.Address) d.Line(a, "text-align:right");
        d.Gap("12mm");
        for (int i = 1; i < l.LetterLines.Length; i++) d.Line(l.LetterLines[i]);
        d.Gap("6mm");
        d.Line(l.LetterLines[0], "text-align:right");
        d.Gap("10mm");
        d.Line(l.LetterBody[0], "font-weight:bold");
        d.Gap("5mm");
        for (int p = 1; p < l.LetterBody.Length; p++)
        {
            d.Lines_(Doc.Wrap(l.LetterBody[p], 74));
            d.Gap("5mm");
        }
        d.Line(l.LetterClosing);
        d.Gap("14mm");
        foreach (var s in l.LetterSender) d.Line(s);
        return new PageSpec(id, "letter", l.Code, "serif", false, d.ToHtml(l.Code, Serif, "12pt", "1.4"), d.Lines);
    }

    // ------------------------------------------------------------------ two-column article

    static readonly string[] ArticleEn =
    [
        "In the village of Eastbrook the bakery closed in 2009, and for years the only bread on sale came from a supermarket twenty minutes away by car. Last spring a group of neighbours decided that this was no longer acceptable and bought the empty building together.",
        "They spent the summer replacing the oven, repainting the shop front and learning how to bake. A retired engineer worked out the temperature curve of the old brick oven, while the school helped to design a new sign. By September the first loaves were sold on Saturday mornings.",
        "The project is not run for profit. Every member pays a small monthly share, and in return takes home a loaf each week. Surplus bread goes to the care home and to the football club, and the money left over pays for flour, wood and electricity.",
        "Visitors from other villages have started to arrive on market days, and two of them have already asked how to copy the idea. The volunteers say the most important lesson is patience: the first oven cracked, the second burned a hundred rolls, and the third finally worked.",
    ];

    static readonly string[] ArticleDe =
    [
        "Im Dorf Eichenau schloss die Bäckerei im Jahr 2009, und viele Jahre lang kam das einzige Brot aus einem Supermarkt, der mit dem Auto zwanzig Minuten entfernt lag. Im vergangenen Frühling beschloss eine Gruppe von Nachbarn, dass das nicht länger hinnehmbar sei, und kaufte das leere Haus gemeinsam.",
        "Den Sommer über ersetzten sie den Ofen, strichen die Fassade neu und lernten das Backen. Ein pensionierter Ingenieur berechnete die Temperaturkurve des alten Backsteinofens, während die Schule bei einem neuen Schild half. Im September wurden die ersten Laibe an einem Samstagmorgen verkauft.",
        "Das Projekt arbeitet nicht auf Gewinn. Jedes Mitglied zahlt einen kleinen Monatsbeitrag und nimmt dafür jede Woche ein Brot mit nach Hause. Überschüssiges Brot geht an das Pflegeheim und an den Fußballverein, und das übrige Geld bezahlt Mehl, Holz und Strom.",
        "Besucher aus anderen Dörfern kommen inzwischen an den Markttagen, und zwei von ihnen haben schon gefragt, wie man die Idee nachmachen kann. Die Freiwilligen sagen, die wichtigste Lehre sei Geduld: Der erste Ofen riss, der zweite verbrannte hundert Brötchen, und der dritte funktionierte endlich.",
    ];

    static PageSpec Article(string id, Lang l, string[] paragraphs)
    {
        var d = new Doc();
        string title = l.Code == "en" ? "The village bakery comes back" : "Die Dorfbäckerei kehrt zurück";
        string byline = l.Code == "en" ? "By Hannah Brooks, 9 March 2026" : "Von Hannah Brandt, 9. März 2026";
        d.Line(title, "font-size:22pt;font-weight:bold");
        d.Line(byline, "font-style:italic;margin-bottom:6mm");
        // Two columns of explicit lines: the first two paragraphs on the left, the others on the right.
        var left = new List<string>();
        var right = new List<string>();
        for (int p = 0; p < paragraphs.Length; p++)
        {
            var target = p < paragraphs.Length / 2 ? left : right;
            if (target.Count > 0) target.Add("");
            target.AddRange(Doc.Wrap(paragraphs[p], 47));
        }
        d.Columns([left, right], "10mm");
        return new PageSpec(id, "article", l.Code, "serif", false, d.ToHtml(l.Code, Serif, "10pt", "1.35"), d.Lines);
    }

    // ------------------------------------------------------------------ table

    static PageSpec Grid(string id, Lang l, ulong seed)
    {
        var rng = Rng.For(seed, id);
        var d = new Doc();
        var rows = new List<string[]>();
        string title;
        if (l.Code == "en")
        {
            title = "Quarterly sales by region (thousands of dollars)";
            rows.Add(["Region", "Q1", "Q2", "Q3", "Q4", "Total"]);
            foreach (var r in new[] { "North", "South", "East", "West", "Central", "Overseas", "Online", "Partners" })
            {
                int a = rng.Next(100, 999), b = rng.Next(100, 999), c = rng.Next(100, 999), e = rng.Next(100, 999);
                rows.Add([r, a.ToString(CultureInfo.InvariantCulture), b.ToString(CultureInfo.InvariantCulture), c.ToString(CultureInfo.InvariantCulture), e.ToString(CultureInfo.InvariantCulture), (a + b + c + e).ToString("N0", CultureInfo.InvariantCulture)]);
            }
        }
        else
        {
            title = "Månedligt forbrug i 2025";
            rows.Add(["Måned", "Strøm kWh", "Vand kubikmeter", "Varme kWh", "Pris kr."]);
            for (int m = 0; m < 12; m++)
                rows.Add([l.Months[m], rng.Next(900, 1900).ToString(CultureInfo.InvariantCulture), (rng.Next(80, 190) / 10.0).ToString("0.0", CultureInfo.InvariantCulture).Replace('.', ','),
                    rng.Next(500, 4500).ToString(CultureInfo.InvariantCulture), Money(rng.Next(120000, 450000) / 100m, l, false)]);
        }
        d.Line(title, "font-size:15pt;font-weight:bold;margin-bottom:5mm");
        int cols = rows[0].Length;
        var right = new bool[cols];
        for (int c = 1; c < cols; c++) right[c] = true;
        var widths = new string[cols];
        widths[0] = "38mm";
        for (int c = 1; c < cols; c++) widths[c] = $"{(174 - 38) / (cols - 1)}mm";
        d.Table(rows, right, widths, headerRow: true, tableStyle: "width:174mm;border:0.8pt solid #000",
            headerStyle: "font-weight:bold;background:#e6e6e6", border: "border:0.6pt solid #000;");
        d.Gap("6mm");
        d.Line(l.Code == "en" ? "Source: internal sales ledger, figures rounded." : "Kilde: eget regnskab, tallene er afrundet.", "font-size:9pt");
        return new PageSpec(id, "table", l.Code, "sans", false, d.ToHtml(l.Code, Sans, "10pt", "1.3"), d.Lines);
    }

    // ------------------------------------------------------------------ small print

    static readonly string[] SmallEn =
    [
        "1. Definitions. In these terms, the Supplier means Northgate Services Ltd, and the Customer means the person or company that orders goods or services from the Supplier.",
        "2. Payment. Invoices are payable within 30 days of the invoice date. Interest on overdue amounts is charged at 8 percent per year.",
        "3. Delivery. Delivery dates are estimates only, and the Supplier is not liable for delays caused by events outside its reasonable control.",
        "4. Complaints. The Customer must report visible defects in writing within 8 days of delivery to complaints@northgate.example.",
        "5. Liability. The Supplier's total liability is limited to the price paid for the goods or services in the twelve months before the claim.",
        "6. Data. Personal data is processed as described in the privacy notice at https://www.northgate.example/privacy and is kept for no longer than seven years.",
        "7. Law. These terms are governed by the law of England and Wales, and the courts of England have exclusive jurisdiction.",
    ];

    static readonly string[] SmallFr =
    [
        "1. Définitions. Dans les présentes conditions, le Fournisseur désigne la société Nordgate Services SARL et le Client désigne la personne ou l'entreprise qui commande des biens ou des services au Fournisseur.",
        "2. Paiement. Les factures sont payables dans un délai de 30 jours à compter de leur date d'émission. Les montants échus portent intérêt au taux de 8 % par an.",
        "3. Livraison. Les délais de livraison sont donnés à titre indicatif ; le Fournisseur n'est pas responsable des retards dus à des événements indépendants de sa volonté.",
        "4. Réclamations. Le Client doit signaler tout défaut apparent par écrit dans un délai de 8 jours après la livraison à reclamations@nordgate.example.",
        "5. Responsabilité. La responsabilité totale du Fournisseur est limitée au prix payé pour les biens ou services au cours des douze mois précédant la réclamation.",
        "6. Données. Les données personnelles sont traitées comme indiqué dans la notice de confidentialité et sont conservées pendant sept ans au plus.",
        "7. Droit applicable. Les présentes conditions sont régies par le droit français ; les tribunaux de Bordeaux sont seuls compétents en cas de litige.",
    ];

    static PageSpec SmallPrint(string id, Lang l, string[] paragraphs, string family)
    {
        var d = new Doc();
        d.Line(l.Code == "en" ? "General terms and conditions" : "Conditions générales de vente", "font-size:14pt;font-weight:bold");
        d.Gap("3mm");
        foreach (var p in paragraphs)
        {
            d.Lines_(Doc.Wrap(p, family == Sans ? 112 : 125));
            d.Gap("2mm");
        }
        // The page is two copies of the terms' text size on purpose: one block at 8 pt, as a contract's back page is.
        return new PageSpec(id, "smallprint", l.Code, family == Sans ? "sans" : "serif", false, d.ToHtml(l.Code, family, "8pt", "1.3"), d.Lines);
    }

    // ------------------------------------------------------------------ headings

    static readonly string[] HeadingEn =
    [
        "Revenue grew by eleven percent over the year, driven by new customers in the north and by the launch of the service plan in the spring.",
        "The new warehouse opened in June and cut the average delivery time from four days to two. Staff turnover fell for the second year in a row.",
        "We expect steady growth in 2026. The board will propose a dividend of two euros per share at the annual meeting on 21 April.",
        "Hire two more engineers, move the call centre to the new building, and finish the move to renewable electricity before the end of September.",
    ];

    static readonly string[] HeadingDa =
    [
        "Omsætningen steg med elleve procent i løbet af året, drevet af nye kunder i nord og af lanceringen af serviceabonnementet i foråret.",
        "Det nye lager åbnede i juni og halverede den gennemsnitlige leveringstid fra fire dage til to. Medarbejderomsætningen faldt for andet år i træk.",
        "Vi forventer stabil vækst i 2026. Bestyrelsen foreslår et udbytte på tre kroner pr. aktie på generalforsamlingen den 21. april.",
        "Ansæt to ingeniører mere, flyt kundecentret til den nye bygning og gennemfør overgangen til vedvarende el inden udgangen af september.",
    ];

    static PageSpec Headings(string id, Lang l, string h1, string h2a, string h2b, string h3, string[] body)
    {
        var d = new Doc();
        d.Line(h1, "font-size:36pt;font-weight:bold;margin-bottom:8mm");
        d.Line(h2a, "font-size:26pt;font-weight:bold;margin-bottom:3mm");
        foreach (var l2 in Doc.Wrap(body[0], 60)) d.Line(l2);
        foreach (var l2 in Doc.Wrap(body[1], 60)) d.Line(l2);
        d.Gap("8mm");
        d.Line(h2b, "font-size:26pt;font-weight:bold;margin-bottom:3mm");
        foreach (var l2 in Doc.Wrap(body[2], 60)) d.Line(l2);
        d.Gap("8mm");
        d.Line(h3, "font-size:18pt;font-weight:bold;margin-bottom:2mm");
        foreach (var l2 in Doc.Wrap(body[3], 60)) d.Line(l2);
        return new PageSpec(id, "headings", l.Code, "sans", false, d.ToHtml(l.Code, Sans, "13pt", "1.35"), d.Lines);
    }

    // ------------------------------------------------------------------ monospace log

    static PageSpec LogPage(string id, ulong seed)
    {
        var rng = Rng.For(seed, id);
        var d = new Doc();
        string[] levels = ["INFO ", "WARN ", "ERROR", "DEBUG"];
        string[] actions = ["order accepted for", "payment received from", "invoice sent to", "refund issued to", "login failed for", "address updated for"];
        string[] users = ["anna.hansen", "lars.jensen", "marie.dubois", "j.weiss", "info", "billing", "peter.olsen", "s.carter"];
        string[] hosts = ["example.com", "example.org", "mail.example.net", "firma.example.dk"];
        d.Line("Service log, node 07", "font-size:13pt;font-weight:bold;margin-bottom:4mm");
        for (int i = 0; i < 34; i++)
        {
            string line = $"2026-{rng.Next(1, 12):00}-{rng.Next(1, 28):00} {rng.Next(0, 24):00}:{rng.Next(0, 60):00}:{rng.Next(0, 60):00} {rng.Pick(levels)} " +
                          $"#{rng.Next(10000, 99999)} {rng.Pick(actions)} {rng.Pick(users)}@{rng.Pick(hosts)}";
            if (line.Length > 80) line = line[..80].TrimEnd();
            d.Line(line);
        }
        d.Gap("4mm");
        d.Line("total 34 events, 0 dropped, 1,284 bytes/s");
        return new PageSpec(id, "mono", "en", "mono", false, d.ToHtml("en", Mono, "9.5pt", "1.35"), d.Lines);
    }

    // ------------------------------------------------------------------ numbers, dates, amounts, emails, URLs

    static PageSpec Identifiers(string id, ulong seed)
    {
        var rng = Rng.For(seed, id);
        var d = new Doc();
        d.Line("Contact and payment details", "font-size:16pt;font-weight:bold;margin-bottom:5mm");
        string[] first = ["anna", "lars", "marie", "jens", "sofie", "peter", "hannah", "jonas"];
        string[] last = ["hansen", "weiss", "dubois", "carter", "olsen", "mueller", "lefevre", "nielsen"];
        string[] domains = ["example.com", "example.org", "example.dk", "example.de", "example.fr"];
        string[] paths = ["docs", "invoices", "reports", "archive", "files"];
        var langs = new[] { En, Da, De, Fr };
        for (int i = 0; i < 6; i++)
        {
            var l = langs[i % 4];
            d.Line($"Phone: +{rng.Next(30, 49)} {rng.Next(10, 99)} {rng.Next(10, 99)} {rng.Next(10, 99)} {rng.Next(10, 99)}");
            d.Line($"Email: {rng.Pick(first)}.{rng.Pick(last)}@{rng.Pick(domains)}");
            d.Line($"Web: https://www.{rng.Pick(domains)}/{rng.Pick(paths)}/{rng.Next(2020, 2027)}/report-{rng.Next(100, 9999)}.pdf?id={rng.Next(10000, 99999)}");
            d.Line($"Date: {rng.Next(2020, 2027)}-{rng.Next(1, 12):00}-{rng.Next(1, 28):00}   {rng.Next(1, 28):00}/{rng.Next(1, 12):00}/{rng.Next(2020, 2027)}   {Date(l, rng.Next(1, 28), rng.Next(1, 12), rng.Next(2020, 2027))}");
            d.Line($"Amount: {Money(rng.Next(100, 9999999) / 100m, l)}   Reference: {rng.Next(100000, 999999)}-{rng.Next(10, 99)}   VAT {l.Code.ToUpperInvariant()}{rng.Next(100000000, 999999999)}");
            d.Gap("3mm");
        }
        return new PageSpec(id, "identifiers", "en", "sans", false, d.ToHtml("en", Sans, "11pt", "1.4"), d.Lines);
    }
}
