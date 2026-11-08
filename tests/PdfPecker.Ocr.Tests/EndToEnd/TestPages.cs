using System.Collections.Concurrent;

namespace PdfPecker.Ocr.Tests.EndToEnd;

/// <summary>A page drawn by PdfPecker from HTML and rendered to pixels, as a scanner at that resolution would see it.</summary>
internal sealed record RenderedPage(byte[] Rgba, int Width, int Height, byte[] Png, byte[] Pdf)
{
    /// <summary>The page as gray pixels, from the green channel, which is the gray value of these black-on-white pages.</summary>
    public byte[] Gray()
    {
        var gray = new byte[Width * Height];
        for (int i = 0; i < gray.Length; i++)
        {
            gray[i] = Rgba[(i * 4) + 1];
        }

        return gray;
    }

    /// <summary>The page turned by 180 degrees, as RGBA.</summary>
    public byte[] UpsideDown()
    {
        int n = Width * Height;
        var turned = new byte[Rgba.Length];
        for (int i = 0; i < n; i++)
        {
            Rgba.AsSpan(i * 4, 4).CopyTo(turned.AsSpan((n - 1 - i) * 4));
        }

        return turned;
    }

    /// <summary>The page as gray pixels turned clockwise by 90 or 270 degrees; its width and height swap.</summary>
    public byte[] TurnedGray(int degrees)
    {
        var turned = new byte[Width * Height];
        for (int y = 0; y < Width; y++)
        {
            for (int x = 0; x < Height; x++)
            {
                // Turned clockwise, the top of the page goes to the right: pixel (x, y) comes from (y, H - 1 - x).
                int sx = degrees == 90 ? y : Width - 1 - y;
                int sy = degrees == 90 ? Height - 1 - x : x;
                turned[(y * Height) + x] = Rgba[(((sy * Width) + sx) * 4) + 1];
            }
        }

        return turned;
    }

    /// <summary>A scanned PDF of the page: the PNG as the page's only content, at 300 dpi, with no text.</summary>
    public byte[] ScannedPdf() => ImageToPdf.Convert([Png], null, new ImageToPdfOptions { FitPageToImage = true, Dpi = 300 });
}

/// <summary>
/// The synthetic pages of the end-to-end tests: short HTML documents in PdfPecker's bundled fonts, rendered at 300
/// dpi like a clean scan. Each is rendered once per test run.
/// </summary>
internal static class TestPages
{
    private static readonly ConcurrentDictionary<string, Lazy<RenderedPage>> Cache = new(StringComparer.Ordinal);

    public const string LatinText =
        "The quick brown fox jumps over the lazy dog.\n" +
        "Rødgrød med fløde, blåbærsyltetøj og æblekage.\n" +
        "Größere Straßen führen über die Brücke nach Köln.\n" +
        "Smørrebrød in Ålesund, Øresund and Düsseldorf.";

    public const string NumbersText =
        "Invoice no. 2026-0412 of 08.11.2026\n" +
        "Total: EUR 16,539.45 (VAT 25% included)\n" +
        "Phone +47 22 33 44 55, account 1234.56.78901\n" +
        "Order 7, 42, 365 and 1000 units at 0.75 each";

    public const string SmallText =
        "Small print: payment is due within 30 days of the invoice date.\n" +
        "Late payments carry interest of 8.5% a year.";

    public const string ColumnsText =
        "The first column starts here\nand runs down the left side\nof the page in four lines\nbefore the reader moves on.\n" +
        "The second column follows\non the right-hand side and\nends the short article with\na closing line of its own.";

    /// <summary>The table read row by row, a cell per line.</summary>
    public const string TableText =
        "Item\nQty\nPrice\nAmount\n" +
        "Consulting\n12\n950.00\n11,400.00\n" +
        "Travel\n1\n1,234.56\n1,234.56\n" +
        "Licence\n3\n199.00\n597.00";

    public static RenderedPage Latin => Get("latin", Document("150mm 62mm", "p{font:12pt/1.5 serif;margin:0}", Paragraphs(LatinText)));

    public static RenderedPage Numbers => Get("numbers", Document("150mm 62mm", "p{font:12pt/1.5 sans-serif;margin:0}", Paragraphs(NumbersText)));

    /// <summary>Two lines of 8 point text.</summary>
    public static RenderedPage Small => Get("small", Document("150mm 30mm", "p{font:8pt/1.4 sans-serif;margin:0}", Paragraphs(SmallText)));

    /// <summary>Two columns of four lines side by side, which must be read column by column.</summary>
    public static RenderedPage Columns => Get("columns", Document(
        "170mm 80mm",
        ".c{display:flex;gap:14mm}.c div{flex:1;font:11pt/1.5 serif}",
        "<div class=\"c\"><div>" + string.Join("<br>", ColumnsText.Split('\n')[..4]) + "</div><div>" + string.Join("<br>", ColumnsText.Split('\n')[4..]) + "</div></div>"));

    public static RenderedPage Table => Get("table", Document(
        "150mm 60mm",
        "table{border-collapse:collapse;font:11pt sans-serif}td,th{border:1px solid #666;padding:3px 10px;text-align:left}",
        "<table><tr><th>Item</th><th>Qty</th><th>Price</th><th>Amount</th></tr>" +
        "<tr><td>Consulting</td><td>12</td><td>950.00</td><td>11,400.00</td></tr>" +
        "<tr><td>Travel</td><td>1</td><td>1,234.56</td><td>1,234.56</td></tr>" +
        "<tr><td>Licence</td><td>3</td><td>199.00</td><td>597.00</td></tr></table>"));

    /// <summary>A whole A4 invoice: a heading, addresses, a table and totals, about 30 lines.</summary>
    public static RenderedPage Invoice => Get("invoice", Document(
        "A4",
        "body{font:11pt/1.4 sans-serif}h1{font-size:20pt}table{border-collapse:collapse;width:100%}td,th{border:1px solid #888;padding:4px 6px;text-align:left}",
        "<h1>Invoice 2026-0412</h1><p>Example Customer Ltd.<br>Storgata 12, 0155 Oslo<br>Norway</p>" +
        "<p>Date: 8 November 2026, due 8 December 2026</p>" +
        "<table><tr><th>Item</th><th>Qty</th><th>Price</th><th>Amount</th></tr>" +
        "<tr><td>Consulting, October</td><td>12</td><td>950.00</td><td>11,400.00</td></tr>" +
        "<tr><td>Travel expenses</td><td>1</td><td>1,234.56</td><td>1,234.56</td></tr>" +
        "<tr><td>Software licence</td><td>3</td><td>199.00</td><td>597.00</td></tr>" +
        "<tr><td>Support hours</td><td>7</td><td>120.00</td><td>840.00</td></tr></table>" +
        "<p>Subtotal 14,071.56<br>VAT 25% 3,517.89<br>Total NOK 17,589.45</p>" +
        "<p>Payment within 30 days to account 1234.56.78901.<br>Please quote the invoice number with your payment.<br>" +
        "Thank you for your order. Questions: invoices@example.com</p>"));

    /// <summary>A page, rendered once per test run.</summary>
    public static RenderedPage Get(string name, string html, float dpi = 300) => Cache.GetOrAdd(name, _ => new Lazy<RenderedPage>(() => Render(html, dpi))).Value;

    public static RenderedPage Render(string html, float dpi = 300)
    {
        byte[] pdf = HtmlToPdf.Convert(html);
        var image = PdfRenderer.Open(pdf).RenderPage(0, new ImageOptions { Dpi = dpi });
        return new RenderedPage(image.Rgba, image.Width, image.Height, image.ToPng(), pdf);
    }

    public static string Document(string size, string css, string body)
        => $"<!doctype html><html><head><meta charset=\"utf-8\"><style>@page{{size:{size};margin:8mm}}body{{margin:0}}{css}</style></head><body>{body}</body></html>";

    private static string Paragraphs(string lines) => string.Concat(lines.Split('\n').Select(l => "<p>" + System.Net.WebUtility.HtmlEncode(l) + "</p>"));
}
