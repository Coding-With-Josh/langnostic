import type { Metadata } from "next";
import localFont from "next/font/local";
import { Geist_Mono, Google_Sans_Flex, Host_Grotesk } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import Menu from "@/components/effects/menu";
import "./globals.css";

/**
 * Galaxie Copernicus — the site's primary (serif) voice.
 *
 * Served from the WOFF2 derived from GalaxieCopernicus-Book.ttf in ./fonts
 * (fonttools: TTFont(src).save(dst, flavor="woff2")). The TTF is kept as the
 * source asset but nothing references it, so it is not bundled.
 *
 * weight is declared explicitly: the file's OS/2.usWeightClass is 500 while
 * fsSelection marks it REGULAR, so an inferred weight would be wrong. "400"
 * makes Book the default text weight; heavier Tailwind weights (600/700) are
 * synthesized by the browser until real weights are added.
 *
 * adjustFontFallback is set to a serif so Next generates a metric-matched
 * fallback face (size-adjust/ascent-override) from Copernicus' real metrics.
 * Leaving the default ("Arial") would match a sans and cause a reflow on swap.
 */
// const copernicus = localFont({
//   src: [
//     {
//       path: "./fonts/GalaxieCopernicus-Book.woff2",
//       weight: "400",
//       style: "normal",
//     },
//   ],
//   variable: "--font-copernicus",
//   display: "swap",
//   fallback: ["Georgia", "Times New Roman", "serif"],
//   adjustFontFallback: "Times New Roman",
// });

/**
 * Host Grotesk — UI/sans voice (`font-sans`). Variable font (300–800), so no
 * weight is specified. Downloaded and self-hosted at build time; the browser
 * never contacts Google. latin-ext is included so accented/IPA-adjacent data
 * doesn't tofu in UI copy.
 */
const hostGrotesk = Google_Sans_Flex({
  subsets: ["latin", "latin-ext"],
  variable: "--font-host-grotesk",
  display: "swap",
});

/** Monospace (code, phonetic notation) — unchanged from the scaffold. */
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const siteUrl = "https://langnostic.vercel.app";
const siteTitle =
  "Langnostic: does a small adapter fix Whisper-small on Yoruba?";
const siteDescription =
  "A study of how well Whisper-small transcribes Yoruba, what a small LoRA adapter changes, and what it costs on English. All numbers and failures.";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: siteTitle,
    template: "%s | Langnostic",
  },
  description: siteDescription,
  applicationName: "Langnostic",
  authors: [{ name: "Joshua Idele" }],
  alternates: { canonical: "/" },
  robots: { index: true, follow: true },
  keywords: [
    "whisper",
    "yoruba",
    "lora",
    "speech recognition",
    "low-resource languages",
    "word error rate",
  ],
  openGraph: {
    type: "website",
    url: "/",
    siteName: "Langnostic",
    locale: "en_US",
    title: siteTitle,
    description: siteDescription,
    images: [
      {
        url: "/og.png",
        width: 2000,
        height: 1000,
        alt: "Langnostic wordmark with the tagline: a research study of speech recognition on native languages.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: siteTitle,
    description: siteDescription,
    images: ["/og.png"],
  },
};

// schema.org Article for the report itself. Every field is a compile-time
// constant, and "<" is escaped so a future description can never terminate the
// script element early (a `</script>` breakout).
const articleSchema = {
  "@context": "https://schema.org",
  "@type": "Article",
  headline: siteTitle,
  author: { "@type": "Person", name: "Joshua Idele" },
  datePublished: "2026-11-15",
  image: `${siteUrl}/og.png`,
  url: siteUrl,
  description: siteDescription,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${hostGrotesk.variable} ${geistMono.variable} h-full antialiased`}
    >
      {/* Analytics is best-effort client telemetry: it must never gate render. */}
      <Analytics />
      <body className="min-h-full flex flex-col">
        {/* JSON-LD for the report. "<" is escaped so no field can ever close
            the script element early; everything else is inert script data. */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(articleSchema).replace(/</g, "\\u003c"),
          }}
        />
        {children}
        <Menu />
      </body>
    </html>
  );
}
