import type { MetadataRoute } from "next";

const siteUrl = "https://langnostic.vercel.app";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: `${siteUrl}/`,
      lastModified: "2026-11-15",
    },
  ];
}
