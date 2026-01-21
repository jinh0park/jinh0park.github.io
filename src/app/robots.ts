import { MetadataRoute } from 'next'

export const dynamic = 'force-static'

const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://example.com'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: '/private/', // 크롤링을 원하지 않는 경로가 있다면 추가
    },
    sitemap: `${BASE_URL}/sitemap.xml`,
  }
}
