import { MetadataRoute } from 'next'
import { posts } from '#velite'

export const dynamic = 'force-static'

// 실제 배포할 도메인으로 변경해주세요. 환경 변수로 관리하는 것이 좋습니다.
const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://example.com'

export default function sitemap(): MetadataRoute.Sitemap {
  const postsUrls = posts.map((post) => ({
    url: `${BASE_URL}/blog/${post.slug}`,
    lastModified: new Date(post.date),
    changeFrequency: 'weekly' as const,
    priority: 0.8,
  }))

  // 중복 제거된 카테고리 목록 추출
  const categories = Array.from(new Set(posts.map((post) => post.category)))
  
  const categoryUrls = categories.map((category) => ({
    url: `${BASE_URL}/categories/${category}`,
    lastModified: new Date(),
    changeFrequency: 'weekly' as const,
    priority: 0.6,
  }))

  return [
    {
      url: BASE_URL,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${BASE_URL}/blog`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.9,
    },
    ...categoryUrls,
    ...postsUrls,
  ]
}
