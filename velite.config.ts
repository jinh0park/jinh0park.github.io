import { defineConfig, defineCollection, s } from 'velite';
import rehypePrettyCode, { type Options } from 'rehype-pretty-code';
import rehypeFigure from 'rehype-figure';

// 1. next.config.ts와 동일하게 프로덕션 환경 변수와 저장소 이름을 정의합니다.
// const isProd = process.env.NODE_ENV === 'production';
// const repositoryName = 'new_blog_velite';

const posts = defineCollection({
  name: 'Post',
  pattern: 'posts/**/*.md',
  schema: s.object({
    title: s.string(),
    date: s.coerce.date(),
    description: s.string(),
    category: s.enum(['dev', 'essay', 'law', 'info']),
    slug: s.path().transform((p) => p.replace(/^posts\//, '')),
    content: s.markdown({
      rehypePlugins: [
        [rehypePrettyCode, { theme: 'github-dark' } satisfies Options],
        // Velite의 이미지 처리 기능이 먼저 실행된 후, 이 플러그인이 실행됩니다.
        // js-ts 호환성 문제로 /src/types/index.d.ts 파일 생성 필요
        rehypeFigure,
      ],
    }),
  }),
});

export default defineConfig({
  root: 'content',
  output: {
    data: '.velite',
    assets: 'public/static',
    // 2. 프로덕션 빌드 시에만 이미지 경로(base)에 저장소 이름을 포함시킵니다.
    // base: isProd ? `/${repositoryName}/static/` : '/static/',
    base : '/static/',
    clean: true,
  },
  collections: { posts },
});