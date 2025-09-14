import { defineConfig, defineCollection, s } from 'velite';
import rehypePrettyCode from 'rehype-pretty-code';
import rehypeFigure from 'rehype-figure';

const posts = defineCollection({
  name: 'Post',
  pattern: 'posts/**/*.md',
  schema: s.object({
    title: s.string(),
    date: s.coerce.date(),
    description: s.string(),
    category: s.enum(['Dev', 'Essay', 'Info']),
    slug: s.path().transform((p) => p.replace(/^posts\//, '')),
    content: s.markdown({
      rehypePlugins: [
        [rehypePrettyCode as any, { theme: 'github-dark' }],
        // Velite의 이미지 처리 기능이 먼저 실행된 후, 이 플러그인이 실행됩니다.
        // js-ts 호환성 문제로 /src/types/index.d.ts 파일 생성 필요
        rehypeFigure as any,
      ],
    }),
  }),
});

export default defineConfig({
  root: 'content',
  output: {
    data: '.velite',
    assets: 'public/static',
    base: '/static/',
    clean: true,
  },
  collections: { posts },
});