// velite.config.ts
import { defineConfig, defineCollection, s } from 'velite';
import rehypePrettyCode from 'rehype-pretty-code';

// 'posts' 라는 이름의 데이터 컬렉션을 정의합니다.
const posts = defineCollection({
  name: 'Post', // 컬렉션의 이름
  pattern: 'posts/**/*.md', // 어떤 파일을 대상으로 할지 (content/posts 폴더 아래 모든 .md 파일)
  schema: s.object({
    title: s.string(), // 제목 (문자열)
    date: s.coerce.date(), // 작성일 (날짜)
    description: s.string(), // 요약 (문자열)
    category: s.enum(['Dev', 'Essay', 'Info']), 
    // slug와 content는 파일 경로와 내용에서 자동으로 추출됩니다.
    slug: s.path().transform((p) => p.replace(/^posts\//, '')),
    // --- markdown() 스키마의 rehypePlugins 부분을 수정합니다. ---
    content: s.markdown({
      rehypePlugins: [
        [
          rehypePrettyCode,
          {
            theme: 'github-dark',
          },
        ],
      ],
    }),
  })
});

export default defineConfig({
  root: 'content', // 콘텐츠 파일이 있는 루트 폴더
  output: {
    data: '.velite', // 처리된 JSON 데이터가 생성될 폴더
    assets: 'public/static', // 정적 파일(이미지 등)이 복사될 폴더
    base: '/static/', // public 폴더에 복사된 정적 파일에 접근하기 위한 기본 경로
    clean: true // 빌드 시 이전에 생성된 파일을 삭제할지 여부
  },
  collections: { posts } // 위에서 정의한 posts 컬렉션을 등록
});
