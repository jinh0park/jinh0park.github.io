// src/app/blog/[slug]/page.tsx
import { posts } from "#velite";
import { notFound } from "next/navigation";
import { format } from "date-fns";
import Link from "next/link";

interface PageProps {
  params: {
    slug: string;
  };
}

// 해당 slug를 가진 포스트 데이터를 찾는 함수
async function getPostFromParams(slug: string) {
  const post = posts.find((post) => post.slug === slug);
  return post;
}

// 빌드 시점에 정적 페이지를 미리 생성하기 위한 설정
export async function generateStaticParams(): Promise<PageProps["params"][]> {
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function Page({ params }: PageProps) {
  const { slug } = await params;
  const post = await getPostFromParams(slug);

  // 포스트가 없으면 404 페이지를 보여줌
  if (!post) {
    notFound();
  }

  return (
    <article className="prose dark:prose-invert mx-auto py-8">
      <h1 className="text-4xl font-bold">{post.title}</h1>
      <div className="flex items-center gap-4 mt-2 mb-2">
        <p className="text-gray-500 !my-0">
          {format(new Date(post.date), "yyyy년 M월 d일")}
        </p>
        {/* --- 카테고리 뱃지를 추가합니다 --- */}
        <Link href={`/categories/${post.category}`}>
          <span className="bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1 rounded-full !my-0">
            {post.category}
          </span>
        </Link>
      </div>
      <hr className="!my-4" />
      <div dangerouslySetInnerHTML={{ __html: post.content }} />
    </article>
  );
}
