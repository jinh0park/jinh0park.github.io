import { posts } from "#velite";
import Link from "next/link";
import { notFound } from "next/navigation";
import { format } from "date-fns";

// Stack Overflow에서 찾은 방식을 적용합니다.
// params가 Promise를 포함하고 있음을 명시적으로 타이핑합니다.
type PageProps = {
  params: Promise<{ slug: string }>;
};

async function getPostFromParams(slug: string) {
  const post = posts.find((post) => post.slug === slug);
  return post;
}

export async function generateStaticParams() {
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function Page({ params }: PageProps) {
  // props로 받은 params를 await으로 해결(resolve)한 후 slug를 구조분해합니다.
  const { slug } = await params;
  const post = await getPostFromParams(slug);

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
        <Link href={`/categories/${post.category}`}>
          <span className="bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1 rounded-full !my-0 hover:bg-gray-300">
            {post.category}
          </span>
        </Link>
      </div>
      <hr className="!my-4" />
      <div dangerouslySetInnerHTML={{ __html: post.content }} />
    </article>
  );
}
