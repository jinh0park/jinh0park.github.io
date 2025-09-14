import { posts } from "#velite";
import Link from "next/link";
import { notFound } from "next/navigation";
import { format } from "date-fns";
import type { Metadata } from "next"; // --- 1. Metadata 타입을 가져옵니다. ---

type PageProps = {
  params: Promise<{ slug: string }>;
};

// --- 2. generateMetadata 함수를 추가합니다. ---
export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const post = posts.find((post) => post.slug === slug);

  if (!post) {
    return {
      title: "Post Not Found",
    };
  }

  return {
    title: `${post.title} | My Velite Blog`,
    description: post.description,
    openGraph: {
      title: post.title,
      description: post.description,
      type: "article",
      publishedTime: new Date(post.date).toISOString(),
    },
  };
}

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
