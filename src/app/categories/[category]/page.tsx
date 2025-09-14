import Link from "next/link";
import { posts } from "#velite";
import { notFound } from "next/navigation";
import { format } from "date-fns";
import type { Metadata } from "next";

// Page 컴포넌트를 위한 타입 정의 (params가 Promise임을 명시)
type PageProps = {
  params: Promise<{ category: string }>;
};

// generateMetadata 함수: 여기서는 params를 await할 필요가 없습니다.
export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { category: categoryFromParams } = await params;
  const category = decodeURIComponent(categoryFromParams);

  return {
    title: `Category: ${category} | My Velite Blog`,
    description: `'${category}' 카테고리에 대한 포스트 목록입니다.`,
    openGraph: {
      title: `Category: ${category}`,
      description: `'${category}' 카테고리에 대한 포스트 목록입니다.`,
      type: "website",
    },
  };
}

export async function generateStaticParams() {
  const categories = posts.map((post) => post.category);
  const uniqueCategories = [...new Set(categories)];
  return uniqueCategories.map((category) => ({
    category: category,
  }));
}

export default async function CategoryPage({ params }: PageProps) {
  // Page 컴포넌트에서는 params를 await으로 해결합니다.
  const { category: categoryFromParams } = await params;
  const category = decodeURIComponent(categoryFromParams);

  // 전체 카테고리 목록을 가져옵니다.
  const allCategories = [...new Set(posts.map((post) => post.category))];

  const categoryPosts = posts
    .filter((post) => post.category === category)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  if (categoryPosts.length === 0) {
    notFound();
  }

  return (
    <main>
      <div className="mb-8">
        {/* 카테고리 필터 UI를 다시 추가합니다. */}
        <div className="flex flex-wrap gap-2">
          <Link
            href="/"
            className="bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1 rounded-full hover:bg-gray-300"
          >
            All
          </Link>
          {allCategories.map((cat) => (
            <Link
              key={cat}
              href={`/categories/${cat}`}
              className={
                cat === category
                  ? "bg-gray-800 text-white text-sm font-medium px-3 py-1 rounded-full"
                  : "bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1 rounded-full hover:bg-gray-300"
              }
            >
              {cat}
            </Link>
          ))}
        </div>
      </div>

      <div className="space-y-6">
        {categoryPosts.map((post) => (
          <article key={post.slug}>
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-semibold">
                <Link href={`/blog/${post.slug}`} className="hover:underline">
                  {post.title}
                </Link>
              </h2>
              {/* span을 Link로 감싸줍니다. */}
              <Link href={`/categories/${post.category}`}>
                {/* --- 카테고리 뱃지를 추가합니다 --- */}
                <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                  {post.category}
                </span>
              </Link>
            </div>
            <p className="text-gray-500 text-sm mt-1">
              {format(new Date(post.date), "yyyy년 M월 d일")}
            </p>
            <p className="mt-2 text-gray-700">{post.description}</p>
          </article>
        ))}
      </div>
    </main>
  );
}
