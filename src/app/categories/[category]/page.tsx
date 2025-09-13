import Link from "next/link";
import { posts } from "#velite";
import { notFound } from "next/navigation";
import { format } from "date-fns";

interface PageProps {
  params: {
    // URL의 [category] 부분이 여기에 들어옵니다.
    category: string;
  };
}

// 빌드 시점에 Next.js가 어떤 카테고리 페이지들을 미리 만들어야 할지 알려줍니다.
export async function generateStaticParams(): Promise<PageProps["params"][]> {
  // 1. 모든 포스트에서 카테고리만 추출합니다.
  const categories = posts.map((post) => post.category);
  // 2. 중복된 카테고리를 제거하여 유니크한 목록을 만듭니다.
  const uniqueCategories = [...new Set(categories)];

  // 3. Next.js가 요구하는 형식으로 변환하여 반환합니다. ex: [{ category: 'Dev' }, { category: 'Essay' }]
  return uniqueCategories.map((category) => ({
    category: category,
  }));
}

export default async function CategoryPage({ params }: PageProps) {
  // URL 경로에 있는 카테고리 이름입니다. (예: 'Dev')
  // 한글 등 비영어권 문자가 URL에 포함될 경우를 대비해 decodeURIComponent를 사용합니다.
  const category = decodeURIComponent(params.category);

  // 해당 카테고리에 속한 포스트들만 필터링합니다.
  const categoryPosts = posts
    .filter((post) => post.category === category)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  // 해당 카테고리에 글이 없으면 404 페이지를 보여줍니다.
  if (categoryPosts.length === 0) {
    notFound();
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">
        Category: <span className="text-blue-600">{category}</span>
      </h1>
      <div className="space-y-6">
        {categoryPosts.map((post) => (
          <article key={post.slug}>
            <h2 className="text-2xl font-semibold">
              <Link href={`/blog/${post.slug}`} className="hover:underline">
                {post.title}
              </Link>
            </h2>
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
