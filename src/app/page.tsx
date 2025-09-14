// src/app/page.tsx
import Link from "next/link";
import { posts } from "#velite"; // <--- 이렇게 수정!
import { format } from "date-fns";

export default function Home() {
  const sortedPosts = posts.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );

  // 모든 포스트에서 유니크한 카테고리 목록을 추출합니다.
  const allCategories = [...new Set(posts.map((post) => post.category))];

  return (
    <main className="container mx-auto">
      <div className="flex flex-wrap gap-2 mb-8">
        <Link
          href="/"
          className="bg-gray-800 text-white text-sm font-medium px-3 py-1 rounded-full"
        >
          All
        </Link>
        {allCategories.map((category) => (
          <Link
            key={category}
            href={`/categories/${category}`}
            className="bg-gray-200 text-gray-800 text-sm font-medium px-3 py-1 rounded-full hover:bg-gray-300"
          >
            {category}
          </Link>
        ))}
      </div>

      <div className="space-y-6">
        {sortedPosts.map((post) => (
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
