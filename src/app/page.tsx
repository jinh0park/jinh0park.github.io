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
  const linkUrl = "https://picsum.photos/id/866/600/400.jpg";
  const imageUrl = "https://picsum.photos/id/866/600/400.jpg";

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">My Blog</h1>
      <div className="max-w-sm rounded-lg border border-gray-200 bg-white shadow-md">
        {/* 이미지 영역 */}
        <a href={linkUrl}>
          <img className="rounded-t-lg" src={imageUrl} />
        </a>

        {/* 텍스트 및 버튼 영역 */}
        <div className="p-5">
          <a href={linkUrl}>
            <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900">
              {"title"}
            </h5>
          </a>
          <p className="mb-3 font-normal text-gray-700">{"description"}</p>
          <a
            href={linkUrl}
            className="inline-flex items-center rounded-lg bg-blue-700 px-3 py-2 text-center text-sm font-medium text-white hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-300"
          >
            {"buttonText"}
            <svg
              className="ml-2 h-3.5 w-3.5"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 14 10"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M1 5h12m0 0L9 1m4 4L9 9"
              />
            </svg>
          </a>
        </div>
      </div>

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
