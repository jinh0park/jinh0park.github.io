import Link from "next/link";
import { posts } from "#velite";
import { notFound } from "next/navigation";
import { format } from "date-fns";

// params가 Promise를 포함하도록 타입을 정의합니다.
type PageProps = {
  params: Promise<{ category: string }>;
};

export async function generateStaticParams() {
  const categories = posts.map((post) => post.category);
  const uniqueCategories = [...new Set(categories)];
  return uniqueCategories.map((category) => ({
    category: category,
  }));
}

export default async function CategoryPage({ params }: PageProps) {
  // props로 받은 params를 await으로 해결(resolve)합니다.
  const { category: categoryFromParams } = await params;
  const category = decodeURIComponent(categoryFromParams);

  const categoryPosts = posts
    .filter((post) => post.category === category)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

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
