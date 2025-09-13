// src/components/Header.tsx
import Link from "next/link";

export const Header = () => {
  return (
    <header className="bg-gray-100 dark:bg-gray-900 border-b">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link href="/" className="text-xl font-bold hover:text-blue-600">
          My Velite Blog
        </Link>
        <nav>
          {/* 나중에 GitHub 링크나 다른 메뉴를 여기에 추가할 수 있습니다. */}
          <a
            href="https://github.com/jinh0park" // 본인 GitHub ID로 변경하세요
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 dark:text-gray-300 hover:text-black dark:hover:text-white"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
};
