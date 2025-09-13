import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 1. 정적 사이트로 출력하도록 설정
  output: 'export',

  // 2. (GitHub Pages 배포 시 필요) 저장소 이름을 basePath로 설정
  // '/new_blog_velite' 부분을 실제 저장소 이름으로 변경해주세요.
  basePath: '/new_blog_velite', 

  // 3. 정적 사이트에서는 Next.js 이미지 최적화 기능을 비활성화
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
