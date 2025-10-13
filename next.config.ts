// next.config.ts
import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';
// const repositoryName = 'new_blog_velite'; // 본인 저장소 이름 / 루트 ghpage 쓸때는 주석처리

const nextConfig: NextConfig = {
  output: 'export',
  // basePath: isProd ?  `/${repositoryName}` : '',
  // assetPrefix: isProd ? `/${repositoryName}/` : '',
  basePath : '',
  assetPrefix :'',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;