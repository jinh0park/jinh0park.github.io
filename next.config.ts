import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';

const nextConfig: NextConfig = {
  output: 'export',
  
  // GitHub Pages 배포를 위한 basePath 설정
  basePath: isProd ? '/new_blog_velite' : '',
  
  // assetPrefix를 추가하여 모든 에셋 경로에 basePath를 적용합니다.
  assetPrefix: isProd ? '/new_blog_velite/' : '',

  images: {
    unoptimized: true,
  },
};

export default nextConfig;