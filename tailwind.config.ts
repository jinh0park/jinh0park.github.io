import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // --- 이 부분을 추가합니다 ---
      typography: {
        DEFAULT: {
          css: {
            'pre code': {
              // prose가 코드 블록에 배경색이나 색상을 적용하지 않도록 초기화합니다.
              backgroundColor: 'transparent',
              color: 'inherit',
            },
            'code::before': false, // prose가 추가하는 따옴표 제거
            'code::after': false,  // prose가 추가하는 따옴표 제거
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
