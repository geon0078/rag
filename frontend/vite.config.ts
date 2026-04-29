import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// 운영웹통합명세서 §10.1 — Vite + React + TS 진입 설정.
// Path alias `@/...` → `src/...` (shadcn/ui 표준 컨벤션과 일치).
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
