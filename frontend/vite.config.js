import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard is served by FastAPI in production (same origin, so /api works directly).
// In dev, Vite proxies /api to the Python server so the frontend never needs to know
// which mode it is running in. VITE_STATIC=1 builds the serverless variant for GitHub Pages:
// relative asset paths (the site lives under /<repo>/) and JSON files instead of /api.
export default defineConfig({
  base: process.env.VITE_STATIC === "1" ? "./" : "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Recharts is most of the bundle and changes far less often than the app code.
    rollupOptions: { output: { manualChunks: { charts: ["recharts"] } } },
    chunkSizeWarningLimit: 600,
  },
});
