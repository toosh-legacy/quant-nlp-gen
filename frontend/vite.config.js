import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard is served by FastAPI in production (same origin, so /api works directly).
// In dev, Vite proxies /api to the Python server so the frontend never needs to know
// which mode it is running in.
export default defineConfig({
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
