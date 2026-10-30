import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    include: ["mc-react-header"],
    // Required to run wasm modules in dev mode for vite.
    exclude: ["@spglib/moyo-wasm"],
  },
  server: {
    // In dev, forward API calls to the FastAPI backend (uvicorn default port).
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
