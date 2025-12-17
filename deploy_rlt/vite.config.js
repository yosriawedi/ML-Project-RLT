import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/upload-and-train": "http://127.0.0.1:8000",
      "/predict-row": "http://127.0.0.1:8000",
      "/predict-file": "http://127.0.0.1:8000",
      "/benchmark-results": "http://127.0.0.1:8000",
      "/status": "http://127.0.0.1:8000",
      "/eda": "http://127.0.0.1:8000",
      "/xai": "http://127.0.0.1:8000",
      "/dso2": "http://127.0.0.1:8000"
    }
  }
});
