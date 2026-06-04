/** Alternate Vite config for testing alongside a running dev instance.
 *  Frontend → http://127.0.0.1:5174
 *  Backend  → http://127.0.0.1:8083  (run: uvicorn aiml_discovery.api:app --port 8083 --reload)
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["react-markdown"]
  },
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8083",
        changeOrigin: true
      }
    }
  }
});
