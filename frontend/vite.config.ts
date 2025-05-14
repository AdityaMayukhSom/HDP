import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/",
  server: {
    host: "0.0.0.0",
    allowedHosts: ["host.docker.internal", "prime-ubuntu"],
  },
  build: {
    outDir: "./dist",
    emptyOutDir: true,
  },
});
