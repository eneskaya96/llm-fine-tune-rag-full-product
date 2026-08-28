import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Tailwind v4 needs no config file: the plugin plus one @import in index.css
// is the whole setup.
export default defineConfig({ plugins: [react(), tailwindcss()] });
