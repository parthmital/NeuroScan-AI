import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import glsl from "vite-plugin-glsl";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
	server: {
		host: "::",
		port: 8080,
		hmr: {
			overlay: false,
		},
	},
	plugins: [
		react(),
		// Required for vtk.js: it internally imports .glsl shader files
		glsl(),
	],
	resolve: {
		alias: {
			"@": path.resolve(__dirname, "./src"),
		},
	},
	optimizeDeps: {
		// nifti-reader-js is a CommonJS package that
		// Vite's esbuild pre-bundler cannot process correctly sometimes.
		exclude: ["nifti-reader-js"],
	},
}));
