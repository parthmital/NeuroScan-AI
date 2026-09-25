import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import glsl from "vite-plugin-glsl";
import path from "path";

const API_TARGET = process.env.VITE_DEV_API_TARGET ?? "http://127.0.0.1:8000";

// Headers for `vite preview`; production hosting must send the same set.
const SECURITY_HEADERS = {
	"Content-Security-Policy": [
		"default-src 'self'",
		"script-src 'self'",
		"style-src 'self' 'unsafe-inline'",
		"img-src 'self' data: blob:",
		"worker-src 'self' blob:",
		"connect-src 'self'",
		"frame-ancestors 'none'",
		"base-uri 'self'",
		"form-action 'self'",
		"object-src 'none'",
	].join("; "),
	"X-Content-Type-Options": "nosniff",
	"X-Frame-Options": "DENY",
	"Referrer-Policy": "no-referrer",
	"Permissions-Policy": "camera=(), microphone=(), geolocation=()",
};

// The browser talks to one origin; /api is proxied to the backend so the session
// cookie stays first-party and no CORS is needed.
const proxy = { "/api": { target: API_TARGET, changeOrigin: false } };

// https://vitejs.dev/config/
export default defineConfig({
	server: {
		host: "localhost",
		port: 8080,
		strictPort: true,
		proxy,
		hmr: {
			overlay: false,
		},
	},
	preview: {
		host: "localhost",
		port: 8080,
		strictPort: true,
		proxy,
		headers: SECURITY_HEADERS,
	},
	plugins: [
		react(),
		// Required for vtk.js: it internally imports .glsl shader files
		glsl(),
	],
	resolve: {
		alias: {
			"@": path.resolve(import.meta.dirname, "src"),
		},
	},
	optimizeDeps: {
		// nifti-reader-js is a CommonJS package that
		// Vite's esbuild pre-bundler cannot process correctly sometimes.
		exclude: ["nifti-reader-js"],
	},
});
