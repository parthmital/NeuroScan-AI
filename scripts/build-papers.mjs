// `npm run papers [-- <filter>]`: compile the LaTeX paper projects in Materials/Research Papers
// with Tectonic. Each project builds main.tex (and supplementary.tex when present), or every
// .tex file when it has no main.tex.
// <filter> keeps only project folders whose name contains it (case-insensitive).
// Uses only Node built-ins.

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { styleText } from "node:util";

const ROOT = path.resolve(import.meta.dirname, "..");
const PAPERS = path.join(ROOT, "Materials", "Research Papers");
const IS_WIN = process.platform === "win32";
// Materials/tectonic(.exe) is Git-ignored; fall back to a tectonic on PATH.
const LOCAL_TECTONIC = path.join(
	ROOT,
	"Materials",
	IS_WIN ? "tectonic.exe" : "tectonic",
);
const TECTONIC =
	process.env.TECTONIC ??
	(existsSync(LOCAL_TECTONIC) ? LOCAL_TECTONIC : "tectonic");

// Keep Tectonic's package and font cache inside the repository (Git-ignored).
process.env.TECTONIC_CACHE_DIR ??= path.join(ROOT, ".cache", "tectonic");

// The papers use only TeX fonts from the bundle, but Tectonic's font lookup still loads a
// Fontconfig configuration and prints "Cannot load default config file" when the system
// has none (Windows). Point it at an empty one inside the repository instead.
if (!process.env.FONTCONFIG_FILE) {
	const fontconfig = path.join(ROOT, ".cache", "fontconfig", "fonts.conf");
	if (!existsSync(fontconfig)) {
		mkdirSync(path.dirname(fontconfig), { recursive: true });
		writeFileSync(
			fontconfig,
			'<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig></fontconfig>\n',
		);
	}
	process.env.FONTCONFIG_FILE = fontconfig;
}

const filter = (process.argv[2] ?? "").toLowerCase();
const projects = readdirSync(PAPERS, { withFileTypes: true })
	.filter(
		(entry) => entry.isDirectory() && entry.name.toLowerCase().includes(filter),
	)
	.map((entry) => entry.name);

if (projects.length === 0) {
	console.warn(styleText("yellow", `No project folder matches "${filter}".`));
	process.exit(0);
}

const failed = [];
for (const project of projects) {
	const dir = path.join(PAPERS, project);
	const texFiles = readdirSync(dir).filter((file) => file.endsWith(".tex"));
	// A paper with a separate supplementary.tex cross-references it through xr-hyper, which
	// reads the other document's .aux: keep the intermediates (Git-ignored) and compile
	// main, supplementary, then main again so both sides resolve.
	const linked =
		texFiles.includes("main.tex") && texFiles.includes("supplementary.tex");
	const targets = linked
		? ["main.tex", "supplementary.tex", "main.tex"]
		: texFiles.includes("main.tex")
			? ["main.tex"]
			: texFiles.filter((file) => file !== "preamble.tex");
	const args = linked ? ["--keep-intermediates"] : [];
	for (const tex of targets) {
		console.log(styleText("cyan", `\n${project}: ${tex}`));
		const result = spawnSync(TECTONIC, [...args, tex], {
			cwd: dir,
			stdio: "inherit",
		});
		if (result.error?.code === "ENOENT") {
			console.error(
				styleText(
					"red",
					`Tectonic not found. Place it at ${LOCAL_TECTONIC} or on PATH (https://tectonic-typesetting.github.io/).`,
				),
			);
			process.exit(1);
		}
		if (result.status !== 0) failed.push(`${project}/${tex}`);
	}
}

if (failed.length > 0) {
	console.error(styleText("red", `\nFailed: ${failed.join(", ")}`));
	process.exit(1);
}
console.log(styleText("green", "\nAll papers compiled."));
