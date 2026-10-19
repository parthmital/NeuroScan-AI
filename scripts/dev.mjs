// `npm run dev`: set up everything on first run, then start the backend and frontend together.
// Later runs skip each setup step whose inputs have not changed. Ctrl+C stops both services.
// Uses only Node built-ins, so it runs before any dependency is installed.

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { createInterface } from "node:readline";
import { styleText } from "node:util";

const ROOT = path.resolve(import.meta.dirname, "..");
const BACKEND = path.join(ROOT, "Backend");
const FRONTEND = path.join(ROOT, "Frontend");
const VENV = path.join(ROOT, ".venv");
const IS_WIN = process.platform === "win32";
const VENV_PYTHON = path.join(
	VENV,
	IS_WIN ? "Scripts/python.exe" : "bin/python",
);
// The development set includes requirements.txt plus tooling such as black, pytest and ruff.
const REQUIREMENTS = path.join(BACKEND, "requirements-dev.txt");
const RUNTIME_REQUIREMENTS = path.join(BACKEND, "requirements.txt");
const PYTHON_STAMP = path.join(VENV, ".requirements.sha256");
const LOCKFILE = path.join(FRONTEND, "package-lock.json");
const NODE_STAMP = path.join(FRONTEND, "node_modules", ".package-lock.sha256");
const ROOT_LOCKFILE = path.join(ROOT, "package-lock.json");
const ROOT_NODE_STAMP = path.join(ROOT, "node_modules", ".package-lock.sha256");
const ENV_FILE = path.join(ROOT, ".env");
const FRONTEND_URL = "http://localhost:8080";
const BACKEND_HEALTH_URL = "http://127.0.0.1:8000/api/health";
const READY_TIMEOUT_MS = 120_000;
const MIN_PYTHON = [3, 11];
const MIN_NODE_MAJOR = 22;
const SHUTDOWN_GRACE_MS = 5000;
// Model weights cannot be fetched automatically (see README); warn instead of failing.
// Default paths match Backend/neuroscan/config.py and are relative to Backend/.
const WEIGHTS = {
	DETECTION_WEIGHTS: [
		"../Materials/Notebooks/binary-detection/weights/deploy/EfficientNetB0.pth",
		"uploads stay disabled",
	],
	CLASSIFICATION_WEIGHTS: [
		"../Materials/Notebooks/classification/weights/deploy/EfficientNetB0.pth",
		"uploads stay disabled",
	],
	SEGMENTATION_WEIGHTS: [
		"../Materials/Notebooks/segmentation/weights/deploy/VNet.pth",
		"scans are processed without segmentation",
	],
};

// Keep package-manager caches inside the repository (Git-ignored) unless the user set their own.
process.env.PIP_CACHE_DIR ??= path.join(ROOT, ".cache", "pip");
process.env.npm_config_cache ??= path.join(ROOT, ".cache", "npm");

const log = (message) => console.log(styleText("gray", message));
const warn = (message) => console.warn(styleText("yellow", message));

function fail(message) {
	console.error(styleText("red", message));
	process.exit(1);
}

function run(command, args, options = {}) {
	const result = spawnSync(command, args, { stdio: "inherit", ...options });
	if (result.status !== 0) {
		fail(`Command failed: ${command} ${args.join(" ")}`);
	}
}

// npm itself, launched through the running Node so no shell or .cmd shim is involved.
function npmCommand(args) {
	const cli = process.env.npm_execpath;
	if (cli?.endsWith(".js")) {
		return [process.execPath, [cli, ...args], {}];
	}
	return ["npm", args, { shell: IS_WIN }];
}

const sha256 = (...files) =>
	files
		.reduce(
			(hash, file) => hash.update(readFileSync(file)),
			createHash("sha256"),
		)
		.digest("hex");
const isCurrent = (stamp, hash) =>
	existsSync(stamp) && readFileSync(stamp, "utf8").trim() === hash;

function checkNode() {
	const major = Number(process.versions.node.split(".")[0]);
	if (major < MIN_NODE_MAJOR) {
		fail(
			`Node.js ${MIN_NODE_MAJOR} or newer is required; found ${process.versions.node}.`,
		);
	}
}

// Prefer the version pinned in .python-version; accept any newer Python 3 as a fallback.
function findPython() {
	const pinned = readFileSync(
		path.join(ROOT, ".python-version"),
		"utf8",
	).trim();
	const candidates = IS_WIN
		? [
				["py", [`-${pinned}`]],
				["python", []],
				["python3", []],
				["py", []],
			]
		: [
				[`python${pinned}`, []],
				["python3", []],
				["python", []],
			];
	let fallback;
	for (const [command, args] of candidates) {
		const result = spawnSync(
			command,
			[...args, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
			{ encoding: "utf8" },
		);
		if (result.status !== 0) continue;
		const version = result.stdout.trim();
		if (version === pinned) return { command, args, version };
		const [major, minor] = version.split(".").map(Number);
		if (!fallback && major === MIN_PYTHON[0] && minor >= MIN_PYTHON[1]) {
			fallback = { command, args, version };
		}
	}
	if (fallback) {
		warn(`Python ${pinned} not found; using Python ${fallback.version}.`);
		return fallback;
	}
	fail(
		`Python ${pinned} is required. Install it from https://www.python.org/downloads/`,
	);
}

function setUpPython() {
	if (!existsSync(VENV_PYTHON)) {
		const python = findPython();
		log(
			`Creating the virtual environment in .venv with Python ${python.version}...`,
		);
		run(python.command, [...python.args, "-m", "venv", VENV]);
	}
	const hash = sha256(REQUIREMENTS, RUNTIME_REQUIREMENTS);
	if (isCurrent(PYTHON_STAMP, hash)) return;
	log("Installing Python dependencies from Backend/requirements-dev.txt...");
	run(VENV_PYTHON, [
		"-m",
		"pip",
		"install",
		"--disable-pip-version-check",
		"-r",
		REQUIREMENTS,
	]);
	writeFileSync(PYTHON_STAMP, `${hash}\n`);
}

function setUpFrontend() {
	const hash = sha256(LOCKFILE);
	if (isCurrent(NODE_STAMP, hash)) return;
	log("Installing frontend dependencies from Frontend/package-lock.json...");
	const [command, args, options] = npmCommand(["ci"]);
	run(command, args, { cwd: FRONTEND, ...options });
	writeFileSync(NODE_STAMP, `${hash}\n`);
}

// Root tooling (the pinned Prettier), so `npx prettier` from the root matches CI.
function setUpRootTools() {
	const hash = sha256(ROOT_LOCKFILE);
	if (isCurrent(ROOT_NODE_STAMP, hash)) return;
	log("Installing root tooling from package-lock.json...");
	const [command, args, options] = npmCommand(["ci"]);
	run(command, args, { cwd: ROOT, ...options });
	writeFileSync(ROOT_NODE_STAMP, `${hash}\n`);
}

// A development .env with fresh random secrets on first run (Git-ignored, never overwritten).
function setUpEnvFile() {
	if (existsSync(ENV_FILE)) return;
	log("Creating a development .env with new random secrets...");
	const result = spawnSync(
		VENV_PYTHON,
		[
			"-c",
			"import secrets; from cryptography.fernet import Fernet; " +
				"print(secrets.token_urlsafe(48)); print(Fernet.generate_key().decode())",
		],
		{ encoding: "utf8" },
	);
	const [jwtSecret, encryptionKey] = result.stdout?.trim().split(/\r?\n/) ?? [];
	if (result.status !== 0 || !jwtSecret || !encryptionKey) {
		fail(`Could not generate secrets:\n${result.stderr}`);
	}
	const lines = [
		"ENVIRONMENT=development",
		`JWT_SECRET=${jwtSecret}`,
		`DATA_ENCRYPTION_KEY=${encryptionKey}`,
		"COOKIE_SECURE=true",
		"HOST=127.0.0.1",
		"PORT=8000",
	];
	writeFileSync(ENV_FILE, `${lines.join("\n")}\n`);
}

function checkWeights() {
	const envText = readFileSync(ENV_FILE, "utf8");
	for (const [name, [defaultPath, effect]] of Object.entries(WEIGHTS)) {
		const overridden =
			process.env[name] || new RegExp(`^${name}=\\S`, "m").test(envText);
		const weightsPath = path.join(BACKEND, defaultPath);
		if (!overridden && !existsSync(weightsPath)) {
			warn(
				`Model weights missing: ${path.relative(ROOT, weightsPath)}; ${effect} until it exists.`,
			);
		}
	}
}

const BACKEND_ARGS = [
	"main:app",
	"--reload",
	"--host",
	"127.0.0.1",
	"--port",
	"8000",
	// Readable, aligned log lines from the reloader too (the app sets the same format itself).
	"--log-config",
	"log-config.dev.json",
];

// Windows: each service in its own PowerShell window for dedicated logs, as start.ps1 did.
// -NoExit keeps a window open after a crash so its log stays readable.
const windowPids = [];

// A PowerShell single-quoted string literal.
const psQuote = (text) => `'${text.replaceAll("'", "''")}'`;

function openWindow(title, cwd, commands) {
	const script = [
		`$Host.UI.RawUI.WindowTitle = ${psQuote(title)}`,
		"chcp 65001 | Out-Null",
		"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
		`Set-Location -LiteralPath ${psQuote(cwd)}`,
		...commands,
	].join("\n");
	// Encoded so paths with spaces or quotes survive the command line intact.
	const encoded = Buffer.from(script, "utf16le").toString("base64");
	// Start-Process opens a new console window and reports its PID for closing it later.
	const result = spawnSync(
		"powershell.exe",
		[
			"-NoProfile",
			"-Command",
			"(Start-Process powershell.exe -PassThru -ArgumentList " +
				`'-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-NoExit','-EncodedCommand','${encoded}').Id`,
		],
		{ encoding: "utf8" },
	);
	const pid = Number(result.stdout?.trim());
	if (result.status !== 0 || !pid) {
		fail(`Could not open the ${title} window:\n${result.stderr}`);
	}
	windowPids.push(pid);
}

function isRunning(pid) {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

// Closes both windows and everything running in them. Synchronous, so it also works from
// the "exit" event and in the few seconds Windows allows after the launcher window closes.
function closeWindows() {
	for (const pid of windowPids.splice(0)) {
		spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], {
			stdio: "ignore",
		});
	}
}

function startInWindows() {
	openWindow("NeuroScan-Backend", BACKEND, [
		"$env:PYTHONIOENCODING = 'utf-8'",
		// Localhost only; the frontend dev server proxies /api to this port.
		`& ${psQuote(VENV_PYTHON)} -m uvicorn ${BACKEND_ARGS.join(" ")}`,
	]);
	openWindow("NeuroScan-Frontend", FRONTEND, [
		"$env:NODE_ENV = 'development'",
		"npm run dev",
	]);
	process.on("exit", closeWindows);
	// SIGHUP: the launcher window was closed. SIGBREAK: Ctrl+Break.
	for (const signal of ["SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"]) {
		process.on(signal, () => {
			stopping = true;
			log("\nStopping all services...");
			closeWindows();
			log("All services stopped.");
			process.exit(0);
		});
	}
	// Stay alive while any service window is open; exit once the user has closed both.
	const watch = setInterval(() => {
		if (!windowPids.some(isRunning)) {
			clearInterval(watch);
			stopping = true;
			log("Both service windows were closed.");
		}
	}, 2000);
}

// macOS and Linux: both services in this terminal, each log line prefixed with its name.
const children = [];
let stopping = false;

function startService(name, color, command, args, options) {
	const child = spawn(command, args, {
		stdio: ["ignore", "pipe", "pipe"],
		// Its own process group so the whole tree can be signalled at once.
		detached: true,
		...options,
	});
	const prefix = styleText(color, `[${name}]`);
	for (const stream of [child.stdout, child.stderr]) {
		createInterface({ input: stream }).on("line", (line) =>
			console.log(`${prefix} ${line}`),
		);
	}
	child.exited = new Promise((resolve) => child.on("exit", resolve));
	child.on("error", (error) => {
		console.error(`${prefix} ${error.message}`);
		stopAll(1);
	});
	child.on("exit", (code, signal) => {
		if (stopping) return;
		console.error(
			`${prefix} exited (${signal ?? `code ${code}`}); stopping the other service.`,
		);
		stopAll(code || 1);
	});
	children.push(child);
}

function signalTree(child, signal) {
	if (child.exitCode !== null || child.signalCode !== null) return;
	try {
		process.kill(-child.pid, signal);
	} catch {
		// Already gone.
	}
}

// Ask each service to stop, then force-kill whatever is still running after a grace period.
async function stopAll(code) {
	if (stopping) return;
	stopping = true;
	log("\nStopping all services...");
	children.forEach((child) => signalTree(child, "SIGINT"));
	const grace = new Promise((resolve) =>
		setTimeout(resolve, SHUTDOWN_GRACE_MS),
	);
	await Promise.race([
		Promise.all(children.map((child) => child.exited)),
		grace,
	]);
	children.forEach((child) => signalTree(child, "SIGKILL"));
	log("All services stopped.");
	process.exit(code);
}

function startInTerminal() {
	for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
		process.on(signal, () => stopAll(0));
	}
	// Output is piped for the name prefixes, so ask both services for colour explicitly.
	const tty = process.stdout.isTTY;
	startService(
		"backend",
		"magenta",
		VENV_PYTHON,
		["-m", "uvicorn", ...BACKEND_ARGS],
		{
			cwd: BACKEND,
			env: {
				...process.env,
				PYTHONIOENCODING: "utf-8",
				PYTHONUNBUFFERED: "1",
				...(tty ? { FORCE_COLOR: "1" } : {}),
			},
		},
	);
	const [npm, npmArgs, npmOptions] = npmCommand(["run", "dev"]);
	startService("frontend", "blue", npm, npmArgs, {
		cwd: FRONTEND,
		env: {
			...process.env,
			NODE_ENV: "development",
			...(tty ? { FORCE_COLOR: "1" } : {}),
		},
		...npmOptions,
	});
}

function openBrowser(url) {
	const [command, args] = IS_WIN
		? ["cmd", ["/c", "start", "", url]]
		: [process.platform === "darwin" ? "open" : "xdg-open", [url]];
	const opener = spawn(command, args, {
		stdio: "ignore",
		detached: true,
		windowsHide: true,
	});
	opener.on("error", () => log(`Open ${url} in your browser.`));
	opener.unref();
}

// Open the browser only once both services answer, so it does not load a half-started app.
async function openWhenReady() {
	const deadline = Date.now() + READY_TIMEOUT_MS;
	while (!stopping && Date.now() < deadline) {
		try {
			for (const url of [BACKEND_HEALTH_URL, FRONTEND_URL]) {
				await fetch(url, { signal: AbortSignal.timeout(2000) });
			}
			openBrowser(FRONTEND_URL);
			console.log(styleText("green", `\nReady: ${FRONTEND_URL}`));
			console.log(
				styleText(
					"cyan",
					"Backend API: http://127.0.0.1:8000 (proxied at /api)",
				),
			);
			if (IS_WIN) console.log("Logs are in the service windows.");
			console.log("Press Ctrl+C to stop both services.\n");
			return;
		} catch {
			await new Promise((resolve) => setTimeout(resolve, 500));
		}
	}
	if (!stopping) {
		warn(
			`Services did not respond within ${READY_TIMEOUT_MS / 1000} seconds; check their logs.`,
		);
	}
}

checkNode();
setUpPython();
setUpFrontend();
setUpRootTools();
setUpEnvFile();
checkWeights();

if (IS_WIN) {
	startInWindows();
} else {
	startInTerminal();
}
openWhenReady();
