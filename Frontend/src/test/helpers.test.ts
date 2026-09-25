import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import {
	DEFAULT_OVERLAY_COLORS,
	useOverlayColors,
} from "@/hooks/use-overlay-colors";
import { isHexColour, parseHex } from "@/lib/color";
import { passwordProblem } from "@/lib/password";
import { matchesSearch, resultLabel } from "@/lib/scan-format";
import type { MRIScan } from "@/lib/types";
import { initials } from "@/lib/user-format";

const scan = (overrides: Partial<MRIScan> = {}): MRIScan => ({
	id: "scan-1",
	patientId: "PT-2026-AB12",
	patientName: "Jane Doe",
	scanDate: "2026-09-26",
	modalities: ["FLAIR"],
	status: "completed",
	pipelineStep: "complete",
	progress: 100,
	results: {
		detected: true,
		classification: "glioma",
		confidence: 0.8,
		tumorVolume: 1,
		wtVolume: 1,
		tcVolume: 0.5,
		etVolume: 0.2,
	},
	createdAt: "2026-09-26T00:00:00Z",
	hasSegmentation: true,
	errorMessage: null,
	...overrides,
});

describe("colour helpers", () => {
	it("accepts only #rrggbb", () => {
		expect(isHexColour("#A1b2C3")).toBe(true);
		expect(isHexColour("red")).toBe(false);
		expect(isHexColour("#fff")).toBe(false);
		expect(isHexColour("#123456;background:url(x)")).toBe(false);
	});

	it("parses channels", () => {
		expect(parseHex("#ff8000")).toEqual({ r: 255, g: 128, b: 0 });
		expect(parseHex("nonsense")).toEqual({ r: 255, g: 255, b: 255 });
	});
});

describe("useOverlayColors", () => {
	beforeEach(() => localStorage.clear());

	it("ignores tampered or malformed stored values", () => {
		localStorage.setItem(
			"overlayColors",
			JSON.stringify({ wt: "url(javascript:alert(1))", tc: "#00ff00" }),
		);
		const { result } = renderHook(() => useOverlayColors());
		expect(result.current.colors).toEqual({
			...DEFAULT_OVERLAY_COLORS,
			tc: "#00ff00",
		});
	});

	it("survives unparseable storage", () => {
		localStorage.setItem("overlayColors", "{not json");
		const { result } = renderHook(() => useOverlayColors());
		expect(result.current.colors).toEqual(DEFAULT_OVERLAY_COLORS);
	});

	it("persists valid choices and rejects invalid ones", () => {
		const { result } = renderHook(() => useOverlayColors());
		act(() => result.current.setColor("et", "#123456"));
		act(() => result.current.setColor("wt", "expression(alert(1))"));
		expect(result.current.colors.et).toBe("#123456");
		expect(result.current.colors.wt).toBe(DEFAULT_OVERLAY_COLORS.wt);
		expect(JSON.parse(localStorage.getItem("overlayColors")!).et).toBe(
			"#123456",
		);
	});
});

describe("scan formatting", () => {
	it("labels results by status", () => {
		expect(resultLabel(scan())).toBe("glioma");
		expect(
			resultLabel(scan({ results: { ...scan().results!, detected: false } })),
		).toBe("No Tumour");
		expect(resultLabel(scan({ status: "processing", results: null }))).toBe(
			"Processing",
		);
		expect(resultLabel(scan({ status: "failed", results: null }))).toBe(
			"Failed",
		);
	});

	it("matches name or ID case-insensitively", () => {
		expect(matchesSearch(scan(), "jane")).toBe(true);
		expect(matchesSearch(scan(), "ab12")).toBe(true);
		expect(matchesSearch(scan(), "smith")).toBe(false);
	});
});

describe("password policy", () => {
	it("mirrors the server rules", () => {
		expect(passwordProblem("short")).toMatch(/at least 12/);
		expect(passwordProblem("a".repeat(12))).toBeNull();
		expect(passwordProblem("é".repeat(37))).toMatch(/72 bytes/);
	});
});

describe("initials", () => {
	it("takes the first letter of each name part", () => {
		expect(initials("Alex  Rivera")).toBe("AR");
		expect(initials("")).toBe("");
	});
});
