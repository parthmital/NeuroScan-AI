import type { Plane } from "./slicing";

export type Tool = "crosshair" | "pan" | "zoom" | "window" | "ruler";

/** A distance measured on one slice, in image pixel coordinates. */
export interface Measurement {
	depth: number;
	from: [number, number];
	to: [number, number];
}

export type Measurements = Record<Plane, Measurement | null>;

export interface ViewTransform {
	zoom: number;
	/** Offset of the image centre from the viewport centre, in CSS pixels. */
	panX: number;
	panY: number;
}

export const IDENTITY_VIEW: ViewTransform = { zoom: 1, panX: 0, panY: 0 };
export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 16;

export const clampZoom = (zoom: number) =>
	Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));

/**
 * Zooms to `zoom` while keeping the image point that was under (x, y) at `start` under
 * (toX, toY). Coordinates are relative to the viewport centre.
 */
export const zoomAround = (
	start: ViewTransform,
	zoom: number,
	x: number,
	y: number,
	toX = x,
	toY = y,
): ViewTransform => {
	const next = clampZoom(zoom);
	const u = (x - start.panX) / start.zoom;
	const v = (y - start.panY) / start.zoom;
	return { zoom: next, panX: toX - u * next, panY: toY - v * next };
};

const NICE_LENGTHS = [1, 2, 5, 10, 20, 50, 100, 200];

/** The longest round length in mm whose bar stays under `maxPx` at `pxPerMm`. */
export const scaleBarLength = (pxPerMm: number, maxPx: number) => {
	let best = NICE_LENGTHS[0];
	for (const length of NICE_LENGTHS) {
		if (length * pxPerMm <= maxPx) best = length;
	}
	return best;
};

export const measurementLength = (
	measurement: Measurement,
	pixel: [number, number],
) =>
	Math.hypot(
		(measurement.to[0] - measurement.from[0]) * pixel[0],
		(measurement.to[1] - measurement.from[1]) * pixel[1],
	);

/** Compact intensity for on-image labels: no decimals once values are large. */
export const formatValue = (value: number) =>
	Math.abs(value) >= 100
		? Math.round(value).toString()
		: Number(value.toPrecision(3)).toString();
