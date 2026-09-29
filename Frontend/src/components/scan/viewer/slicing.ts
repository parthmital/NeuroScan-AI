import { parseHex } from "@/lib/color";
import type { OverlayColors, Overlays } from "@/lib/types";
import {
	windowBounds,
	type IntensityWindow,
	type LabelVolume,
	type Vec3,
	type Volume,
} from "./volume";

export type Plane = "axial" | "coronal" | "sagittal";
export type SlabMode = "thin" | "mip" | "mean" | "minip";
export type Axis = 0 | 1 | 2;

export const PLANES: readonly Plane[] = ["axial", "coronal", "sagittal"];

/** Crosshair colours, one per plane; validated for separation on the dark viewport. */
export const PLANE_COLOURS: Record<Plane, string> = {
	axial: "#0891b2",
	coronal: "#16a34a",
	sagittal: "#d946ef",
};

/** The voxel axis each plane cuts across. */
export const DEPTH_AXIS: Record<Plane, Axis> = {
	sagittal: 0,
	coronal: 1,
	axial: 2,
};

export const planeOfAxis = (axis: Axis): Plane =>
	PLANES.find((plane) => DEPTH_AXIS[plane] === axis)!;

/** Anatomical labels at the left, right, top, and bottom edges of each view. */
export const ORIENTATION: Record<Plane, [string, string, string, string]> = {
	axial: ["R", "L", "A", "P"],
	coronal: ["R", "L", "S", "I"],
	sagittal: ["A", "P", "S", "I"],
};

/**
 * How one plane maps onto the volume. Screen columns run left to right and rows top to
 * bottom; `colAxis` and `rowAxis` are the voxel axes they follow, and `colFlip`/`rowFlip`
 * mark the axes that run against the screen direction.
 */
export interface PlaneGeometry {
	plane: Plane;
	cols: number;
	rows: number;
	depths: number;
	/** Pixel size in mm across and down. */
	pixel: [number, number];
	colAxis: Axis;
	rowAxis: Axis;
	depthAxis: Axis;
	colFlip: boolean;
	rowFlip: boolean;
	origin: number;
	colStride: number;
	rowStride: number;
	depthStride: number;
}

// Volumes arrive in LPS order (see volume.ts). Views follow the radiological convention:
// the patient's right on the screen's left, anterior or superior at the top.
const LAYOUT: Record<
	Plane,
	{ col: Axis; row: Axis; colFlip: boolean; rowFlip: boolean }
> = {
	axial: { col: 0, row: 1, colFlip: false, rowFlip: false },
	coronal: { col: 0, row: 2, colFlip: false, rowFlip: true },
	sagittal: { col: 1, row: 2, colFlip: false, rowFlip: true },
};

export const planeGeometry = (
	plane: Plane,
	dims: Vec3,
	spacing: Vec3,
): PlaneGeometry => {
	const { col, row, colFlip, rowFlip } = LAYOUT[plane];
	const depth = DEPTH_AXIS[plane];
	const strides: Vec3 = [1, dims[0], dims[0] * dims[1]];
	const cols = dims[col];
	const rows = dims[row];
	return {
		plane,
		cols,
		rows,
		depths: dims[depth],
		pixel: [spacing[col], spacing[row]],
		colAxis: col,
		rowAxis: row,
		depthAxis: depth,
		colFlip,
		rowFlip,
		origin:
			(colFlip ? (cols - 1) * strides[col] : 0) +
			(rowFlip ? (rows - 1) * strides[row] : 0),
		colStride: colFlip ? -strides[col] : strides[col],
		rowStride: rowFlip ? -strides[row] : strides[row],
		depthStride: strides[depth],
	};
};

/** Voxel coordinates of a (fractional) screen pixel on a given depth. */
export const screenToVoxel = (
	geometry: PlaneGeometry,
	col: number,
	row: number,
	depth: number,
): Vec3 => {
	const voxel: Vec3 = [0, 0, 0];
	voxel[geometry.colAxis] = geometry.colFlip ? geometry.cols - 1 - col : col;
	voxel[geometry.rowAxis] = geometry.rowFlip ? geometry.rows - 1 - row : row;
	voxel[geometry.depthAxis] = depth;
	return voxel;
};

/** Screen pixel (column, row) of a voxel, ignoring its depth. */
export const voxelToScreen = (
	geometry: PlaneGeometry,
	voxel: Vec3,
): [number, number] => {
	const c = voxel[geometry.colAxis];
	const r = voxel[geometry.rowAxis];
	return [
		geometry.colFlip ? geometry.cols - 1 - c : c,
		geometry.rowFlip ? geometry.rows - 1 - r : r,
	];
};

export interface Slab {
	mode: SlabMode;
	/** Slices combined in a projection; ignored for thin slices. */
	thickness: number;
}

/** First and last slice a slab covers, centred on `depth` and clipped to the volume. */
export const slabRange = (
	geometry: PlaneGeometry,
	depth: number,
	slab: Slab,
) => {
	const thickness = slab.mode === "thin" ? 1 : Math.max(1, slab.thickness);
	const start = Math.max(0, depth - Math.floor((thickness - 1) / 2));
	const end = Math.min(geometry.depths - 1, start + thickness - 1);
	return [start, end] as const;
};

type Reduce = "max" | "mean" | "min";

const REDUCERS: Record<SlabMode, Reduce> = {
	thin: "max",
	mip: "max",
	mean: "mean",
	minip: "min",
};

const project = <T extends Float32Array | Uint8Array>(
	source: ArrayLike<number>,
	out: T,
	geometry: PlaneGeometry,
	start: number,
	end: number,
	reduce: Reduce,
): T => {
	const { cols, rows, origin, colStride, rowStride, depthStride } = geometry;
	const count = end - start + 1;
	for (let r = 0; r < rows; r++) {
		const rowBase = origin + r * rowStride + start * depthStride;
		for (let c = 0; c < cols; c++) {
			let index = rowBase + c * colStride;
			let value = source[index];
			for (let d = 1; d < count; d++) {
				index += depthStride;
				const next = source[index];
				if (reduce === "max") value = next > value ? next : value;
				else if (reduce === "min") value = next < value ? next : value;
				else value += next;
			}
			out[r * cols + c] = reduce === "mean" ? value / count : value;
		}
	}
	return out;
};

/** Intensities of one slice, or of a projection through a slab of slices. */
export const extractSlice = (
	volume: Volume,
	geometry: PlaneGeometry,
	depth: number,
	slab: Slab,
): Float32Array => {
	const [start, end] = slabRange(geometry, depth, slab);
	return project(
		volume.data,
		new Float32Array(geometry.cols * geometry.rows),
		geometry,
		start,
		end,
		REDUCERS[slab.mode],
	);
};

/**
 * Labels of one slice. Through a slab the highest label wins: labels nest (ET inside TC
 * inside WT), so this shows every region the slab passes through.
 */
export const extractLabels = (
	labels: LabelVolume,
	geometry: PlaneGeometry,
	depth: number,
	slab: Slab,
): Uint8Array => {
	const [start, end] = slabRange(geometry, depth, slab);
	return project(
		labels.data,
		new Uint8Array(geometry.cols * geometry.rows),
		geometry,
		start,
		end,
		"max",
	);
};

/** The innermost visible region a label belongs to, if any. */
export const visibleRegion = (label: number, overlays: Overlays) => {
	if (label >= 3 && overlays.et) return "et" as const;
	if (label >= 2 && overlays.tc) return "tc" as const;
	if (label >= 1 && overlays.wt) return "wt" as const;
	return null;
};

/** Greyscale RGBA pixels for a slice, with the tumour regions blended on top. */
export const renderSlice = (
	values: Float32Array,
	labels: Uint8Array | null,
	window: IntensityWindow,
	invert: boolean,
	overlays: Overlays,
	colours: OverlayColors,
): Uint8ClampedArray<ArrayBuffer> => {
	const { lower, upper } = windowBounds(window);
	const scale = 255 / Math.max(upper - lower, Number.EPSILON);
	const palette = {
		wt: parseHex(colours.wt),
		tc: parseHex(colours.tc),
		et: parseHex(colours.et),
	};
	const alpha = Math.min(1, Math.max(0, overlays.opacity));
	const pixels = new Uint8ClampedArray(values.length * 4);
	for (let i = 0; i < values.length; i++) {
		let grey = (values[i] - lower) * scale;
		grey = grey < 0 ? 0 : grey > 255 ? 255 : grey;
		if (invert) grey = 255 - grey;
		let r = grey;
		let g = grey;
		let b = grey;
		const region =
			labels && labels[i] ? visibleRegion(labels[i], overlays) : null;
		if (region) {
			const colour = palette[region];
			r = r + (colour.r - r) * alpha;
			g = g + (colour.g - g) * alpha;
			b = b + (colour.b - b) * alpha;
		}
		const o = i * 4;
		pixels[o] = r;
		pixels[o + 1] = g;
		pixels[o + 2] = b;
		pixels[o + 3] = 255;
	}
	return pixels;
};
