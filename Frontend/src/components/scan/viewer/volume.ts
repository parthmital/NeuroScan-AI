import * as nifti from "nifti-reader-js";

export type Vec3 = [number, number, number];

/** Intensity window: values below `lower` render black, above `upper` white. */
export interface IntensityWindow {
	level: number;
	width: number;
}

/** A scalar image volume, stored x fastest (NIfTI order). */
export interface Volume {
	dims: Vec3;
	/** Voxel size in mm. */
	spacing: Vec3;
	data: Float32Array;
	min: number;
	max: number;
	/** Counts of foreground voxels (above `min`) in HISTOGRAM_BINS equal bins over min..max. */
	histogram: Uint32Array;
	/** Foreground intensity at the 0.5th and 99.5th percentiles. */
	percentiles: { low: number; high: number };
}

/** Segmentation labels: 0 background, 1 WT only, 2 TC (not enhancing), 3 ET. */
export interface LabelVolume {
	dims: Vec3;
	data: Uint8Array;
}

export const HISTOGRAM_BINS = 128;
// Finer bins used only to place the percentiles.
const PERCENTILE_BINS = 4096;

interface Decoded {
	dims: Vec3;
	spacing: Vec3;
	values: ArrayLike<number>;
	slope: number;
	intercept: number;
}

type Reader = (view: DataView, offset: number, littleEndian: boolean) => number;

const READERS: Record<number, [bytes: number, read: Reader]> = {
	[nifti.NIFTI1.TYPE_UINT8]: [1, (v, o) => v.getUint8(o)],
	[nifti.NIFTI1.TYPE_INT8]: [1, (v, o) => v.getInt8(o)],
	[nifti.NIFTI1.TYPE_INT16]: [2, (v, o, le) => v.getInt16(o, le)],
	[nifti.NIFTI1.TYPE_UINT16]: [2, (v, o, le) => v.getUint16(o, le)],
	[nifti.NIFTI1.TYPE_INT32]: [4, (v, o, le) => v.getInt32(o, le)],
	[nifti.NIFTI1.TYPE_UINT32]: [4, (v, o, le) => v.getUint32(o, le)],
	[nifti.NIFTI1.TYPE_FLOAT32]: [4, (v, o, le) => v.getFloat32(o, le)],
	[nifti.NIFTI1.TYPE_FLOAT64]: [8, (v, o, le) => v.getFloat64(o, le)],
};

// The viewer works in one voxel order whatever the file uses: +x towards the patient's
// left, +y posterior, +z superior (LPS). NIfTI affines map voxels to RAS, so these are
// the RAS signs of the canonical directions.
const CANONICAL_RAS_SIGN: Vec3 = [-1, -1, 1];

interface AxisMapping {
	axis: number;
	flip: boolean;
}

/**
 * For each file axis, the canonical axis it becomes and whether it runs backwards. Oblique
 * volumes snap to the nearest axis; an unusable affine leaves the order unchanged.
 */
export const canonicalAxes = (affine: number[][]): AxisMapping[] => {
	const mapping = [0, 1, 2].map((column) => {
		let world = 0;
		for (let w = 1; w < 3; w++) {
			if (Math.abs(affine[w][column]) > Math.abs(affine[world][column]))
				world = w;
		}
		return {
			axis: world,
			flip: affine[world][column] * CANONICAL_RAS_SIGN[world] < 0,
		};
	});
	return new Set(mapping.map((m) => m.axis)).size === 3
		? mapping
		: [0, 1, 2].map((axis) => ({ axis, flip: false }));
};

const reorient = (
	values: Float64Array,
	dims: Vec3,
	spacing: Vec3,
	mapping: AxisMapping[],
) => {
	if (mapping.every((m, c) => m.axis === c && !m.flip)) {
		return { values, dims, spacing };
	}
	const outDims: Vec3 = [0, 0, 0];
	const outSpacing: Vec3 = [0, 0, 0];
	mapping.forEach((m, c) => {
		outDims[m.axis] = dims[c];
		outSpacing[m.axis] = spacing[c];
	});
	const strides = [1, outDims[0], outDims[0] * outDims[1]];
	const step = mapping.map((m) => (m.flip ? -1 : 1) * strides[m.axis]);
	const start = mapping.reduce(
		(sum, m, c) => sum + (m.flip ? (dims[c] - 1) * strides[m.axis] : 0),
		0,
	);
	const out = new Float64Array(values.length);
	let source = 0;
	for (let k = 0; k < dims[2]; k++) {
		for (let j = 0; j < dims[1]; j++) {
			let target = start + j * step[1] + k * step[2];
			for (let i = 0; i < dims[0]; i++) {
				out[target] = values[source++];
				target += step[0];
			}
		}
	}
	return { values: out, dims: outDims, spacing: outSpacing };
};

/** Reads the first 3D frame of a (possibly gzipped) NIfTI file, in LPS voxel order. */
const decode = (buffer: ArrayBuffer): Decoded => {
	const raw = nifti.isCompressed(buffer) ? nifti.decompress(buffer) : buffer;
	if (!nifti.isNIFTI(raw)) {
		throw new Error("The server did not return a valid NIfTI volume.");
	}
	const header = nifti.readHeader(raw);
	const reader = READERS[header.datatypeCode];
	if (!reader) {
		throw new Error(
			`NIfTI data type ${header.datatypeCode} is not supported by the viewer.`,
		);
	}
	const dims: Vec3 = [
		header.dims[1],
		Math.max(1, header.dims[2]),
		Math.max(1, header.dims[3]),
	];
	const spacing: Vec3 = [
		Math.abs(header.pixDims[1]) || 1,
		Math.abs(header.pixDims[2]) || 1,
		Math.abs(header.pixDims[3]) || 1,
	];
	const count = dims[0] * dims[1] * dims[2];
	const [bytes, read] = reader;
	const view = new DataView(nifti.readImage(header, raw));
	if (view.byteLength < count * bytes) {
		throw new Error("The NIfTI volume is truncated.");
	}
	const values = new Float64Array(count);
	for (let i = 0; i < count; i++) {
		values[i] = read(view, i * bytes, header.littleEndian);
	}
	const slope = Number.isFinite(header.scl_slope) ? header.scl_slope : 0;
	const intercept = Number.isFinite(header.scl_inter) ? header.scl_inter : 0;
	return {
		...reorient(values, dims, spacing, canonicalAxes(header.affine)),
		slope,
		intercept,
	};
};

const percentileOf = (
	bins: Uint32Array,
	total: number,
	fraction: number,
	min: number,
	binWidth: number,
) => {
	const target = total * fraction;
	let seen = 0;
	for (let i = 0; i < bins.length; i++) {
		seen += bins[i];
		if (seen >= target) return min + (i + 1) * binWidth;
	}
	return min + bins.length * binWidth;
};

/** Parses an intensity volume and precomputes the statistics the viewer needs. */
export const parseVolume = (buffer: ArrayBuffer): Volume => {
	const { dims, spacing, values, slope, intercept } = decode(buffer);
	const data = new Float32Array(values.length);
	const scale = slope === 0 ? 1 : slope;
	let min = Infinity;
	let max = -Infinity;
	for (let i = 0; i < values.length; i++) {
		const value = values[i] * scale + intercept;
		data[i] = value;
		if (value < min) min = value;
		if (value > max) max = value;
	}
	if (!Number.isFinite(min)) {
		min = 0;
		max = 0;
	}

	const range = max - min || 1;
	const fine = new Uint32Array(PERCENTILE_BINS);
	const histogram = new Uint32Array(HISTOGRAM_BINS);
	let foreground = 0;
	for (let i = 0; i < data.length; i++) {
		// The minimum is almost always the air around the head; leaving it out keeps it
		// from dominating the histogram and the automatic window.
		if (data[i] <= min) continue;
		const t = (data[i] - min) / range;
		fine[Math.min(PERCENTILE_BINS - 1, Math.floor(t * PERCENTILE_BINS))]++;
		histogram[Math.min(HISTOGRAM_BINS - 1, Math.floor(t * HISTOGRAM_BINS))]++;
		foreground++;
	}
	const binWidth = range / PERCENTILE_BINS;
	const percentiles =
		foreground === 0
			? { low: min, high: max }
			: {
					low: percentileOf(fine, foreground, 0.005, min, binWidth),
					high: percentileOf(fine, foreground, 0.995, min, binWidth),
				};
	return { dims, spacing, data, min, max, histogram, percentiles };
};

/** Parses a segmentation volume, keeping only the known labels. */
export const parseLabels = (buffer: ArrayBuffer): LabelVolume => {
	const { dims, values } = decode(buffer);
	const data = new Uint8Array(values.length);
	for (let i = 0; i < values.length; i++) {
		const label = Math.round(values[i]);
		data[i] = label >= 1 && label <= 3 ? label : 0;
	}
	return { dims, data };
};

export const sameGrid = (a: { dims: Vec3 }, b: { dims: Vec3 }) =>
	a.dims[0] === b.dims[0] && a.dims[1] === b.dims[1] && a.dims[2] === b.dims[2];

export const windowBounds = ({ level, width }: IntensityWindow) => ({
	lower: level - width / 2,
	upper: level + width / 2,
});

export const windowFromBounds = (
	lower: number,
	upper: number,
): IntensityWindow => ({
	level: (lower + upper) / 2,
	width: Math.max(upper - lower, Number.EPSILON),
});

export type WindowPreset = "auto" | "full" | "tight";

/** Windows suited to MRI, whose intensities have no fixed units. */
export const presetWindow = (
	volume: Volume,
	preset: WindowPreset,
): IntensityWindow => {
	const { low, high } = volume.percentiles;
	switch (preset) {
		case "full":
			return windowFromBounds(volume.min, volume.max);
		case "tight":
			// Narrows onto the upper tissue range, where lesions usually stand out.
			return windowFromBounds(low + (high - low) * 0.35, high);
		default:
			return windowFromBounds(low, high);
	}
};

export const voxelIndex = (dims: Vec3, [i, j, k]: Vec3) =>
	i + j * dims[0] + k * dims[0] * dims[1];
