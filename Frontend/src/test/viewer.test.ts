import { describe, expect, it } from "vitest";
import {
	extractLabels,
	extractSlice,
	planeGeometry,
	PLANES,
	renderSlice,
	screenToVoxel,
	voxelToScreen,
} from "@/components/scan/viewer/slicing";
import {
	measurementLength,
	scaleBarLength,
	zoomAround,
} from "@/components/scan/viewer/tools";
import {
	parseLabels,
	parseVolume,
	presetWindow,
	voxelIndex,
	windowFromBounds,
	type Vec3,
} from "@/components/scan/viewer/volume";

const DATATYPES = { uint8: [2, 8], int16: [4, 16], float32: [16, 32] } as const;

/** A minimal single-file NIfTI-1 volume. */
const niftiBuffer = (
	dims: Vec3,
	values: number[],
	{
		type = "int16",
		spacing = [1, 1, 1],
		slope = 0,
		intercept = 0,
		// RAS direction of each voxel axis; the default is already the viewer's LPS order.
		directions = [-1, -1, 1],
	}: {
		type?: keyof typeof DATATYPES;
		spacing?: Vec3;
		slope?: number;
		intercept?: number;
		directions?: Vec3;
	} = {},
) => {
	const [code, bits] = DATATYPES[type];
	const bytes = bits / 8;
	const buffer = new ArrayBuffer(352 + values.length * bytes);
	const view = new DataView(buffer);
	view.setInt32(0, 348, true);
	[3, ...dims, 1, 1, 1, 1].forEach((d, i) =>
		view.setInt16(40 + i * 2, d, true),
	);
	view.setInt16(70, code, true);
	view.setInt16(72, bits, true);
	[1, ...spacing].forEach((s, i) => view.setFloat32(76 + i * 4, s, true));
	view.setFloat32(108, 352, true);
	view.setFloat32(112, slope, true);
	view.setFloat32(116, intercept, true);
	view.setInt16(254, 1, true);
	directions.forEach((sign, axis) =>
		view.setFloat32(280 + axis * 20, sign * spacing[axis], true),
	);
	[..."n+1"].forEach((c, i) => view.setUint8(344 + i, c.charCodeAt(0)));
	values.forEach((v, i) => {
		const offset = 352 + i * bytes;
		if (type === "uint8") view.setUint8(offset, v);
		else if (type === "int16") view.setInt16(offset, v, true);
		else view.setFloat32(offset, v, true);
	});
	return buffer;
};

// Each voxel's value encodes its own coordinates, so slices are easy to check.
const DIMS: Vec3 = [4, 3, 2];
const code = ([i, j, k]: Vec3) => 100 * k + 10 * j + i + 1;
const coded = () => {
	const values: number[] = [];
	for (let k = 0; k < DIMS[2]; k++)
		for (let j = 0; j < DIMS[1]; j++)
			for (let i = 0; i < DIMS[0]; i++) values.push(code([i, j, k]));
	return parseVolume(niftiBuffer(DIMS, values, { spacing: [1, 2, 3] }));
};

describe("parseVolume", () => {
	it("reads dimensions, spacing, and range", () => {
		const volume = coded();
		expect(volume.dims).toEqual([4, 3, 2]);
		expect(volume.spacing).toEqual([1, 2, 3]);
		expect(volume.min).toBe(1);
		expect(volume.max).toBe(124);
		expect(volume.data[voxelIndex(volume.dims, [2, 1, 1])]).toBe(
			code([2, 1, 1]),
		);
	});

	it("applies the header's intensity scaling", () => {
		const volume = parseVolume(
			niftiBuffer([2, 1, 1], [1, 3], { slope: 2, intercept: 10 }),
		);
		expect(Array.from(volume.data)).toEqual([12, 16]);
	});

	it("leaves the background out of the histogram and automatic window", () => {
		const values = Array.from({ length: 100 }, (_, i) => (i < 50 ? 0 : i));
		const volume = parseVolume(niftiBuffer([100, 1, 1], values));
		const total = volume.histogram.reduce((sum, count) => sum + count, 0);
		expect(total).toBe(50);
		const auto = presetWindow(volume, "auto");
		expect(auto.level - auto.width / 2).toBeGreaterThanOrEqual(50);
	});

	it("reorients RAS files into LPS order", () => {
		// Voxel x runs towards the patient's right, so it is flipped; z already points up.
		const volume = parseVolume(
			niftiBuffer([2, 1, 2], [1, 2, 3, 4], { directions: [1, 1, 1] }),
		);
		expect(volume.dims).toEqual([2, 1, 2]);
		expect(Array.from(volume.data)).toEqual([2, 1, 4, 3]);
	});

	it("reorders swapped axes and their spacing", () => {
		const buffer = niftiBuffer([2, 3, 1], [1, 2, 3, 4, 5, 6], {
			spacing: [1, 2, 3],
		});
		const view = new DataView(buffer);
		// File axis 0 points posterior and file axis 1 points left.
		view.setFloat32(280, 0, true);
		view.setFloat32(284, -2, true);
		view.setFloat32(296, -1, true);
		view.setFloat32(300, 0, true);
		const volume = parseVolume(buffer);
		expect(volume.dims).toEqual([3, 2, 1]);
		expect(volume.spacing).toEqual([2, 1, 3]);
		expect(volume.data[voxelIndex(volume.dims, [2, 1, 0])]).toBe(6);
	});

	it("rejects data that is not NIfTI", () => {
		expect(() => parseVolume(new ArrayBuffer(400))).toThrow(/NIfTI/);
	});
});

describe("parseLabels", () => {
	it("keeps only the known tumour labels", () => {
		const labels = parseLabels(
			niftiBuffer([5, 1, 1], [0, 1, 2, 3, 4], { type: "uint8" }),
		);
		expect(Array.from(labels.data)).toEqual([0, 1, 2, 3, 0]);
	});
});

describe("plane geometry", () => {
	const volume = coded();

	it.each(PLANES)(
		"round-trips voxels through %s screen coordinates",
		(plane) => {
			const geometry = planeGeometry(plane, volume.dims, volume.spacing);
			const voxel: Vec3 = [3, 1, 0];
			const [col, row] = voxelToScreen(geometry, voxel);
			expect(
				screenToVoxel(geometry, col, row, voxel[geometry.depthAxis]),
			).toEqual(voxel);
		},
	);

	it.each(PLANES)("extracts the %s slice the geometry describes", (plane) => {
		const geometry = planeGeometry(plane, volume.dims, volume.spacing);
		const values = extractSlice(volume, geometry, 1, {
			mode: "thin",
			thickness: 1,
		});
		for (let row = 0; row < geometry.rows; row++) {
			for (let col = 0; col < geometry.cols; col++) {
				expect(values[row * geometry.cols + col]).toBe(
					code(screenToVoxel(geometry, col, row, 1)),
				);
			}
		}
	});

	it("shows anterior at the top of the axial view and superior on top elsewhere", () => {
		const axial = planeGeometry("axial", volume.dims, volume.spacing);
		// +y is posterior, so the most anterior row (y = 0) is drawn first.
		expect(screenToVoxel(axial, 0, 0, 0)[1]).toBe(0);
		const coronal = planeGeometry("coronal", volume.dims, volume.spacing);
		expect(screenToVoxel(coronal, 0, 0, 0)[2]).toBe(DIMS[2] - 1);
		expect(coronal.pixel).toEqual([1, 3]);
	});

	it("projects slabs with MIP, mean, and MinIP", () => {
		const axial = planeGeometry("axial", volume.dims, volume.spacing);
		const at = (mode: "mip" | "mean" | "minip") =>
			extractSlice(volume, axial, 0, { mode, thickness: 2 })[0];
		const bottom = code(screenToVoxel(axial, 0, 0, 0));
		const top = code(screenToVoxel(axial, 0, 0, 1));
		expect(at("mip")).toBe(top);
		expect(at("minip")).toBe(bottom);
		expect(at("mean")).toBe((top + bottom) / 2);
	});

	it("keeps the highest label through a slab", () => {
		const labels = { dims: DIMS, data: new Uint8Array(24) };
		labels.data[voxelIndex(DIMS, [0, 0, 1])] = 3;
		const axial = planeGeometry("axial", DIMS, [1, 1, 1]);
		const slab = extractLabels(labels, axial, 0, { mode: "mip", thickness: 2 });
		expect(slab[0]).toBe(3);
		expect(
			extractLabels(labels, axial, 0, { mode: "thin", thickness: 2 })[0],
		).toBe(0);
	});
});

describe("renderSlice", () => {
	const colours = { wt: "#ffff00", tc: "#ff0000", et: "#0000ff" };
	const all = { wt: true, tc: true, et: true, opacity: 1 };

	it("maps the window to black-to-white and can invert", () => {
		const values = Float32Array.from([0, 50, 110, 150]);
		const window = windowFromBounds(50, 150);
		const pixels = renderSlice(values, null, window, false, all, colours);
		expect([pixels[0], pixels[4], pixels[8], pixels[12]]).toEqual([
			0, 0, 153, 255,
		]);
		const inverted = renderSlice(values, null, window, true, all, colours);
		expect(inverted[0]).toBe(255);
	});

	it("colours each voxel with its innermost visible region", () => {
		const values = new Float32Array(3);
		const labels = Uint8Array.from([1, 3, 3]);
		const window = windowFromBounds(0, 1);
		const pixels = renderSlice(values, labels, window, false, all, colours);
		expect(Array.from(pixels.slice(0, 3))).toEqual([255, 255, 0]);
		expect(Array.from(pixels.slice(4, 7))).toEqual([0, 0, 255]);
		const noEt = renderSlice(
			values,
			labels,
			window,
			false,
			{ ...all, et: false },
			colours,
		);
		expect(Array.from(noEt.slice(8, 11))).toEqual([255, 0, 0]);
	});
});

describe("view tools", () => {
	it("zooms around a fixed point", () => {
		const start = { zoom: 1, panX: 10, panY: -5 };
		const next = zoomAround(start, 2, 40, 30);
		// The image point under (40, 30) before zooming is still under it afterwards.
		expect((40 - next.panX) / next.zoom).toBeCloseTo(
			(40 - start.panX) / start.zoom,
		);
		expect((30 - next.panY) / next.zoom).toBeCloseTo(
			(30 - start.panY) / start.zoom,
		);
	});

	it("picks a round scale bar length that fits", () => {
		expect(scaleBarLength(3, 100)).toBe(20);
		expect(scaleBarLength(0.5, 100)).toBe(200);
	});

	it("measures in millimetres using the pixel size", () => {
		expect(
			measurementLength({ depth: 0, from: [0, 0], to: [3, 2] }, [1, 2]),
		).toBe(5);
	});
});
