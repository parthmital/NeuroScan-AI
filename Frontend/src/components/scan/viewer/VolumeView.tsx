/* eslint-disable @typescript-eslint/no-explicit-any -- vtk.js ships no usable types for these dynamic imports */
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { parseHex } from "@/lib/color";
import type { OverlayColors, Overlays } from "@/lib/types";
import { PaneFrame } from "./PaneFrame";
import { DEPTH_AXIS, PLANE_COLOURS, PLANES, type Plane } from "./slicing";
import {
	windowBounds,
	type IntensityWindow,
	type LabelVolume,
	type Vec3,
	type Volume,
} from "./volume";

export type RenderPreset = "tissue" | "tumour" | "mip";
export type CutPlane = "off" | Plane;

export interface RenderSettings {
	preset: RenderPreset;
	/** Intensity below which tissue is fully transparent. */
	threshold: number;
	/** Peak tissue opacity, 0-1. */
	opacity: number;
	cut: CutPlane;
	outlines: boolean;
	turntable: boolean;
}

interface VolumeViewProps {
	volume: Volume;
	labels: LabelVolume | null;
	window: IntensityWindow;
	invert: boolean;
	settings: RenderSettings;
	crosshair: Vec3;
	overlays: Overlays;
	overlayColors: OverlayColors;
	resetToken: number;
	active: boolean;
	onActivate: () => void;
	maximised: boolean;
	onToggleMaximise?: () => void;
	registerCapture: (capture: (() => Promise<string>) | null) => void;
}

const BACKGROUND: [number, number, number] = [10 / 255, 10 / 255, 10 / 255];
// Tissue stays faintly visible around the tumour for context.
const TUMOUR_CONTEXT_OPACITY = 0.06;
const TURNTABLE_DEGREES_PER_FRAME = 0.3;

const unitRgb = (hex: string): Vec3 => {
	const { r, g, b } = parseHex(hex);
	return [r / 255, g / 255, b / 255];
};

interface Scene {
	renderer: any;
	renderWindow: any;
	mapper: any;
	property: any;
	colour: any;
	opacity: any;
	segMapper: any;
	segColour: any;
	segOpacity: any;
	outlines: Record<Plane, { actor: any; polyData: any }>;
	newPlane: (origin: Vec3, normal: Vec3) => any;
	/** Shows a volume; returns true when its size differs from the previous one. */
	setVolume: (volume: Volume) => boolean;
	setLabels: (labels: LabelVolume | null) => void;
	resetCamera: () => void;
	dispose: () => void;
}

/** Corners of the rectangle where a plane cuts the volume's bounding box, in mm. */
const outlinePoints = (plane: Plane, crosshair: Vec3, volume: Volume) => {
	const extent = volume.dims.map((d, axis) => (d - 1) * volume.spacing[axis]);
	const axis = DEPTH_AXIS[plane];
	const [u, v] = [0, 1, 2].filter((a) => a !== axis);
	const points = new Float32Array(12);
	[
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	].forEach(([a, b], corner) => {
		points[corner * 3 + axis] = crosshair[axis] * volume.spacing[axis];
		points[corner * 3 + u] = a * extent[u];
		points[corner * 3 + v] = b * extent[v];
	});
	return points;
};

/**
 * Builds the renderer once; volumes are swapped in later. Returns null when the view went
 * away while vtk.js was loading, so no WebGL context is created for nothing.
 */
const createScene = async (
	container: HTMLDivElement,
	isCancelled: () => boolean,
): Promise<Scene | null> => {
	const [
		{ default: vtkGenericRenderWindow },
		{ default: vtkVolume },
		{ default: vtkVolumeMapper },
		{ default: vtkColorTransferFunction },
		{ default: vtkPiecewiseFunction },
		{ default: vtkImageData },
		{ default: vtkDataArray },
		{ default: vtkInteractorStyleTrackballCamera },
		{ default: vtkPlane },
		{ default: vtkPolyData },
		{ default: vtkActor },
		{ default: vtkMapper },
		{ default: vtkAnnotatedCubeActor },
		{ default: vtkOrientationMarkerWidget },
	] = await Promise.all([
		import("@kitware/vtk.js/Rendering/Misc/GenericRenderWindow"),
		import("@kitware/vtk.js/Rendering/Core/Volume"),
		import("@kitware/vtk.js/Rendering/Core/VolumeMapper"),
		import("@kitware/vtk.js/Rendering/Core/ColorTransferFunction"),
		import("@kitware/vtk.js/Common/DataModel/PiecewiseFunction"),
		import("@kitware/vtk.js/Common/DataModel/ImageData"),
		import("@kitware/vtk.js/Common/Core/DataArray"),
		import("@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera"),
		import("@kitware/vtk.js/Common/DataModel/Plane"),
		import("@kitware/vtk.js/Common/DataModel/PolyData"),
		import("@kitware/vtk.js/Rendering/Core/Actor"),
		import("@kitware/vtk.js/Rendering/Core/Mapper"),
		import("@kitware/vtk.js/Rendering/Core/AnnotatedCubeActor"),
		import("@kitware/vtk.js/Interaction/Widgets/OrientationMarkerWidget"),
		// Register the OpenGL implementations of the classes above.
		import("@kitware/vtk.js/Rendering/Profiles/Volume"),
		import("@kitware/vtk.js/Rendering/Profiles/Geometry"),
	]);
	if (isCancelled()) return null;

	const grw = vtkGenericRenderWindow.newInstance({ background: BACKGROUND });
	grw.setContainer(container);
	grw.resize();
	const renderer = grw.getRenderer();
	const renderWindow = grw.getRenderWindow();
	const interactor = renderWindow.getInteractor();
	interactor.setInteractorStyle(
		vtkInteractorStyleTrackballCamera.newInstance(),
	);

	let current: Volume | null = null;

	const imageOf = (
		source: { dims: Vec3; spacing: Vec3 },
		values: Float32Array | Uint8Array,
	) => {
		const image = vtkImageData.newInstance();
		image.setDimensions(source.dims);
		image.setSpacing(source.spacing);
		image.getPointData().setScalars(
			vtkDataArray.newInstance({
				name: "Scalars",
				values,
				numberOfComponents: 1,
			}),
		);
		return image;
	};

	const mapper = vtkVolumeMapper.newInstance();
	mapper.setSampleDistance(0.7);
	mapper.setAutoAdjustSampleDistances(true);
	const actor = vtkVolume.newInstance();
	actor.setMapper(mapper);
	const colour = vtkColorTransferFunction.newInstance();
	const opacity = vtkPiecewiseFunction.newInstance();
	const property = actor.getProperty();
	property.setRGBTransferFunction(0, colour);
	property.setScalarOpacity(0, opacity);
	property.setInterpolationTypeToLinear();
	property.setAmbient(0.2);
	property.setDiffuse(0.7);
	property.setSpecular(0.3);
	renderer.addVolume(actor);

	const segMapper = vtkVolumeMapper.newInstance();
	segMapper.setSampleDistance(0.7);
	segMapper.setAutoAdjustSampleDistances(true);
	const segColour = vtkColorTransferFunction.newInstance();
	const segOpacity = vtkPiecewiseFunction.newInstance();
	const segActor = vtkVolume.newInstance();
	segActor.setMapper(segMapper);
	const segProperty = segActor.getProperty();
	segProperty.setRGBTransferFunction(0, segColour);
	segProperty.setScalarOpacity(0, segOpacity);
	// Labels are categories: never blend between them.
	segProperty.setInterpolationTypeToNearest();
	segProperty.setShade(false);

	const outlines = Object.fromEntries(
		PLANES.map((plane) => {
			const polyData = vtkPolyData.newInstance();
			// Positioned once a volume and crosshair arrive.
			polyData.getPoints().setData(new Float32Array(12), 3);
			polyData.getLines().setData(Uint32Array.from([5, 0, 1, 2, 3, 0]));
			const lineMapper = vtkMapper.newInstance();
			lineMapper.setInputData(polyData);
			const lineActor = vtkActor.newInstance();
			lineActor.setMapper(lineMapper);
			lineActor.getProperty().setColor(...unitRgb(PLANE_COLOURS[plane]));
			lineActor.getProperty().setLineWidth(2);
			renderer.addActor(lineActor);
			return [plane, { actor: lineActor, polyData }];
		}),
	) as Scene["outlines"];

	// Anatomical labels follow the 2D views: +x is the patient's left, +y posterior, +z superior.
	// Face rotations are vtk.js's LPS preset, which keeps text upright with z pointing up.
	// Everything is passed up front: each later style change redraws all six faces.
	// `defaultStyle` is a supported initial value that vtk.js's typings leave out.
	const cube = vtkAnnotatedCubeActor.newInstance({
		defaultStyle: {
			fontStyle: "bold",
			fontFamily: "IBM Plex Sans, Arial, sans-serif",
			fontColor: "#f4f4f5",
			faceColor: "#3f3f46",
			edgeThickness: 0.06,
			edgeColor: "#71717a",
			resolution: 256,
			// This style replaces vtk.js's default rather than merging, so keep its font scale.
			fontSizeScale: (resolution: number) => resolution / 1.8,
		},
		xPlusFaceProperty: { text: "L", faceRotation: 90 },
		xMinusFaceProperty: { text: "R", faceRotation: -90 },
		yPlusFaceProperty: { text: "P", faceRotation: 180 },
		yMinusFaceProperty: { text: "A", faceRotation: 0 },
		zPlusFaceProperty: { text: "S", faceRotation: 0 },
		zMinusFaceProperty: { text: "I", faceRotation: 180 },
	} as any);
	const marker = vtkOrientationMarkerWidget.newInstance({
		actor: cube,
		interactor,
	});
	marker.setEnabled(true);
	marker.setViewportCorner(vtkOrientationMarkerWidget.Corners.BOTTOM_LEFT);
	marker.setViewportSize(0.16);
	marker.setMinPixelSize(68);
	marker.setMaxPixelSize(104);

	const resetCamera = () => {
		if (!current) return;
		const camera = renderer.getActiveCamera();
		const { dims, spacing } = current;
		const centre = dims.map((d, axis) => ((d - 1) * spacing[axis]) / 2);
		// The marker re-fits its own camera on every change, and the half-updated camera in
		// between can have its view-up along the view direction; pause it until done.
		marker.setEnabled(false);
		// Seen from the front (-y is anterior), turned and raised a little so it reads as 3D.
		camera.setFocalPoint(centre[0], centre[1], centre[2]);
		camera.setPosition(centre[0], centre[1] - 1, centre[2]);
		camera.setViewUp(0, 0, 1);
		renderer.resetCamera();
		camera.azimuth(-35);
		camera.elevation(18);
		camera.orthogonalizeViewUp();
		renderer.resetCameraClippingRange();
		marker.setEnabled(true);
		renderWindow.render();
	};

	const sameGeometry = (a: Volume, b: Volume) =>
		a.dims.every((d, axis) => d === b.dims[axis]) &&
		a.spacing.every((s, axis) => s === b.spacing[axis]);

	const setVolume = (volume: Volume) => {
		const resized = !current || !sameGeometry(current, volume);
		current = volume;
		mapper.setInputData(imageOf(volume, volume.data));
		return resized;
	};

	let hasLabels = false;
	const setLabels = (labels: LabelVolume | null) => {
		if (labels && current) {
			segMapper.setInputData(imageOf(current, labels.data));
			if (!hasLabels) renderer.addVolume(segActor);
		} else if (hasLabels) {
			renderer.removeVolume(segActor);
		}
		hasLabels = Boolean(labels && current);
	};

	const observer = new ResizeObserver(() => {
		grw.resize();
		renderWindow.render();
	});
	observer.observe(container);

	return {
		renderer,
		renderWindow,
		mapper,
		property,
		colour,
		opacity,
		segMapper,
		segColour,
		segOpacity,
		outlines,
		newPlane: (origin, normal) => vtkPlane.newInstance({ origin, normal }),
		setVolume,
		setLabels,
		resetCamera,
		dispose: () => {
			observer.disconnect();
			marker.setEnabled(false);
			marker.delete();
			grw.delete();
		},
	};
};

export const VolumeView = ({
	volume,
	labels,
	window,
	invert,
	settings,
	crosshair,
	overlays,
	overlayColors,
	resetToken,
	active,
	onActivate,
	maximised,
	onToggleMaximise,
	registerCapture,
}: VolumeViewProps) => {
	const containerRef = useRef<HTMLDivElement>(null);
	const [scene, setScene] = useState<Scene | null>(null);
	const [error, setError] = useState<string | null>(null);
	useEffect(() => {
		const container = containerRef.current;
		if (!container) return;
		let cancelled = false;
		let built: Scene | null = null;
		createScene(container, () => cancelled)
			.then((created) => {
				if (!created) return;
				if (cancelled) {
					created.dispose();
					return;
				}
				built = created;
				setScene(created);
			})
			.catch((err: unknown) => {
				if (!cancelled) {
					setError(
						err instanceof Error
							? err.message
							: "This browser could not start the 3D renderer.",
					);
				}
			});
		return () => {
			cancelled = true;
			built?.dispose();
		};
	}, []);

	// Switching modality keeps the camera; only a differently sized volume re-frames it.
	useEffect(() => {
		if (scene?.setVolume(volume)) scene.resetCamera();
		scene?.renderWindow.render();
	}, [scene, volume]);

	useEffect(() => {
		scene?.setLabels(labels);
		scene?.renderWindow.render();
	}, [scene, labels]);

	// Tissue colour and opacity follow the shared intensity window.
	useEffect(() => {
		if (!scene) return;
		const { lower, upper } = windowBounds(window);
		const [dark, light] = invert ? [1, 0] : [0, 1];
		scene.colour.removeAllPoints();
		scene.colour.addRGBPoint(lower, dark, dark, dark);
		scene.colour.addRGBPoint(upper, light, light, light);

		const isMip = settings.preset === "mip";
		if (isMip) scene.mapper.setBlendModeToMaximumIntensity();
		else scene.mapper.setBlendModeToComposite();
		scene.property.setShade(!isMip);

		const peak =
			settings.opacity *
			(settings.preset === "tumour" ? TUMOUR_CONTEXT_OPACITY : 1);
		const start = Math.max(volume.min, settings.threshold);
		const end = Math.max(upper, start + (volume.max - volume.min) * 0.01);
		scene.opacity.removeAllPoints();
		scene.opacity.addPoint(volume.min, 0);
		if (start > volume.min) scene.opacity.addPoint(start, 0);
		scene.opacity.addPoint(end, peak);
		if (volume.max > end) scene.opacity.addPoint(volume.max, peak);
		scene.renderWindow.render();
	}, [
		scene,
		volume,
		window,
		invert,
		settings.preset,
		settings.threshold,
		settings.opacity,
	]);

	useEffect(() => {
		if (!scene) return;
		const alpha = overlays.opacity;
		scene.segColour.removeAllPoints();
		scene.segColour.addRGBPoint(0, 0, 0, 0);
		scene.segOpacity.removeAllPoints();
		scene.segOpacity.addPoint(0, 0);
		(["wt", "tc", "et"] as const).forEach((region, index) => {
			scene.segColour.addRGBPoint(index + 1, ...unitRgb(overlayColors[region]));
			scene.segOpacity.addPoint(index + 1, overlays[region] ? alpha : 0);
		});
		scene.renderWindow.render();
	}, [scene, overlays, overlayColors]);

	// Cutting at the crosshair removes the far side of the chosen plane.
	useEffect(() => {
		if (!scene) return;
		for (const mapper of [scene.mapper, scene.segMapper]) {
			mapper.removeAllClippingPlanes();
			if (settings.cut !== "off") {
				const axis = DEPTH_AXIS[settings.cut];
				const origin: Vec3 = [0, 0, 0];
				const normal: Vec3 = [0, 0, 0];
				origin[axis] = crosshair[axis] * volume.spacing[axis];
				normal[axis] = -1;
				mapper.addClippingPlane(scene.newPlane(origin, normal));
			}
		}
		scene.renderWindow.render();
	}, [scene, settings.cut, crosshair, volume]);

	useEffect(() => {
		if (!scene) return;
		for (const plane of PLANES) {
			const { actor, polyData } = scene.outlines[plane];
			polyData.getPoints().setData(outlinePoints(plane, crosshair, volume), 3);
			polyData.modified();
			actor.setVisibility(settings.outlines);
		}
		scene.renderWindow.render();
	}, [scene, settings.outlines, crosshair, volume]);

	useEffect(() => {
		if (!scene || !settings.turntable) return;
		let frame = requestAnimationFrame(function spin() {
			scene.renderer.getActiveCamera().azimuth(TURNTABLE_DEGREES_PER_FRAME);
			scene.renderer.resetCameraClippingRange();
			scene.renderWindow.render();
			frame = requestAnimationFrame(spin);
		});
		return () => cancelAnimationFrame(frame);
	}, [scene, settings.turntable]);

	const handledReset = useRef(resetToken);
	useEffect(() => {
		if (!scene || handledReset.current === resetToken) return;
		handledReset.current = resetToken;
		scene.resetCamera();
	}, [scene, resetToken]);

	useEffect(() => {
		if (!scene) return;
		registerCapture(async () => {
			const [image] = scene.renderWindow.captureImages("image/png");
			return image;
		});
		return () => registerCapture(null);
	}, [scene, registerCapture]);

	const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
		if (!scene) return;
		const camera = scene.renderer.getActiveCamera();
		const actions: Record<string, () => void> = {
			ArrowLeft: () => camera.azimuth(-10),
			ArrowRight: () => camera.azimuth(10),
			ArrowUp: () => camera.elevation(10),
			ArrowDown: () => camera.elevation(-10),
			"+": () => camera.dolly(1.2),
			"=": () => camera.dolly(1.2),
			"-": () => camera.dolly(1 / 1.2),
		};
		if (event.key === "0") {
			event.preventDefault();
			scene.resetCamera();
			return;
		}
		const action = actions[event.key];
		if (!action) return;
		event.preventDefault();
		action();
		camera.orthogonalizeViewUp();
		scene.renderer.resetCameraClippingRange();
		scene.renderWindow.render();
	};

	return (
		<PaneFrame
			title="3D"
			colour="#a1a1aa"
			active={active}
			maximised={maximised}
			onToggleMaximise={onToggleMaximise}
			info={
				<span className="truncate">
					{settings.preset === "mip"
						? "MIP"
						: settings.preset === "tumour"
							? "Tumour"
							: "Tissue"}
				</span>
			}
			footer={
				<p className="truncate font-mono text-[10.5px] text-white/50">
					Drag to rotate · scroll to zoom
				</p>
			}
		>
			<div
				ref={containerRef}
				tabIndex={0}
				role="application"
				aria-label="3D volume rendering. Drag or use arrow keys to rotate, plus and minus to zoom, 0 resets the camera."
				onPointerDown={onActivate}
				onFocus={onActivate}
				onKeyDown={onKeyDown}
				className="absolute inset-0 touch-none outline-hidden"
			/>
			{!scene && !error && (
				<div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2">
					<span className="h-5 w-5 animate-spin rounded-full border-2 border-white/60 border-t-transparent motion-reduce:animate-none" />
					<p className="font-mono text-[10px] uppercase tracking-[0.12em] text-white/60">
						Preparing 3D view
					</p>
				</div>
			)}
			{error && (
				<div
					role="alert"
					className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 bg-[#0a0a0a] px-6 text-center"
				>
					<p className="text-[13px] font-medium text-white/85">
						3D view unavailable
					</p>
					<p className="max-w-xs text-xs text-white/55">
						{error} The slice views still work.
					</p>
				</div>
			)}
		</PaneFrame>
	);
};
