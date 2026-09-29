import {
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
	type KeyboardEvent,
	type PointerEvent,
} from "react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import type { OverlayColors, Overlays } from "@/lib/types";
import { PaneFrame } from "./PaneFrame";
import {
	extractLabels,
	extractSlice,
	ORIENTATION,
	PLANE_COLOURS,
	planeOfAxis,
	renderSlice,
	screenToVoxel,
	voxelToScreen,
	type PlaneGeometry,
	type Slab,
} from "./slicing";
import {
	clampZoom,
	formatValue,
	IDENTITY_VIEW,
	measurementLength,
	scaleBarLength,
	zoomAround,
	type Measurement,
	type Tool,
	type ViewTransform,
} from "./tools";
import type { IntensityWindow, LabelVolume, Vec3, Volume } from "./volume";

interface SliceViewProps {
	geometry: PlaneGeometry;
	volume: Volume;
	labels: LabelVolume | null;
	crosshair: Vec3;
	onCrosshairChange: (crosshair: Vec3) => void;
	window: IntensityWindow;
	onWindowChange: (window: IntensityWindow) => void;
	invert: boolean;
	slab: Slab;
	overlays: Overlays;
	overlayColors: OverlayColors;
	tool: Tool;
	measurement: Measurement | null;
	onMeasurementChange: (measurement: Measurement | null) => void;
	resetToken: number;
	active: boolean;
	onActivate: () => void;
	maximised: boolean;
	onToggleMaximise?: () => void;
	registerCapture: (capture: (() => Promise<string>) | null) => void;
}

// Leaves a little room around the image so edge labels never cover anatomy.
const FIT_MARGIN = 0.92;
const VIEWPORT_BG = "#0a0a0a";
// Crosshair lines stop short of their crossing so the voxel under it stays visible.
const CROSSHAIR_GAP = 10;
// Trackpads send many small wheel deltas; this much scrolling moves one slice.
const WHEEL_STEP = 40;
const LINE_HEIGHT_PX = 16;

const CURSORS: Record<Tool, string> = {
	crosshair: "cursor-crosshair",
	pan: "cursor-grab active:cursor-grabbing",
	zoom: "cursor-zoom-in",
	window: "cursor-move",
	ruler: "cursor-crosshair",
};

interface Size {
	width: number;
	height: number;
}

const layoutImage = (
	geometry: PlaneGeometry,
	size: Size,
	view: ViewTransform,
) => {
	const widthMm = geometry.cols * geometry.pixel[0];
	const heightMm = geometry.rows * geometry.pixel[1];
	const fit =
		Math.min(size.width / widthMm, size.height / heightMm) * FIT_MARGIN;
	const pxPerMm = fit * view.zoom;
	const width = widthMm * pxPerMm;
	const height = heightMm * pxPerMm;
	return {
		pxPerMm,
		width,
		height,
		left: (size.width - width) / 2 + view.panX,
		top: (size.height - height) / 2 + view.panY,
		colPx: geometry.pixel[0] * pxPerMm,
		rowPx: geometry.pixel[1] * pxPerMm,
	};
};

type Layout = ReturnType<typeof layoutImage>;

const drawCrosshair = (
	ctx: CanvasRenderingContext2D,
	geometry: PlaneGeometry,
	layout: Layout,
	size: Size,
	crosshair: Vec3,
) => {
	const [col, row] = voxelToScreen(geometry, crosshair);
	const x = Math.round(layout.left + (col + 0.5) * layout.colPx) + 0.5;
	const y = Math.round(layout.top + (row + 0.5) * layout.rowPx) + 0.5;
	ctx.lineWidth = 1;
	ctx.globalAlpha = 0.9;
	// The vertical line is the plane that cuts across screen columns, and vice versa.
	ctx.strokeStyle = PLANE_COLOURS[planeOfAxis(geometry.colAxis)];
	ctx.beginPath();
	ctx.moveTo(x, 0);
	ctx.lineTo(x, y - CROSSHAIR_GAP);
	ctx.moveTo(x, y + CROSSHAIR_GAP);
	ctx.lineTo(x, size.height);
	ctx.stroke();
	ctx.strokeStyle = PLANE_COLOURS[planeOfAxis(geometry.rowAxis)];
	ctx.beginPath();
	ctx.moveTo(0, y);
	ctx.lineTo(x - CROSSHAIR_GAP, y);
	ctx.moveTo(x + CROSSHAIR_GAP, y);
	ctx.lineTo(size.width, y);
	ctx.stroke();
	ctx.globalAlpha = 1;
};

const drawMeasurement = (
	ctx: CanvasRenderingContext2D,
	geometry: PlaneGeometry,
	layout: Layout,
	measurement: Measurement,
) => {
	const toScreen = ([c, r]: [number, number]) =>
		[layout.left + c * layout.colPx, layout.top + r * layout.rowPx] as const;
	const [x1, y1] = toScreen(measurement.from);
	const [x2, y2] = toScreen(measurement.to);
	ctx.strokeStyle = "#fde047";
	ctx.fillStyle = "#fde047";
	ctx.lineWidth = 1.5;
	ctx.beginPath();
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
	for (const [x, y] of [
		[x1, y1],
		[x2, y2],
	]) {
		ctx.beginPath();
		ctx.arc(x, y, 3, 0, Math.PI * 2);
		ctx.fill();
	}
	const label = `${measurementLength(measurement, geometry.pixel).toFixed(1)} mm`;
	ctx.font = "500 11px 'IBM Plex Mono', ui-monospace, monospace";
	const textWidth = ctx.measureText(label).width;
	const lx = (x1 + x2) / 2 + 8;
	const ly = (y1 + y2) / 2 - 8;
	ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
	ctx.fillRect(lx - 4, ly - 12, textWidth + 8, 17);
	ctx.fillStyle = "#fde047";
	ctx.fillText(label, lx, ly);
};

interface Drag {
	tool: Tool;
	lastX: number;
	lastY: number;
	startX: number;
	startY: number;
	startView: ViewTransform;
	startWindow: IntensityWindow;
}

interface Pinch {
	distance: number;
	centreX: number;
	centreY: number;
	startView: ViewTransform;
}

export const SliceView = ({
	geometry,
	volume,
	labels,
	crosshair,
	onCrosshairChange,
	window,
	onWindowChange,
	invert,
	slab,
	overlays,
	overlayColors,
	tool,
	measurement,
	onMeasurementChange,
	resetToken,
	active,
	onActivate,
	maximised,
	onToggleMaximise,
	registerCapture,
}: SliceViewProps) => {
	const viewportRef = useRef<HTMLDivElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [size, setSize] = useState<Size>({ width: 0, height: 0 });
	// Zoom and pan, dropped whenever the viewer's reset button bumps `resetToken`.
	const [viewState, setViewState] = useState({
		token: resetToken,
		view: IDENTITY_VIEW,
	});
	const view = viewState.token === resetToken ? viewState.view : IDENTITY_VIEW;
	const setView = useCallback(
		(next: ViewTransform | ((current: ViewTransform) => ViewTransform)) =>
			setViewState((state) => {
				const current = state.token === resetToken ? state.view : IDENTITY_VIEW;
				return {
					token: resetToken,
					view: typeof next === "function" ? next(current) : next,
				};
			}),
		[resetToken],
	);
	const { plane, depthAxis, depths } = geometry;
	const depth = crosshair[depthAxis];

	useEffect(() => {
		const viewport = viewportRef.current;
		if (!viewport) return;
		const observer = new ResizeObserver(([entry]) => {
			setSize({
				width: entry.contentRect.width,
				height: entry.contentRect.height,
			});
		});
		observer.observe(viewport);
		return () => observer.disconnect();
	}, []);

	const values = useMemo(
		() => extractSlice(volume, geometry, depth, slab),
		[volume, geometry, depth, slab],
	);
	const sliceLabels = useMemo(
		() => (labels ? extractLabels(labels, geometry, depth, slab) : null),
		[labels, geometry, depth, slab],
	);
	const image = useMemo(() => {
		const canvas = document.createElement("canvas");
		canvas.width = geometry.cols;
		canvas.height = geometry.rows;
		const pixels = renderSlice(
			values,
			sliceLabels,
			window,
			invert,
			overlays,
			overlayColors,
		);
		canvas
			.getContext("2d")
			?.putImageData(new ImageData(pixels, geometry.cols, geometry.rows), 0, 0);
		return canvas;
	}, [values, sliceLabels, window, invert, overlays, overlayColors, geometry]);

	const layout = layoutImage(geometry, size, view);
	const visibleMeasurement =
		measurement && measurement.depth === depth ? measurement : null;

	useEffect(() => {
		const canvas = canvasRef.current;
		const ctx = canvas?.getContext("2d");
		if (!canvas || !ctx || size.width === 0) return;
		const dpr = globalThis.devicePixelRatio || 1;
		const width = Math.round(size.width * dpr);
		const height = Math.round(size.height * dpr);
		if (canvas.width !== width || canvas.height !== height) {
			canvas.width = width;
			canvas.height = height;
		}
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.fillStyle = VIEWPORT_BG;
		ctx.fillRect(0, 0, size.width, size.height);
		// Smooth when shrinking or mildly enlarging; show true voxels when zoomed in.
		ctx.imageSmoothingEnabled = layout.colPx < 4;
		ctx.imageSmoothingQuality = "high";
		ctx.drawImage(image, layout.left, layout.top, layout.width, layout.height);
		drawCrosshair(ctx, geometry, layout, size, crosshair);
		if (visibleMeasurement) {
			drawMeasurement(ctx, geometry, layout, visibleMeasurement);
		}
	});

	useEffect(() => {
		registerCapture(
			async () => canvasRef.current?.toDataURL("image/png") ?? "",
		);
		return () => registerCapture(null);
	}, [registerCapture]);

	// Latest values for the listeners below, which are attached once.
	const latest = useRef({
		layout,
		size,
		view,
		crosshair,
		window,
		measurement,
	});
	useEffect(() => {
		latest.current = { layout, size, view, crosshair, window, measurement };
	});

	const setDepth = useCallback(
		(next: number) => {
			const clamped = Math.min(depths - 1, Math.max(0, Math.round(next)));
			const current = latest.current.crosshair;
			if (clamped === current[depthAxis]) return;
			const updated: Vec3 = [...current];
			updated[depthAxis] = clamped;
			onCrosshairChange(updated);
		},
		[depths, depthAxis, onCrosshairChange],
	);

	/** Image coordinates (fractional pixels) of a point in the viewport. */
	const toImage = useCallback((x: number, y: number) => {
		const { layout: l } = latest.current;
		return [(x - l.left) / l.colPx, (y - l.top) / l.rowPx] as [number, number];
	}, []);

	const moveCrosshairTo = useCallback(
		(x: number, y: number) => {
			const [col, row] = toImage(x, y);
			const clampedCol = Math.min(
				geometry.cols - 1,
				Math.max(0, Math.floor(col)),
			);
			const clampedRow = Math.min(
				geometry.rows - 1,
				Math.max(0, Math.floor(row)),
			);
			const next = screenToVoxel(
				geometry,
				clampedCol,
				clampedRow,
				latest.current.crosshair[depthAxis],
			);
			const current = latest.current.crosshair;
			if (next.some((value, axis) => value !== current[axis])) {
				onCrosshairChange(next);
			}
		},
		[geometry, depthAxis, onCrosshairChange, toImage],
	);

	const clampToImage = ([col, row]: [number, number]): [number, number] => [
		Math.min(geometry.cols, Math.max(0, col)),
		Math.min(geometry.rows, Math.max(0, row)),
	];

	// Wheel scrolls slices, or zooms with Ctrl/Cmd (which is also how trackpads pinch).
	const wheelRemainder = useRef(0);
	useEffect(() => {
		const viewport = viewportRef.current;
		if (!viewport) return;
		const onWheel = (event: WheelEvent) => {
			event.preventDefault();
			const delta =
				event.deltaMode === 1 ? event.deltaY * LINE_HEIGHT_PX : event.deltaY;
			const { size: s, view: v, crosshair: c } = latest.current;
			if (event.ctrlKey || event.metaKey) {
				const rect = viewport.getBoundingClientRect();
				setView(
					zoomAround(
						v,
						v.zoom * Math.exp(-delta * 0.002),
						event.clientX - rect.left - s.width / 2,
						event.clientY - rect.top - s.height / 2,
					),
				);
				return;
			}
			wheelRemainder.current += delta;
			const steps = Math.trunc(wheelRemainder.current / WHEEL_STEP);
			if (steps === 0) return;
			wheelRemainder.current -= steps * WHEEL_STEP;
			setDepth(c[depthAxis] + steps);
		};
		viewport.addEventListener("wheel", onWheel, { passive: false });
		return () => viewport.removeEventListener("wheel", onWheel);
	}, [depthAxis, setDepth, setView]);

	const pointers = useRef(new Map<number, { x: number; y: number }>());
	const drag = useRef<Drag | null>(null);
	const pinch = useRef<Pinch | null>(null);

	const localPoint = (event: PointerEvent<HTMLDivElement>) => {
		const rect = event.currentTarget.getBoundingClientRect();
		return { x: event.clientX - rect.left, y: event.clientY - rect.top };
	};

	const startPinch = () => {
		const [a, b] = [...pointers.current.values()];
		const { size: s } = latest.current;
		pinch.current = {
			distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
			centreX: (a.x + b.x) / 2 - s.width / 2,
			centreY: (a.y + b.y) / 2 - s.height / 2,
			startView: latest.current.view,
		};
		drag.current = null;
	};

	const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
		if (event.button === 2) return;
		onActivate();
		event.currentTarget.focus({ preventScroll: true });
		event.currentTarget.setPointerCapture(event.pointerId);
		const point = localPoint(event);
		pointers.current.set(event.pointerId, point);
		if (pointers.current.size === 2) {
			startPinch();
			return;
		}
		if (pointers.current.size > 2) return;
		const dragTool: Tool = event.button === 1 ? "pan" : tool;
		drag.current = {
			tool: dragTool,
			lastX: point.x,
			lastY: point.y,
			startX: point.x - latest.current.size.width / 2,
			startY: point.y - latest.current.size.height / 2,
			startView: latest.current.view,
			startWindow: latest.current.window,
		};
		if (dragTool === "crosshair") moveCrosshairTo(point.x, point.y);
		if (dragTool === "ruler") {
			const start = clampToImage(toImage(point.x, point.y));
			onMeasurementChange({ depth, from: start, to: start });
		}
	};

	const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
		if (!pointers.current.has(event.pointerId)) return;
		const point = localPoint(event);
		pointers.current.set(event.pointerId, point);

		if (pinch.current && pointers.current.size >= 2) {
			const [a, b] = [...pointers.current.values()];
			const { size: s } = latest.current;
			const p = pinch.current;
			const distance = Math.hypot(a.x - b.x, a.y - b.y);
			setView(
				zoomAround(
					p.startView,
					(p.startView.zoom * distance) / p.distance,
					p.centreX,
					p.centreY,
					(a.x + b.x) / 2 - s.width / 2,
					(a.y + b.y) / 2 - s.height / 2,
				),
			);
			return;
		}

		const d = drag.current;
		if (!d) return;
		const dx = point.x - d.lastX;
		const dy = point.y - d.lastY;
		d.lastX = point.x;
		d.lastY = point.y;
		switch (d.tool) {
			case "crosshair":
				moveCrosshairTo(point.x, point.y);
				break;
			case "pan":
				setView((v) => ({ ...v, panX: v.panX + dx, panY: v.panY + dy }));
				break;
			case "zoom": {
				const travel = d.startY + latest.current.size.height / 2 - point.y;
				setView(
					zoomAround(
						d.startView,
						d.startView.zoom * Math.exp(travel * 0.01),
						d.startX,
						d.startY,
					),
				);
				break;
			}
			case "window": {
				// Right widens the window, up brightens the image.
				const perPixel = (volume.max - volume.min) / 500;
				const totalX = point.x - (d.startX + latest.current.size.width / 2);
				const totalY = point.y - (d.startY + latest.current.size.height / 2);
				onWindowChange({
					width: Math.max(perPixel, d.startWindow.width + totalX * perPixel),
					level: d.startWindow.level + totalY * perPixel,
				});
				break;
			}
			case "ruler": {
				const current = latest.current.measurement;
				if (current) {
					onMeasurementChange({
						...current,
						to: clampToImage(toImage(point.x, point.y)),
					});
				}
				break;
			}
		}
	};

	const onPointerEnd = (event: PointerEvent<HTMLDivElement>) => {
		pointers.current.delete(event.pointerId);
		if (pointers.current.size < 2) pinch.current = null;
		if (pointers.current.size === 0) {
			const current = latest.current.measurement;
			if (
				drag.current?.tool === "ruler" &&
				current &&
				measurementLength(current, geometry.pixel) < 0.5
			) {
				onMeasurementChange(null);
			}
			drag.current = null;
		}
	};

	const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
		const actions: Record<string, () => void> = {
			ArrowUp: () => setDepth(depth + 1),
			ArrowRight: () => setDepth(depth + 1),
			ArrowDown: () => setDepth(depth - 1),
			ArrowLeft: () => setDepth(depth - 1),
			PageUp: () => setDepth(depth + 10),
			PageDown: () => setDepth(depth - 10),
			Home: () => setDepth(0),
			End: () => setDepth(depths - 1),
			"+": () => setView((v) => ({ ...v, zoom: clampZoom(v.zoom * 1.25) })),
			"=": () => setView((v) => ({ ...v, zoom: clampZoom(v.zoom * 1.25) })),
			"-": () => setView((v) => ({ ...v, zoom: clampZoom(v.zoom / 1.25) })),
			"0": () => setView(IDENTITY_VIEW),
		};
		const action = actions[event.key];
		if (!action) return;
		event.preventDefault();
		action();
	};

	const [left, right, top, bottom] = ORIENTATION[plane];
	const barMm = scaleBarLength(layout.pxPerMm, Math.max(40, size.width * 0.22));
	const planeName = plane[0].toUpperCase() + plane.slice(1);
	const slabLabel =
		slab.mode === "thin"
			? null
			: `${slab.mode.toUpperCase()} ${slab.thickness}`;

	return (
		<PaneFrame
			title={planeName}
			colour={PLANE_COLOURS[plane]}
			active={active}
			maximised={maximised}
			onToggleMaximise={onToggleMaximise}
			info={
				<>
					{slabLabel && <span className="text-white/45">{slabLabel}</span>}
					<span>
						{depth + 1} / {depths}
					</span>
				</>
			}
			footer={
				<Slider
					aria-label={`${planeName} slice`}
					value={[depth]}
					onValueChange={([value]) => setDepth(value)}
					onPointerDown={onActivate}
					min={0}
					max={depths - 1}
					step={1}
				/>
			}
		>
			<div
				ref={viewportRef}
				tabIndex={0}
				role="application"
				aria-label={`${planeName} view, slice ${depth + 1} of ${depths}. Arrow keys change slice, plus and minus zoom, 0 resets the view.`}
				onPointerDown={onPointerDown}
				onPointerMove={onPointerMove}
				onPointerUp={onPointerEnd}
				onPointerCancel={onPointerEnd}
				onKeyDown={onKeyDown}
				onFocus={onActivate}
				onContextMenu={(event) => event.preventDefault()}
				className={cn(
					"absolute inset-0 touch-none select-none outline-hidden",
					CURSORS[tool],
				)}
			>
				<canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
				<div className="pointer-events-none absolute inset-0 font-mono text-[11px] font-medium text-white/55">
					<span className="absolute left-2 top-1/2 -translate-y-1/2">
						{left}
					</span>
					<span className="absolute right-2 top-1/2 -translate-y-1/2">
						{right}
					</span>
					<span className="absolute left-1/2 top-1.5 -translate-x-1/2">
						{top}
					</span>
					<span className="absolute bottom-1.5 left-1/2 -translate-x-1/2">
						{bottom}
					</span>
				</div>
				<div className="pointer-events-none absolute bottom-2 left-2.5 font-mono text-[10px] leading-4 text-white/50 tabular-nums">
					W {formatValue(window.width)} · L {formatValue(window.level)}
					{view.zoom !== 1 && <span> · {view.zoom.toFixed(2)}×</span>}
				</div>
				<div className="pointer-events-none absolute bottom-2 right-2.5 flex flex-col items-end gap-0.5">
					<span className="font-mono text-[10px] text-white/55 tabular-nums">
						{barMm} mm
					</span>
					<span
						className="h-1 border-x border-b border-white/60"
						style={{ width: barMm * layout.pxPerMm }}
					/>
				</div>
			</div>
		</PaneFrame>
	);
};
