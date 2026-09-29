import {
	useCallback,
	useMemo,
	useRef,
	useState,
	useSyncExternalStore,
	type ReactNode,
} from "react";
import {
	Camera,
	Contrast,
	Crosshair,
	Eraser,
	Hand,
	LayoutGrid,
	RotateCcw,
	Ruler,
	Square,
	ZoomIn,
	type LucideIcon,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/segmented";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { OverlayColors, Overlays } from "@/lib/types";
import { SliceView } from "./SliceView";
import { planeGeometry, PLANES, type Plane, type Slab } from "./slicing";
import {
	formatValue,
	type Measurement,
	type Measurements,
	type Tool,
} from "./tools";
import { useLabels, useVolume } from "./use-volume";
import { ViewerControls, type ContrastPreset } from "./ViewerControls";
import {
	presetWindow,
	sameGrid,
	voxelIndex,
	windowBounds,
	type IntensityWindow,
	type Vec3,
	type Volume,
	type WindowPreset,
} from "./volume";
import { VolumeView, type RenderSettings } from "./VolumeView";

type Pane = Plane | "3d";

// Reading order of the grid: axial and 3D on top, coronal and sagittal below.
const PANES: readonly Pane[] = ["axial", "3d", "coronal", "sagittal"];

const PANE_OPTIONS = [
	{ value: "axial", label: "Axial" },
	{ value: "coronal", label: "Coronal" },
	{ value: "sagittal", label: "Sagittal" },
	{ value: "3d", label: "3D" },
] as const;

const TOOLS: { value: Tool; label: string; icon: LucideIcon }[] = [
	{ value: "crosshair", label: "Crosshair: click to place", icon: Crosshair },
	{ value: "pan", label: "Pan", icon: Hand },
	{ value: "zoom", label: "Zoom: drag up or down", icon: ZoomIn },
	{ value: "window", label: "Contrast: drag to adjust", icon: Contrast },
	{ value: "ruler", label: "Measure distance", icon: Ruler },
];

const REGION_NAMES = ["", "Whole tumour", "Tumour core", "Enhancing tumour"];

const NO_MEASUREMENTS: Measurements = {
	axial: null,
	coronal: null,
	sagittal: null,
};

const DEFAULT_RENDER: Omit<RenderSettings, "threshold"> = {
	preset: "tissue",
	opacity: 0.6,
	cut: "off",
	outlines: true,
	turntable: false,
};

const WIDE_QUERY = "(min-width: 768px)";

const useIsWide = () =>
	useSyncExternalStore(
		(onChange) => {
			const query = globalThis.matchMedia?.(WIDE_QUERY);
			query?.addEventListener("change", onChange);
			return () => query?.removeEventListener("change", onChange);
		},
		() => globalThis.matchMedia?.(WIDE_QUERY).matches ?? true,
	);

/** State that resets to a default whenever the volume changes. */
const usePerVolume = <T,>(
	volume: Volume | undefined,
	initial: (v: Volume) => T,
) => {
	const [state, setState] = useState<{ source: Volume; value: T } | null>(null);
	const value =
		volume && (state?.source === volume ? state.value : initial(volume));
	const set = useCallback(
		(next: T) => volume && setState({ source: volume, value: next }),
		[volume],
	);
	return [value, set] as const;
};

const autoThreshold = (volume: Volume) => {
	const { lower, upper } = windowBounds(presetWindow(volume, "auto"));
	return lower + (upper - lower) * 0.15;
};

const saveDataUrl = (url: string, filename: string) => {
	const link = document.createElement("a");
	link.href = url;
	link.download = filename;
	link.click();
};

const IconButton = ({
	label,
	icon: Icon,
	pressed,
	onClick,
	className,
}: {
	label: string;
	icon: LucideIcon;
	pressed?: boolean;
	onClick: () => void;
	className?: string;
}) => (
	<Tooltip>
		<TooltipTrigger asChild>
			<button
				type="button"
				aria-label={label}
				aria-pressed={pressed}
				onClick={onClick}
				className={cn(
					"flex h-8 w-8 items-center justify-center rounded-[5px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring",
					pressed &&
						"bg-card text-foreground shadow-[0_1px_2px_hsl(220_20%_10%/0.12),0_0_0_1px_hsl(var(--border))]",
					className,
				)}
			>
				<Icon className="h-4 w-4" />
			</button>
		</TooltipTrigger>
		<TooltipContent>{label}</TooltipContent>
	</Tooltip>
);

const PaneSkeleton = ({ children }: { children?: ReactNode }) => (
	<div className="flex h-full items-center justify-center rounded-lg border border-white/10 bg-[#0a0a0a]">
		{children}
	</div>
);

interface ScanViewerProps {
	scanId: string;
	modalities: string[];
	hasSegmentation: boolean;
	overlays: Overlays;
	overlayColors: OverlayColors;
	/** Extra panels shown under the display controls. */
	aside?: ReactNode;
}

export const ScanViewer = ({
	scanId,
	modalities,
	hasSegmentation,
	overlays,
	overlayColors,
	aside,
}: ScanViewerProps) => {
	const [modality, setModality] = useState(modalities[0] ?? "FLAIR");
	const volumeQuery = useVolume(scanId, modality);
	const labelsQuery = useLabels(scanId, hasSegmentation);
	const volume = volumeQuery.data;
	const labels =
		volume && labelsQuery.data && sameGrid(volume, labelsQuery.data)
			? labelsQuery.data
			: null;
	const labelsMismatch = Boolean(volume && labelsQuery.data && !labels);

	const [storedCrosshair, setCrosshair] = useState<Vec3 | null>(null);
	const [windowState, setWindowState] = usePerVolume<{
		window: IntensityWindow;
		preset: ContrastPreset;
	}>(volume, (v) => ({ window: presetWindow(v, "auto"), preset: "auto" }));
	const [threshold, setThreshold] = usePerVolume(volume, autoThreshold);
	const [invert, setInvert] = useState(false);
	const [slab, setSlab] = useState<Slab>({ mode: "thin", thickness: 9 });
	const [renderSettings, setRenderSettings] = useState(DEFAULT_RENDER);
	const [tool, setTool] = useState<Tool>("crosshair");
	const [layout, setLayout] = useState<"grid" | "single">("grid");
	const [activePane, setActivePane] = useState<Pane>("axial");
	const [resetToken, setResetToken] = useState(0);
	const [measurements, setMeasurements] = useState(NO_MEASUREMENTS);
	const captures = useRef<Partial<Record<Pane, () => Promise<string>>>>({});

	const isWide = useIsWide();
	const single = layout === "single" || !isWide;

	const geometries = useMemo(
		() =>
			volume &&
			(Object.fromEntries(
				PLANES.map((plane) => [
					plane,
					planeGeometry(plane, volume.dims, volume.spacing),
				]),
			) as Record<Plane, ReturnType<typeof planeGeometry>>),
		[volume],
	);

	// Keeps the crosshair where it was when switching between same-sized modalities.
	const crosshair: Vec3 | null = volume
		? storedCrosshair &&
			storedCrosshair.every((value, axis) => value < volume.dims[axis])
			? storedCrosshair
			: (volume.dims.map((d) => Math.floor(d / 2)) as Vec3)
		: null;

	const registerCapture = useMemo(
		() =>
			Object.fromEntries(
				PANES.map((pane) => [
					pane,
					(capture: (() => Promise<string>) | null) => {
						if (capture) captures.current[pane] = capture;
						else delete captures.current[pane];
					},
				]),
			) as Record<Pane, (capture: (() => Promise<string>) | null) => void>,
		[],
	);

	const setMeasurement = useCallback(
		(plane: Plane) => (measurement: Measurement | null) =>
			setMeasurements((current) => ({ ...current, [plane]: measurement })),
		[],
	);

	const setWindow = useCallback(
		(window: IntensityWindow) => setWindowState({ window, preset: "custom" }),
		[setWindowState],
	);

	const applyPreset = (preset: WindowPreset) =>
		volume && setWindowState({ window: presetWindow(volume, preset), preset });

	const saveImage = async () => {
		const capture = captures.current[activePane];
		try {
			const url = capture && (await capture());
			if (!url) throw new Error();
			saveDataUrl(url, `${scanId}-${modality.toLowerCase()}-${activePane}.png`);
		} catch {
			toast.error("Could not save this view as an image");
		}
	};

	const toggleMaximise = (pane: Pane) => {
		setActivePane(pane);
		setLayout(layout === "single" ? "grid" : "single");
	};

	const hasMeasurements = Object.values(measurements).some(Boolean);
	const value =
		volume && crosshair
			? volume.data[voxelIndex(volume.dims, crosshair)]
			: null;
	const region =
		labels && crosshair
			? REGION_NAMES[labels.data[voxelIndex(labels.dims, crosshair)]]
			: "";

	const renderPane = (pane: Pane) => {
		if (!volume || !crosshair || !geometries || !windowState) return null;
		const shared = {
			active: activePane === pane,
			onActivate: () => setActivePane(pane),
			maximised: single,
			onToggleMaximise: isWide ? () => toggleMaximise(pane) : undefined,
			registerCapture: registerCapture[pane],
			resetToken,
			window: windowState.window,
			invert,
			overlays,
			overlayColors,
			volume,
			labels,
			crosshair,
		};
		if (pane === "3d") {
			return (
				<VolumeView
					{...shared}
					settings={{ ...renderSettings, threshold: threshold ?? 0 }}
				/>
			);
		}
		return (
			<SliceView
				{...shared}
				geometry={geometries[pane]}
				onCrosshairChange={setCrosshair}
				onWindowChange={setWindow}
				slab={slab}
				tool={tool}
				measurement={measurements[pane]}
				onMeasurementChange={setMeasurement(pane)}
			/>
		);
	};

	const loading = volumeQuery.isPending;
	const failed = volumeQuery.isError;

	return (
		<div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
			<div className="min-w-0 space-y-3">
				<div className="flex flex-wrap items-center gap-2">
					{modalities.length > 1 && (
						<Segmented
							aria-label="Modality"
							options={modalities.map((m) => ({ value: m, label: m }))}
							value={modality}
							onChange={setModality}
						/>
					)}
					<div
						role="group"
						aria-label="Mouse tool"
						className="inline-flex h-9 items-center rounded-md border border-border bg-muted/60 p-0.5"
					>
						{TOOLS.map((t) => (
							<IconButton
								key={t.value}
								label={t.label}
								icon={t.icon}
								pressed={tool === t.value}
								onClick={() => setTool(t.value)}
							/>
						))}
					</div>
					<div className="ml-auto flex items-center gap-1">
						{hasMeasurements && (
							<IconButton
								label="Clear measurements"
								icon={Eraser}
								onClick={() => setMeasurements(NO_MEASUREMENTS)}
							/>
						)}
						<IconButton
							label="Reset zoom, pan, and camera"
							icon={RotateCcw}
							onClick={() => setResetToken((t) => t + 1)}
						/>
						<IconButton
							label={`Save ${activePane === "3d" ? "3D" : activePane} view as PNG`}
							icon={Camera}
							onClick={saveImage}
						/>
						{isWide && (
							<div
								role="group"
								aria-label="Layout"
								className="ml-1 inline-flex h-9 items-center rounded-md border border-border bg-muted/60 p-0.5"
							>
								<IconButton
									label="Four views"
									icon={LayoutGrid}
									pressed={layout === "grid"}
									onClick={() => setLayout("grid")}
								/>
								<IconButton
									label="One view"
									icon={Square}
									pressed={layout === "single"}
									onClick={() => setLayout("single")}
								/>
							</div>
						)}
					</div>
				</div>

				{single && (
					<Segmented
						aria-label="Visible view"
						options={PANE_OPTIONS}
						value={activePane}
						onChange={setActivePane}
						className="flex w-full [&>button]:flex-1 [&>button]:px-1.5"
					/>
				)}

				{failed ? (
					<div
						role="alert"
						className="flex aspect-[4/3] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border px-6 text-center"
					>
						<div>
							<p className="text-sm font-medium text-foreground">
								Could not load the {modality} volume
							</p>
							<p className="mt-1 text-[13px] text-muted-foreground">
								{volumeQuery.error instanceof Error
									? volumeQuery.error.message
									: "Check your connection and try again."}
							</p>
						</div>
						<Button
							variant="outline"
							size="sm"
							onClick={() => volumeQuery.refetch()}
						>
							<RotateCcw />
							Try again
						</Button>
					</div>
				) : (
					<div
						aria-busy={loading}
						className={cn("grid gap-2", single ? "grid-cols-1" : "grid-cols-2")}
					>
						{PANES.map((pane) => (
							<div
								key={pane}
								className={cn(
									"min-w-0",
									single
										? pane === activePane
											? "aspect-square sm:aspect-[4/3]"
											: "hidden"
										: "aspect-square",
								)}
							>
								{loading ? (
									<PaneSkeleton>
										{pane === activePane && (
											<div
												role="status"
												className="flex flex-col items-center gap-2"
											>
												<span className="h-5 w-5 animate-spin rounded-full border-2 border-white/60 border-t-transparent motion-reduce:animate-none" />
												<p className="font-mono text-[10px] uppercase tracking-[0.12em] text-white/60">
													Loading {modality} volume
												</p>
											</div>
										)}
									</PaneSkeleton>
								) : (
									renderPane(pane)
								)}
							</div>
						))}
					</div>
				)}

				{volume && crosshair && windowState && (
					<div className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-[11px] text-muted-foreground tabular-nums">
						<dl className="contents">
							<div className="flex gap-1.5">
								<dt>Voxel</dt>
								<dd className="text-foreground">{crosshair.join(", ")}</dd>
							</div>
							<div className="flex gap-1.5">
								<dt>Position</dt>
								<dd className="text-foreground">
									{crosshair
										.map((c, axis) => (c * volume.spacing[axis]).toFixed(1))
										.join(", ")}{" "}
									mm
								</dd>
							</div>
							<div className="flex gap-1.5">
								<dt>Value</dt>
								<dd className="text-foreground">
									{value === null ? "–" : formatValue(value)}
								</dd>
							</div>
							{region && (
								<div className="flex gap-1.5">
									<dt>Region</dt>
									<dd className="text-foreground">{region}</dd>
								</div>
							)}
						</dl>
						{hasSegmentation && labelsQuery.isPending && (
							<p role="status">Loading segmentation…</p>
						)}
						{(labelsMismatch || labelsQuery.isError) && (
							<p role="status" className="text-warning">
								Segmentation overlay unavailable for this modality
							</p>
						)}
					</div>
				)}
			</div>

			<div className="space-y-6">
				{volume && windowState ? (
					<ViewerControls
						volume={volume}
						window={windowState.window}
						onWindowChange={setWindow}
						preset={windowState.preset}
						onPresetChange={applyPreset}
						invert={invert}
						onInvertChange={setInvert}
						slab={slab}
						onSlabChange={setSlab}
						render={{ ...renderSettings, threshold: threshold ?? 0 }}
						onRenderChange={({ threshold: next, ...rest }) => {
							setRenderSettings(rest);
							setThreshold(next);
						}}
						hasLabels={Boolean(labels)}
					/>
				) : (
					<div
						aria-hidden
						className="h-96 animate-pulse rounded-lg border border-border bg-muted/60 motion-reduce:animate-none"
					/>
				)}
				{aside}
			</div>
		</div>
	);
};
