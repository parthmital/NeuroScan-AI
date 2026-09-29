import { useId, type ReactNode } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import type { SlabMode, Slab } from "./slicing";
import { formatValue } from "./tools";
import {
	HISTOGRAM_BINS,
	windowBounds,
	type IntensityWindow,
	type Volume,
	type WindowPreset,
} from "./volume";
import type { CutPlane, RenderPreset, RenderSettings } from "./VolumeView";

export type ContrastPreset = WindowPreset | "custom";

interface ViewerControlsProps {
	volume: Volume;
	window: IntensityWindow;
	onWindowChange: (window: IntensityWindow) => void;
	preset: ContrastPreset;
	onPresetChange: (preset: WindowPreset) => void;
	invert: boolean;
	onInvertChange: (invert: boolean) => void;
	slab: Slab;
	onSlabChange: (slab: Slab) => void;
	render: RenderSettings;
	onRenderChange: (render: RenderSettings) => void;
	hasLabels: boolean;
}

const CONTRAST_PRESETS = [
	{ value: "auto", label: "Auto" },
	{ value: "full", label: "Full range" },
	{ value: "tight", label: "Tight" },
] as const;

const SLAB_MODES = [
	{ value: "thin", label: "Thin" },
	{ value: "mip", label: "MIP" },
	{ value: "mean", label: "Mean" },
	{ value: "minip", label: "MinIP" },
] as const;

const CUT_PLANES = [
	{ value: "off", label: "Off" },
	{ value: "axial", label: "Axial" },
	{ value: "coronal", label: "Coronal" },
	{ value: "sagittal", label: "Sagittal" },
] as const;

const MAX_SLAB = 40;

// Stretches a segmented control across the panel with evenly sized options.
const fullWidth = "flex w-full [&>button]:flex-1 [&>button]:px-1.5";

const Section = ({
	title,
	children,
}: {
	title: string;
	children: ReactNode;
}) => (
	<section className="space-y-4 px-5 py-4">
		<h4 className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
			{title}
		</h4>
		{children}
	</section>
);

const SliderRow = ({
	label,
	value,
	disabled,
	...slider
}: {
	label: string;
	value: number;
	display: string;
	min: number;
	max: number;
	step: number;
	disabled?: boolean;
	onChange: (value: number) => void;
}) => (
	<div className={cn("space-y-2.5", disabled && "opacity-50")}>
		<div className="flex items-center justify-between gap-3">
			<span className="text-[13px] font-medium text-foreground">{label}</span>
			<span className="font-mono text-xs text-muted-foreground tabular-nums">
				{slider.display}
			</span>
		</div>
		<Slider
			aria-label={label}
			aria-valuetext={slider.display}
			value={[value]}
			onValueChange={([next]) => slider.onChange(next)}
			min={slider.min}
			max={slider.max}
			step={slider.step}
			disabled={disabled}
		/>
	</div>
);

const SwitchRow = ({
	label,
	checked,
	onChange,
}: {
	label: string;
	checked: boolean;
	onChange: (checked: boolean) => void;
}) => {
	const id = useId();
	return (
		<div className="flex items-center justify-between gap-3">
			<label htmlFor={id} className="text-[13px] font-medium text-foreground">
				{label}
			</label>
			<Switch id={id} checked={checked} onCheckedChange={onChange} />
		</div>
	);
};

/** Intensity distribution with the current window shaded over it. */
const Histogram = ({
	volume,
	window,
}: {
	volume: Volume;
	window: IntensityWindow;
}) => {
	const { histogram, min, max } = volume;
	// Log scale: the brain's tissue peaks would otherwise flatten everything else.
	const heights = Array.from(histogram, (count) => Math.log1p(count));
	const peak = Math.max(...heights, 1);
	const path = heights
		.map(
			(h, i) =>
				`L${i} ${40 - (h / peak) * 38}L${i + 1} ${40 - (h / peak) * 38}`,
		)
		.join("");
	const range = max - min || 1;
	const { lower, upper } = windowBounds(window);
	const toX = (value: number) =>
		Math.min(
			HISTOGRAM_BINS,
			Math.max(0, ((value - min) / range) * HISTOGRAM_BINS),
		);
	const x1 = toX(lower);
	const x2 = toX(upper);
	return (
		<figure className="space-y-1">
			<svg
				viewBox={`0 0 ${HISTOGRAM_BINS} 40`}
				preserveAspectRatio="none"
				role="img"
				aria-label={`Intensity histogram from ${formatValue(min)} to ${formatValue(max)}; window shows ${formatValue(lower)} to ${formatValue(upper)}`}
				className="h-14 w-full overflow-visible rounded-md bg-muted/60"
			>
				<rect
					x={x1}
					y={0}
					width={Math.max(0, x2 - x1)}
					height={40}
					className="fill-medical/10"
				/>
				<path
					d={`M0 40${path}L${HISTOGRAM_BINS} 40Z`}
					className="fill-muted-foreground/45"
				/>
				{[x1, x2].map((x, i) => (
					<line
						key={i}
						x1={x}
						x2={x}
						y1={0}
						y2={40}
						vectorEffect="non-scaling-stroke"
						className="stroke-medical"
						strokeWidth={1.5}
					/>
				))}
			</svg>
			<figcaption className="flex justify-between font-mono text-[10.5px] text-muted-foreground tabular-nums">
				<span>{formatValue(min)}</span>
				<span>{formatValue(max)}</span>
			</figcaption>
		</figure>
	);
};

export const ViewerControls = ({
	volume,
	window,
	onWindowChange,
	preset,
	onPresetChange,
	invert,
	onInvertChange,
	slab,
	onSlabChange,
	render,
	onRenderChange,
	hasLabels,
}: ViewerControlsProps) => {
	const range = volume.max - volume.min || 1;
	const step = range / 1000;
	const renderPresets = [
		{ value: "tissue", label: "Tissue" },
		...(hasLabels ? [{ value: "tumour", label: "Tumour" }] : []),
		{ value: "mip", label: "MIP" },
	] as { value: RenderPreset; label: string }[];
	const updateRender = (patch: Partial<RenderSettings>) =>
		onRenderChange({ ...render, ...patch });

	return (
		<Card>
			<CardHeader>
				<CardTitle>Display</CardTitle>
			</CardHeader>
			<div className="divide-y divide-border">
				<Section title="Contrast">
					<Segmented
						aria-label="Contrast preset"
						options={CONTRAST_PRESETS}
						value={preset as WindowPreset}
						onChange={onPresetChange}
						className={fullWidth}
					/>
					<Histogram volume={volume} window={window} />
					<SliderRow
						label="Level"
						value={window.level}
						display={formatValue(window.level)}
						min={volume.min}
						max={volume.max}
						step={step}
						onChange={(level) => onWindowChange({ ...window, level })}
					/>
					<SliderRow
						label="Width"
						value={window.width}
						display={formatValue(window.width)}
						min={step}
						max={range}
						step={step}
						onChange={(width) => onWindowChange({ ...window, width })}
					/>
					<SwitchRow
						label="Invert"
						checked={invert}
						onChange={onInvertChange}
					/>
				</Section>

				<Section title="Slab">
					<Segmented
						aria-label="Slab mode"
						options={SLAB_MODES}
						value={slab.mode}
						onChange={(mode: SlabMode) => onSlabChange({ ...slab, mode })}
						className={fullWidth}
					/>
					<SliderRow
						label="Thickness"
						value={slab.thickness}
						display={
							slab.mode === "thin" ? "1 slice" : `${slab.thickness} slices`
						}
						min={2}
						max={MAX_SLAB}
						step={1}
						disabled={slab.mode === "thin"}
						onChange={(thickness) => onSlabChange({ ...slab, thickness })}
					/>
				</Section>

				<Section title="3D view">
					<Segmented
						aria-label="3D rendering"
						options={renderPresets}
						value={render.preset}
						onChange={(value) => updateRender({ preset: value })}
						className={fullWidth}
					/>
					<SliderRow
						label="Threshold"
						value={render.threshold}
						display={formatValue(render.threshold)}
						min={volume.min}
						max={volume.max}
						step={step}
						onChange={(threshold) => updateRender({ threshold })}
					/>
					<SliderRow
						label="Opacity"
						value={render.opacity}
						display={`${Math.round(render.opacity * 100)}%`}
						min={0.05}
						max={1}
						step={0.05}
						onChange={(opacity) => updateRender({ opacity })}
					/>
					<div className="space-y-2.5">
						<p className="text-[13px] font-medium text-foreground">
							Cut at crosshair
						</p>
						<Segmented
							aria-label="Cut at crosshair"
							options={CUT_PLANES}
							value={render.cut}
							onChange={(cut: CutPlane) => updateRender({ cut })}
							className={fullWidth}
						/>
					</div>
					<SwitchRow
						label="Slice outlines"
						checked={render.outlines}
						onChange={(outlines) => updateRender({ outlines })}
					/>
					<SwitchRow
						label="Turntable"
						checked={render.turntable}
						onChange={(turntable) => updateRender({ turntable })}
					/>
				</Section>
			</div>
		</Card>
	);
};
