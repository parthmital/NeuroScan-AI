import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { OverlayColors, Overlays, TumourRegion } from "@/lib/types";

interface SegmentationControlsProps {
	overlays: Overlays;
	onOverlaysChange: (overlays: Overlays) => void;
	overlayColors: OverlayColors;
}

export const SegmentationControls = ({
	overlays,
	onOverlaysChange,
	overlayColors,
}: SegmentationControlsProps) => {
	const regions = [
		{
			key: "wt" as const,
			label: "Whole Tumour (WT)",
			description: "Includes all tumour sub-regions",
			color: overlayColors.wt,
		},
		{
			key: "tc" as const,
			label: "Tumour Core (TC)",
			description: "Necrotic core and enhancing tumour",
			color: overlayColors.tc,
		},
		{
			key: "et" as const,
			label: "Enhancing Tumour (ET)",
			description: "Active tumour with contrast enhancement",
			color: overlayColors.et,
		},
	];
	const toggleRegion = (key: TumourRegion) => {
		onOverlaysChange({ ...overlays, [key]: !overlays[key] });
	};

	const setOpacity = (value: number) => {
		onOverlaysChange({ ...overlays, opacity: value });
	};

	return (
		<Card>
			<CardHeader>
				<CardTitle>Segmentation Overlay</CardTitle>
			</CardHeader>
			<ul className="divide-y divide-border">
				{regions.map((region) => (
					<li key={region.key} className="flex items-start gap-3 px-5 py-3">
						<span
							className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-[2px]"
							style={{ backgroundColor: region.color }}
						/>
						<div className="min-w-0 flex-1">
							<p className="text-[13px] font-medium text-foreground">
								{region.label}
							</p>
							<p className="text-xs text-muted-foreground">
								{region.description}
							</p>
						</div>
						<Switch
							checked={overlays[region.key]}
							aria-label={`Show ${region.label}`}
							onCheckedChange={() => toggleRegion(region.key)}
							className="mt-0.5 shrink-0"
						/>
					</li>
				))}
			</ul>

			<div className="border-t border-border px-5 py-4">
				<div className="mb-3 flex items-center justify-between">
					<p className="text-[13px] font-medium text-foreground">
						Overlay opacity
					</p>
					<span className="font-mono text-xs text-muted-foreground tabular-nums">
						{Math.round(overlays.opacity * 100)}%
					</span>
				</div>
				<Slider
					aria-label="Overlay opacity"
					value={[overlays.opacity]}
					onValueChange={([v]) => setOpacity(v)}
					min={0}
					max={1}
					step={0.05}
				/>
			</div>
		</Card>
	);
};
