import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

interface Overlays {
	wt: boolean;
	tc: boolean;
	et: boolean;
	opacity: number;
}

interface SegmentationControlsProps {
	overlays: Overlays;
	onOverlaysChange: (overlays: Overlays) => void;
	overlayColors?: {
		wt: string;
		tc: string;
		et: string;
	};
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
			color: overlayColors?.wt || "hsl(50, 100%, 50%)",
		},
		{
			key: "tc" as const,
			label: "Tumour Core (TC)",
			description: "Necrotic core and enhancing tumour",
			color: overlayColors?.tc || "hsl(0, 100%, 50%)",
		},
		{
			key: "et" as const,
			label: "Enhancing Tumour (ET)",
			description: "Active tumour with contrast enhancement",
			color: overlayColors?.et || "hsl(220, 100%, 60%)",
		},
	];
	const toggleRegion = (key: "wt" | "tc" | "et") => {
		onOverlaysChange({ ...overlays, [key]: !overlays[key] });
	};

	const setOpacity = (value: number) => {
		onOverlaysChange({ ...overlays, opacity: value });
	};

	return (
		<div className="space-y-5">
			<div>
				<h3 className="text-xs font-semibold tracking-wider uppercase text-muted-foreground mb-3">
					Segmentation Overlay
				</h3>
				<div className="space-y-3">
					{regions.map((region) => (
						<div
							key={region.key}
							className="flex items-start gap-3 p-3 rounded-lg bg-muted/30 border border-border"
						>
							<div
								className="w-3 h-3 rounded-sm mt-0.5 shrink-0"
								style={{ backgroundColor: region.color }}
							/>
							<div className="flex-1 min-w-0">
								<p className="text-sm font-medium text-foreground">
									{region.label}
								</p>
								<p className="text-xs text-muted-foreground">
									{region.description}
								</p>
							</div>
							<Switch
								checked={overlays[region.key]}
								onCheckedChange={() => toggleRegion(region.key)}
								className="shrink-0"
							/>
						</div>
					))}
				</div>
			</div>

			<div>
				<div className="flex items-center justify-between mb-2">
					<h3 className="text-xs font-semibold tracking-wider uppercase text-muted-foreground">
						Overlay Opacity
					</h3>
					<span className="text-xs font-mono text-muted-foreground">
						{Math.round(overlays.opacity * 100)}%
					</span>
				</div>
				<Slider
					value={[overlays.opacity]}
					onValueChange={([v]) => setOpacity(v)}
					min={0}
					max={1}
					step={0.05}
				/>
			</div>
		</div>
	);
};
