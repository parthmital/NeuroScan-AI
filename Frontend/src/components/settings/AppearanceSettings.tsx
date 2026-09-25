import { useState } from "react";
import { Check, Moon, Sun } from "lucide-react";
import { useOverlayColors } from "@/hooks/use-overlay-colors";
import { isDarkTheme, setTheme } from "@/lib/theme";
import type { TumourRegion } from "@/lib/types";
import { cn } from "@/lib/utils";

const REGIONS: { id: TumourRegion; label: string }[] = [
	{ id: "wt", label: "Whole Tumour (WT)" },
	{ id: "tc", label: "Tumour Core (TC)" },
	{ id: "et", label: "Enhancing Tumour (ET)" },
];

// High-contrast hues that stay distinguishable over greyscale MRI.
const SWATCHES = [
	{ hex: "#eab308", name: "Amber" },
	{ hex: "#ef4444", name: "Red" },
	{ hex: "#3b82f6", name: "Blue" },
	{ hex: "#22c55e", name: "Green" },
	{ hex: "#a855f7", name: "Purple" },
	{ hex: "#f97316", name: "Orange" },
	{ hex: "#06b6d4", name: "Cyan" },
	{ hex: "#ec4899", name: "Pink" },
];

const ThemeOption = ({
	selected,
	onSelect,
	icon,
	title,
	description,
}: {
	selected: boolean;
	onSelect: () => void;
	icon: React.ReactNode;
	title: string;
	description: string;
}) => (
	<button
		type="button"
		role="radio"
		aria-checked={selected}
		onClick={onSelect}
		className={cn(
			"flex items-center gap-3 rounded-lg border-2 p-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
			selected
				? "border-medical bg-medical/5"
				: "border-border hover:border-muted-foreground/30",
		)}
	>
		{icon}
		<div>
			<p className="text-sm font-medium text-foreground">{title}</p>
			<p className="text-xs text-muted-foreground">{description}</p>
		</div>
	</button>
);

export const AppearanceSettings = () => {
	const [dark, setDark] = useState(isDarkTheme);
	const { colors, setColor } = useOverlayColors();

	const chooseTheme = (value: boolean) => {
		setTheme(value);
		setDark(value);
	};

	return (
		<div className="space-y-6 rounded-xl border border-border bg-card p-6">
			<div>
				<h2 className="mb-1 text-sm font-semibold text-foreground">Theme</h2>
				<p className="mb-4 text-xs text-muted-foreground">
					Customise the application appearance.
				</p>
				<div
					role="radiogroup"
					aria-label="Theme"
					className="grid grid-cols-1 gap-3 sm:grid-cols-2"
				>
					<ThemeOption
						selected={!dark}
						onSelect={() => chooseTheme(false)}
						icon={<Sun className="h-5 w-5 text-warning" />}
						title="Light"
						description="Clean and bright interface"
					/>
					<ThemeOption
						selected={dark}
						onSelect={() => chooseTheme(true)}
						icon={<Moon className="h-5 w-5 text-info" />}
						title="Dark"
						description="Optimised for low-light reading"
					/>
				</div>
			</div>

			<div>
				<h2 className="mb-1 text-sm font-semibold text-foreground">
					Overlay Defaults
				</h2>
				<p className="mb-4 text-xs text-muted-foreground">
					Set default segmentation overlay colours.
				</p>
				<div className="space-y-3">
					{REGIONS.map((region) => (
						<div
							key={region.id}
							className="flex flex-col gap-3 rounded-lg border border-border bg-muted/30 p-3 sm:flex-row sm:items-center"
						>
							<span className="flex-1 text-sm text-foreground">
								{region.label}
							</span>
							<div
								role="radiogroup"
								aria-label={`${region.label} colour`}
								className="flex flex-wrap gap-2"
							>
								{SWATCHES.map((swatch) => {
									const selected = colors[region.id] === swatch.hex;
									return (
										<button
											key={swatch.hex}
											type="button"
											role="radio"
											aria-checked={selected}
											aria-label={swatch.name}
											onClick={() => setColor(region.id, swatch.hex)}
											className={cn(
												"flex h-7 w-7 items-center justify-center rounded-md border-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
												selected ? "border-foreground" : "border-transparent",
											)}
											style={{ backgroundColor: swatch.hex }}
										>
											{selected && <Check className="h-4 w-4 text-white" />}
										</button>
									);
								})}
							</div>
						</div>
					))}
				</div>
			</div>
		</div>
	);
};
