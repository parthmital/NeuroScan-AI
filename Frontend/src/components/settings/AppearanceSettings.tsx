import { useState } from "react";
import { Check, Moon, Sun } from "lucide-react";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
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
			"flex items-center gap-3 rounded-md border p-3.5 text-left transition-[border-color,box-shadow] focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring",
			selected
				? "border-medical shadow-[0_0_0_1px_hsl(var(--medical))]"
				: "border-input hover:border-foreground/25",
		)}
	>
		<span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-muted/60 text-muted-foreground">
			{icon}
		</span>
		<div className="min-w-0 flex-1">
			<p className="text-[13px] font-medium text-foreground">{title}</p>
			<p className="text-xs text-muted-foreground">{description}</p>
		</div>
		<span
			className={cn(
				"flex h-4 w-4 shrink-0 items-center justify-center rounded-full border transition-colors",
				selected ? "border-medical bg-medical" : "border-input",
			)}
		>
			{selected && <span className="h-1.5 w-1.5 rounded-full bg-white" />}
		</span>
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
		<div className="space-y-6">
			<Card>
				<CardHeader className="flex-col items-start justify-center gap-0.5">
					<CardTitle>Theme</CardTitle>
					<CardDescription>
						Customise the application appearance.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<div
						role="radiogroup"
						aria-label="Theme"
						className="grid grid-cols-1 gap-3 sm:grid-cols-2"
					>
						<ThemeOption
							selected={!dark}
							onSelect={() => chooseTheme(false)}
							icon={<Sun className="h-4 w-4" />}
							title="Light"
							description="Clean and bright interface"
						/>
						<ThemeOption
							selected={dark}
							onSelect={() => chooseTheme(true)}
							icon={<Moon className="h-4 w-4" />}
							title="Dark"
							description="Optimised for low-light reading"
						/>
					</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader className="flex-col items-start justify-center gap-0.5">
					<CardTitle>Overlay Defaults</CardTitle>
					<CardDescription>
						Set default segmentation overlay colours.
					</CardDescription>
				</CardHeader>
				<ul className="divide-y divide-border">
					{REGIONS.map((region) => (
						<li
							key={region.id}
							className="flex flex-col gap-3 px-5 py-3.5 sm:flex-row sm:items-center"
						>
							<span className="flex-1 text-[13px] font-medium text-foreground">
								{region.label}
							</span>
							<div
								role="radiogroup"
								aria-label={`${region.label} colour`}
								className="flex flex-wrap gap-1.5"
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
												"flex h-6 w-6 items-center justify-center rounded-[5px] ring-offset-2 ring-offset-card transition-shadow focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring",
												selected
													? "ring-2 ring-foreground"
													: "hover:ring-1 hover:ring-foreground/30",
											)}
											style={{ backgroundColor: swatch.hex }}
										>
											{selected && (
												<Check
													className="h-3.5 w-3.5 text-white"
													strokeWidth={2.5}
												/>
											)}
										</button>
									);
								})}
							</div>
						</li>
					))}
				</ul>
			</Card>
		</div>
	);
};
