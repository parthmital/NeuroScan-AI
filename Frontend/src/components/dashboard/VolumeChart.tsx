import {
	BarChart,
	Bar,
	XAxis,
	YAxis,
	Tooltip,
	ResponsiveContainer,
	CartesianGrid,
} from "recharts";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { useScans } from "@/hooks/use-scans";
import { useOverlayColors } from "@/hooks/use-overlay-colors";
import { APP_CONFIG } from "@/lib/config";

const TICK = {
	fontSize: 11,
	fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
	fill: "hsl(var(--muted-foreground))",
};

export const VolumeChart = () => {
	const { data: scans, isLoading } = useScans();
	const { colors: overlayColors } = useOverlayColors();

	if (isLoading || !scans) {
		return (
			<div className="h-80 animate-pulse rounded-lg border border-border bg-card" />
		);
	}

	const chartData = scans
		.filter((s) => s.results?.detected)
		.map((s) => ({
			name: s.patientId.split("-").pop(),
			wt: s.results!.wtVolume,
			tc: s.results!.tcVolume,
			et: s.results!.etVolume,
		}));

	const legend = [
		{ key: "WT", label: "Whole tumour", colour: overlayColors.wt },
		{ key: "TC", label: "Tumour core", colour: overlayColors.tc },
		{ key: "ET", label: "Enhancing", colour: overlayColors.et },
	];

	return (
		<Card className="animate-fade-in">
			<CardHeader className="flex-col items-start justify-center gap-0.5">
				<CardTitle>{APP_CONFIG.volumeChart.title}</CardTitle>
				<CardDescription>{APP_CONFIG.volumeChart.subtitle}</CardDescription>
			</CardHeader>
			<CardContent className="pt-4">
				{chartData.length === 0 ? (
					<div className="flex h-[220px] items-center justify-center rounded-md border border-dashed border-border px-6 text-center text-[13px] text-muted-foreground">
						Volumes appear here once a completed scan is flagged.
					</div>
				) : (
					<div className="h-[220px]">
						<ResponsiveContainer width="100%" height="100%">
							<BarChart
								data={chartData}
								barGap={2}
								barCategoryGap="28%"
								margin={{ top: 8, right: 4, left: 0, bottom: 0 }}
							>
								<CartesianGrid
									vertical={false}
									stroke="hsl(var(--border))"
									strokeDasharray="2 4"
								/>
								<XAxis
									dataKey="name"
									tick={TICK}
									axisLine={{ stroke: "hsl(var(--border))" }}
									tickLine={false}
									tickMargin={8}
								/>
								<YAxis
									tick={TICK}
									axisLine={false}
									tickLine={false}
									width={56}
									unit=" cm³"
								/>
								<Tooltip
									cursor={{ fill: "hsl(var(--muted))" }}
									contentStyle={{
										background: "hsl(var(--popover))",
										border: "1px solid hsl(var(--border))",
										borderRadius: "6px",
										fontSize: "12px",
										boxShadow: "0 12px 32px -8px hsl(220 30% 8% / 0.22)",
									}}
									labelStyle={{
										color: "hsl(var(--foreground))",
										fontFamily: "var(--font-mono)",
										marginBottom: 4,
									}}
									itemStyle={{ padding: 0 }}
								/>
								<Bar
									dataKey="wt"
									name="Whole Tumour"
									fill={overlayColors.wt}
									radius={[2, 2, 0, 0]}
								/>
								<Bar
									dataKey="tc"
									name="Tumour Core"
									fill={overlayColors.tc}
									radius={[2, 2, 0, 0]}
								/>
								<Bar
									dataKey="et"
									name="Enhancing"
									fill={overlayColors.et}
									radius={[2, 2, 0, 0]}
								/>
							</BarChart>
						</ResponsiveContainer>
					</div>
				)}
				<ul className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
					{legend.map((item) => (
						<li
							key={item.key}
							className="flex items-center gap-2 text-xs text-muted-foreground"
						>
							<span
								className="h-2 w-2 rounded-[2px]"
								style={{ background: item.colour }}
							/>
							<span className="font-mono text-foreground">{item.key}</span>
							{item.label}
						</li>
					))}
				</ul>
			</CardContent>
		</Card>
	);
};
