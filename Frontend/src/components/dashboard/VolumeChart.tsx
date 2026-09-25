import {
	BarChart,
	Bar,
	XAxis,
	YAxis,
	Tooltip,
	ResponsiveContainer,
	CartesianGrid,
} from "recharts";
import { useScans } from "@/hooks/use-scans";
import { useOverlayColors } from "@/hooks/use-overlay-colors";
import { APP_CONFIG } from "@/lib/config";

export const VolumeChart = () => {
	const { data: scans, isLoading } = useScans();
	const { colors: overlayColors } = useOverlayColors();

	if (isLoading || !scans) {
		return (
			<div className="h-[320px] rounded-xl border border-border bg-card p-6" />
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

	return (
		<div className="animate-fade-in rounded-xl border border-border bg-card p-6">
			<h2 className="mb-1 text-sm font-semibold text-foreground">
				{APP_CONFIG.volumeChart.title}
			</h2>
			<p className="mb-6 text-xs text-muted-foreground">
				{APP_CONFIG.volumeChart.subtitle}
			</p>
			<div className="h-[220px]">
				<ResponsiveContainer width="100%" height="100%">
					<BarChart
						data={chartData}
						barGap={2}
						margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
					>
						<CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
						<XAxis
							dataKey="name"
							tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
							axisLine={{ stroke: "hsl(var(--border))" }}
							tickLine={false}
						/>
						<YAxis
							tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
							axisLine={false}
							tickLine={false}
							unit=" cm³"
						/>
						<Tooltip
							contentStyle={{
								background: "hsl(var(--card))",
								border: "1px solid hsl(var(--border))",
								borderRadius: "8px",
								fontSize: "12px",
								boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
							}}
						/>
						<Bar
							dataKey="wt"
							name="Whole Tumour"
							fill={overlayColors.wt}
							radius={[3, 3, 0, 0]}
						/>
						<Bar
							dataKey="tc"
							name="Tumour Core"
							fill={overlayColors.tc}
							radius={[3, 3, 0, 0]}
						/>
						<Bar
							dataKey="et"
							name="Enhancing"
							fill={overlayColors.et}
							radius={[3, 3, 0, 0]}
						/>
					</BarChart>
				</ResponsiveContainer>
			</div>
			<div className="mt-4 flex items-center gap-5">
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div
						className="h-2.5 w-2.5 rounded-sm"
						style={{ background: overlayColors.wt }}
					/>
					WT
				</div>
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div
						className="h-2.5 w-2.5 rounded-sm"
						style={{ background: overlayColors.tc }}
					/>
					TC
				</div>
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div
						className="h-2.5 w-2.5 rounded-sm"
						style={{ background: overlayColors.et }}
					/>
					ET
				</div>
			</div>
		</div>
	);
};
