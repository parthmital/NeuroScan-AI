import { useState } from "react";
import {
	BarChart,
	Bar,
	XAxis,
	YAxis,
	Tooltip,
	ResponsiveContainer,
	CartesianGrid,
} from "recharts";
import { useScans } from "@/hooks/use-data";
import { useConfig } from "@/hooks/use-config";
import { MRIScan, AppConfig } from "@/lib/types";

export const VolumeChart = () => {
	const { data: scans, isLoading: scansLoading } = useScans();
	const { data: config, isLoading: configLoading } = useConfig();
	const [overlayColors] = useState(() => {
		const saved = localStorage.getItem("overlayColors");
		return saved
			? JSON.parse(saved)
			: {
					wt: "#eab308",
					tc: "#ef4444",
					et: "#3b82f6",
				};
	});

	if (scansLoading || configLoading || !scans || !config) {
		return (
			<div className="bg-card rounded-xl border border-border p-6 h-[320px]" />
		);
	}

	const chartData = (scans as MRIScan[])
		.filter((s) => s.results?.detected)
		.map((s) => ({
			name: s.patientId.split("-").pop(),
			wt: s.results!.wtVolume,
			tc: s.results!.tcVolume,
			et: s.results!.etVolume,
		}));

	return (
		<div className="bg-card rounded-xl border border-border p-6 animate-fade-in">
			<h2 className="text-sm font-semibold text-foreground mb-1">
				{(config as AppConfig).volumeChart.title}
			</h2>
			<p className="text-xs text-muted-foreground mb-6">
				{(config as AppConfig).volumeChart.subtitle}
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
			<div className="flex items-center gap-5 mt-4">
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div
						className="w-2.5 h-2.5 rounded-sm"
						style={{ background: overlayColors.wt }}
					/>
					WT
				</div>
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div
						className="w-2.5 h-2.5 rounded-sm"
						style={{ background: overlayColors.tc }}
					/>
					TC
				</div>
				<div className="flex items-center gap-2 text-xs text-muted-foreground">
					<div
						className="w-2.5 h-2.5 rounded-sm"
						style={{ background: overlayColors.et }}
					/>
					ET
				</div>
			</div>
		</div>
	);
};
