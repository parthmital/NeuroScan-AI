import {
	Brain,
	Loader2,
	CheckCircle2,
	AlertTriangle,
	HelpCircle,
	LucideIcon,
} from "lucide-react";
import { useScans } from "@/hooks/use-data";
import { cn } from "@/lib/utils";

const iconMap: Record<string, LucideIcon> = {
	total: Brain,
	processing: Loader2,
	completed: CheckCircle2,
	flagged: AlertTriangle,
};

const colorMap: Record<string, string> = {
	total: "text-medical bg-medical-light",
	processing: "text-info bg-info/10",
	completed: "text-success bg-success/10",
	flagged: "text-warning bg-warning/10",
};

export const StatsCards = () => {
	const { data: scans, isLoading } = useScans();

	const stats = scans
		? [
				{ label: "Total Scans", value: scans.length, type: "total" },
				{
					label: "Processing",
					value: 0,
					type: "processing",
				},
				{
					label: "Completed",
					value: scans.length,
					type: "completed",
				},
				{
					label: "Flagged",
					value: scans.filter((s) => s.results?.detected).length,
					type: "flagged",
				},
			]
		: null;

	if (isLoading || !stats) {
		return (
			<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
				{[1, 2, 3, 4].map((i) => (
					<div
						key={i}
						className="bg-card rounded-xl border border-border p-5 h-32 animate-pulse"
					/>
				))}
			</div>
		);
	}

	return (
		<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
			{stats.map((stat, index) => {
				const Icon = iconMap[stat.type] || HelpCircle;
				return (
					<div
						key={stat.label}
						className={cn("bg-card rounded-xl border border-border p-5")}
					>
						<div className="flex items-center justify-between mb-3">
							<span className="text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								{stat.label}
							</span>
							<div
								className={cn(
									"w-9 h-9 rounded-lg flex items-center justify-center",
									colorMap[stat.type] || "bg-muted text-muted-foreground",
								)}
							>
								<Icon className={cn("w-4.5 h-4.5")} />
							</div>
						</div>
						<p className="text-3xl font-bold text-foreground tracking-tight">
							{stat.value}
						</p>
					</div>
				);
			})}
		</div>
	);
};
