import {
	AlertTriangle,
	Brain,
	CheckCircle2,
	Loader2,
	type LucideIcon,
} from "lucide-react";
import { useScans } from "@/hooks/use-scans";
import { isInProgress } from "@/lib/scan-format";
import { cn } from "@/lib/utils";

interface Stat {
	label: string;
	value: number;
	icon: LucideIcon;
	colour: string;
}

export const StatsCards = () => {
	const { data: scans, isLoading } = useScans();

	if (isLoading || !scans) {
		return (
			<div className="grid grid-cols-4 gap-5">
				{[1, 2, 3, 4].map((i) => (
					<div
						key={i}
						className="h-32 animate-pulse rounded-xl border border-border bg-card p-5"
					/>
				))}
			</div>
		);
	}

	const stats: Stat[] = [
		{
			label: "Total Scans",
			value: scans.length,
			icon: Brain,
			colour: "text-medical bg-medical-light",
		},
		{
			label: "Processing",
			value: scans.filter(isInProgress).length,
			icon: Loader2,
			colour: "text-info bg-info/10",
		},
		{
			label: "Completed",
			value: scans.filter((s) => s.status === "completed").length,
			icon: CheckCircle2,
			colour: "text-success bg-success/10",
		},
		{
			label: "Flagged",
			value: scans.filter((s) => s.results?.detected).length,
			icon: AlertTriangle,
			colour: "text-warning bg-warning/10",
		},
	];

	return (
		<div className="grid grid-cols-4 gap-5">
			{stats.map(({ label, value, icon: Icon, colour }) => (
				<div
					key={label}
					className="rounded-xl border border-border bg-card p-5"
				>
					<div className="mb-3 flex items-center justify-between">
						<span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
							{label}
						</span>
						<div
							className={cn(
								"flex h-9 w-9 items-center justify-center rounded-lg",
								colour,
							)}
						>
							<Icon className="h-4 w-4" />
						</div>
					</div>
					<p className="text-3xl font-bold tracking-tight text-foreground">
						{value}
					</p>
				</div>
			))}
		</div>
	);
};
