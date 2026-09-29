import { useScans } from "@/hooks/use-scans";
import { isInProgress } from "@/lib/scan-format";
import { cn } from "@/lib/utils";

interface Stat {
	label: string;
	value: number;
	/** Status dot matching the result colours used in scan tables. */
	dot?: string;
}

const cellClass = "bg-card px-5 py-4";

export const StatsCards = () => {
	const { data: scans, isLoading } = useScans();

	if (isLoading || !scans) {
		return (
			<div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border lg:grid-cols-4">
				{[1, 2, 3, 4].map((i) => (
					<div key={i} className={cn(cellClass, "h-[92px]")}>
						<div className="h-3 w-20 animate-pulse rounded bg-muted" />
						<div className="mt-4 h-7 w-12 animate-pulse rounded bg-muted" />
					</div>
				))}
			</div>
		);
	}

	const stats: Stat[] = [
		{ label: "Total scans", value: scans.length },
		{
			label: "Processing",
			value: scans.filter(isInProgress).length,
			dot: "bg-medical",
		},
		{
			label: "Completed",
			value: scans.filter((s) => s.status === "completed").length,
			dot: "bg-success",
		},
		{
			label: "Flagged",
			value: scans.filter((s) => s.results?.detected).length,
			dot: "bg-warning",
		},
	];

	return (
		<dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border lg:grid-cols-4">
			{stats.map(({ label, value, dot }) => (
				<div key={label} className={cellClass}>
					<dt className="flex items-center gap-2 text-[13px] text-muted-foreground">
						{dot && <span className={cn("h-1.5 w-1.5 rounded-full", dot)} />}
						{label}
					</dt>
					<dd className="mt-2 font-mono text-[28px] font-medium leading-9 tracking-[-0.02em] text-foreground tabular-nums">
						{value}
					</dd>
				</div>
			))}
		</dl>
	);
};
