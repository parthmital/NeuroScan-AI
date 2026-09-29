import { isInProgress, resultLabel } from "@/lib/scan-format";
import type { MRIScan } from "@/lib/types";
import { cn } from "@/lib/utils";

type Tone = "flagged" | "clear" | "progress" | "failed" | "none";

const toneOf = (scan: MRIScan): Tone => {
	if (scan.status === "failed") return "failed";
	if (isInProgress(scan)) return "progress";
	if (!scan.results) return "none";
	return scan.results.detected ? "flagged" : "clear";
};

const DOT: Record<Tone, string> = {
	flagged: "bg-warning",
	clear: "bg-success",
	progress: "bg-medical animate-pulse",
	failed: "bg-destructive",
	none: "bg-muted-foreground/40",
};

const TEXT: Record<Tone, string> = {
	flagged: "text-foreground",
	clear: "text-foreground",
	progress: "text-medical",
	failed: "text-destructive",
	none: "text-muted-foreground",
};

/** Scan outcome as a status dot and label, coloured by clinical state. */
export const ResultStatus = ({
	scan,
	className,
}: {
	scan: MRIScan;
	className?: string;
}) => {
	const tone = toneOf(scan);
	return (
		<span
			className={cn(
				"inline-flex items-center gap-2 text-[13px] font-medium",
				TEXT[tone],
				className,
			)}
		>
			<span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", DOT[tone])} />
			{resultLabel(scan)}
		</span>
	);
};
