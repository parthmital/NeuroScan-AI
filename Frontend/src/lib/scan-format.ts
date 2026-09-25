import type { MRIScan } from "./types";

export const isInProgress = (scan: Pick<MRIScan, "status">) =>
	scan.status === "queued" || scan.status === "processing";

/** Short result label used in scan tables and report lists. */
export const resultLabel = (scan: MRIScan): string => {
	if (scan.status === "failed") return "Failed";
	if (isInProgress(scan)) return "Processing";
	if (!scan.results) return "—";
	return scan.results.detected ? scan.results.classification : "No Tumour";
};

export const formatPercent = (fraction: number) =>
	`${(fraction * 100).toFixed(1)}%`;

/** Case-insensitive match on patient name or patient ID. */
export const matchesSearch = (scan: MRIScan, search: string) => {
	const term = search.trim().toLowerCase();
	return (
		scan.patientName.toLowerCase().includes(term) ||
		scan.patientId.toLowerCase().includes(term)
	);
};
