import { useState } from "react";
import { toast } from "sonner";
import { reportsApi } from "@/lib/api";

type DownloadKind = "csv" | "all";

/** Report downloads with a pending state per kind (and scan) and error toasts; PDFs open in a tab. */
export const useReportDownload = () => {
	const [pending, setPending] = useState<DownloadKind | null>(null);
	const [pendingId, setPendingId] = useState<string | null>(null);

	const run = async (
		kind: DownloadKind,
		action: () => Promise<void>,
		id: string | null = null,
	) => {
		setPending(kind);
		setPendingId(id);
		try {
			await action();
		} catch (error) {
			toast.error(
				error instanceof Error ? error.message : "Download failed. Try again.",
			);
		} finally {
			setPending(null);
			setPendingId(null);
		}
	};

	return {
		pending,
		pendingId,
		openPdf: reportsApi.openPdf,
		downloadCsv: (id: string) => run("csv", () => reportsApi.csv(id), id),
		exportAll: () => run("all", reportsApi.exportAll),
	};
};
