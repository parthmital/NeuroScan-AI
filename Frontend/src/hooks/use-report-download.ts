import { useState } from "react";
import { toast } from "sonner";
import { reportsApi } from "@/lib/api";

type DownloadKind = "pdf" | "csv" | "all";

/** Report downloads with a pending state per kind and error toasts. */
export const useReportDownload = () => {
	const [pending, setPending] = useState<DownloadKind | null>(null);

	const run = async (kind: DownloadKind, action: () => Promise<void>) => {
		setPending(kind);
		try {
			await action();
		} catch (error) {
			toast.error(
				error instanceof Error ? error.message : "Download failed. Try again.",
			);
		} finally {
			setPending(null);
		}
	};

	return {
		pending,
		downloadPdf: (id: string) => run("pdf", () => reportsApi.pdf(id)),
		downloadCsv: (id: string) => run("csv", () => reportsApi.csv(id)),
		exportAll: () => run("all", reportsApi.exportAll),
	};
};
