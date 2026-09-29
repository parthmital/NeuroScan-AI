import { useQuery } from "@tanstack/react-query";
import { scansApi } from "@/lib/api";
import { parseLabels, parseVolume } from "./volume";

// A BraTS volume is about 36 MB once parsed; drop unused ones soon after they go off screen.
const GC_MS = 2 * 60 * 1000;

/** Downloads and parses one modality of a scan; switching back to it is instant. */
export const useVolume = (scanId: string, modality: string) =>
	useQuery({
		queryKey: ["volume", scanId, modality.toLowerCase()],
		queryFn: async () => parseVolume(await scansApi.volume(scanId, modality)),
		staleTime: Infinity,
		gcTime: GC_MS,
		retry: 1,
	});

export const useLabels = (scanId: string, enabled: boolean) =>
	useQuery({
		queryKey: ["volume", scanId, "segmentation"],
		queryFn: async () =>
			parseLabels(await scansApi.volume(scanId, "segmentation")),
		enabled,
		staleTime: Infinity,
		gcTime: GC_MS,
		retry: 1,
	});
