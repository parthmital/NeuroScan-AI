import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { scansApi } from "@/lib/api";
import { isInProgress } from "@/lib/scan-format";
import type { MRIScan } from "@/lib/types";

const SCANS_KEY = ["scans"] as const;
const POLL_MS = 2000;

export const useScans = () =>
	useQuery<MRIScan[]>({
		queryKey: SCANS_KEY,
		queryFn: scansApi.list,
		refetchInterval: (query) =>
			query.state.data?.some(isInProgress) ? POLL_MS : false,
	});

export const useScan = (id: string | undefined) =>
	useQuery<MRIScan>({
		queryKey: [...SCANS_KEY, id],
		queryFn: () => scansApi.get(id as string),
		enabled: Boolean(id),
		refetchInterval: (query) =>
			query.state.data && isInProgress(query.state.data) ? POLL_MS : false,
	});

const useScanMutation = <TArgs, TResult>(
	mutationFn: (args: TArgs) => Promise<TResult>,
) => {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn,
		onSuccess: () => queryClient.invalidateQueries({ queryKey: SCANS_KEY }),
	});
};

export const useUpdateScan = () =>
	useScanMutation(
		({
			id,
			...data
		}: {
			id: string;
			patientName?: string;
			patientId?: string;
		}) => scansApi.update(id, data),
	);

export const useDeleteScan = () => useScanMutation(scansApi.remove);

export const useRetryScan = () => useScanMutation(scansApi.retry);

export const useUploadScan = () =>
	useScanMutation(
		({
			files,
			patientName,
			patientId,
		}: {
			files: File[];
			patientName: string;
			patientId: string;
		}) => scansApi.upload(files, patientName, patientId),
	);
