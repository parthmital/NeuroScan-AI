import { useQuery } from "@tanstack/react-query";
import { fetchScans } from "@/lib/api";
import { MRIScan } from "@/lib/types";

export const useScans = () => {
	return useQuery<MRIScan[]>({
		queryKey: ["scans"],
		queryFn: fetchScans,
	});
};
