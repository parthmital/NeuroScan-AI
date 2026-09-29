import { TableCell } from "@/components/ui/table";
import type { MRIScan } from "@/lib/types";

export const PatientCell = ({ scan }: { scan: MRIScan }) => (
	<TableCell className="pl-5">
		<p className="max-w-[240px] truncate text-[13px] font-medium text-foreground">
			{scan.patientName}
		</p>
		<p className="font-mono text-[11px] text-muted-foreground">
			{scan.patientId}
		</p>
	</TableCell>
);
