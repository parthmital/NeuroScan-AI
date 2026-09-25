import { TableCell } from "@/components/ui/table";
import type { MRIScan } from "@/lib/types";

export const PatientCell = ({ scan }: { scan: MRIScan }) => (
	<TableCell className="px-6 py-4">
		<p className="text-sm font-medium text-foreground">{scan.patientName}</p>
		<p className="text-xs text-muted-foreground">{scan.patientId}</p>
	</TableCell>
);
