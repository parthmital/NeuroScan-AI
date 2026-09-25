import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { ModalityBadges } from "@/components/scan/ModalityBadges";
import { PatientCell } from "@/components/scan/PatientCell";
import {
	DeleteScanDialog,
	EditScanDialog,
} from "@/components/scan/ScanDialogs";
import { useScans } from "@/hooks/use-scans";
import { APP_CONFIG } from "@/lib/config";
import { resultLabel } from "@/lib/scan-format";
import type { MRIScan } from "@/lib/types";

const RECENT_COUNT = 6;
const headClass =
	"px-6 py-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground";

export const RecentScans = () => {
	const navigate = useNavigate();
	const { data: scans, isLoading } = useScans();
	const [editingScan, setEditingScan] = useState<MRIScan | null>(null);
	const [deletingScan, setDeletingScan] = useState<MRIScan | null>(null);

	if (isLoading || !scans) {
		return <div className="h-64 rounded-xl border border-border bg-card" />;
	}

	return (
		<div className="overflow-hidden rounded-xl border border-border bg-card">
			<div className="flex items-center justify-between border-b border-border px-6 py-4">
				<h2 className="text-sm font-semibold text-foreground">
					{APP_CONFIG.dashboard.recentScansTitle}
				</h2>
				<Button
					variant="ghost"
					size="sm"
					onClick={() => navigate("/scans")}
					className="text-xs text-muted-foreground"
				>
					{APP_CONFIG.dashboard.viewAllLabel}
				</Button>
			</div>

			<div className="overflow-x-auto">
				<Table>
					<TableHeader>
						<TableRow className="hover:bg-transparent">
							<TableHead className={headClass}>Patient</TableHead>
							<TableHead className={headClass}>Date</TableHead>
							<TableHead className={headClass}>Modalities</TableHead>
							<TableHead className={headClass}>Result</TableHead>
							<TableHead className="w-[80px] px-6 py-4">
								<span className="sr-only">Actions</span>
							</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{scans.slice(0, RECENT_COUNT).map((scan) => (
							<TableRow
								key={scan.id}
								className="group cursor-pointer"
								onClick={() => navigate(`/scan/${scan.id}`)}
							>
								<PatientCell scan={scan} />
								<TableCell className="px-6 py-4 text-sm text-muted-foreground">
									{scan.scanDate}
								</TableCell>
								<TableCell className="px-6 py-4">
									<ModalityBadges modalities={scan.modalities} />
								</TableCell>
								<TableCell className="px-6 py-4 text-sm text-muted-foreground">
									{resultLabel(scan)}
								</TableCell>
								<TableCell className="px-6 py-4">
									<DropdownMenu>
										<DropdownMenuTrigger asChild>
											<button
												aria-label={`Actions for ${scan.patientName}`}
												onClick={(e) => e.stopPropagation()}
												className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
											>
												<MoreHorizontal className="h-3.5 w-3.5" />
											</button>
										</DropdownMenuTrigger>
										<DropdownMenuContent
											align="end"
											onClick={(e) => e.stopPropagation()}
										>
											<DropdownMenuItem
												onClick={() => navigate(`/scan/${scan.id}`)}
											>
												<Eye className="mr-2 h-3.5 w-3.5" />
												View Details
											</DropdownMenuItem>
											<DropdownMenuItem onClick={() => setEditingScan(scan)}>
												<Pencil className="mr-2 h-3.5 w-3.5" />
												Edit Details
											</DropdownMenuItem>
											<DropdownMenuItem
												className="text-destructive focus:text-destructive"
												onClick={() => setDeletingScan(scan)}
											>
												<Trash2 className="mr-2 h-3.5 w-3.5" />
												Delete Scan
											</DropdownMenuItem>
										</DropdownMenuContent>
									</DropdownMenu>
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
				{scans.length === 0 && (
					<div className="px-6 py-12 text-center">
						<p className="text-sm font-medium text-foreground">No scans yet</p>
						<p className="mt-1 text-xs text-muted-foreground">
							Upload an MRI study to run detection, classification, and
							segmentation.
						</p>
					</div>
				)}
			</div>

			<EditScanDialog scan={editingScan} onClose={() => setEditingScan(null)} />
			<DeleteScanDialog
				scan={deletingScan}
				onClose={() => setDeletingScan(null)}
			/>
		</div>
	);
};
