import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Eye, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
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
import { ResultStatus } from "@/components/scan/ResultStatus";
import {
	DeleteScanDialog,
	EditScanDialog,
} from "@/components/scan/ScanDialogs";
import { useScans } from "@/hooks/use-scans";
import { APP_CONFIG } from "@/lib/config";
import type { MRIScan } from "@/lib/types";

const RECENT_COUNT = 6;

export const RecentScans = () => {
	const navigate = useNavigate();
	const { data: scans, isLoading } = useScans();
	const [editingScan, setEditingScan] = useState<MRIScan | null>(null);
	const [deletingScan, setDeletingScan] = useState<MRIScan | null>(null);

	if (isLoading || !scans) {
		return (
			<div className="h-80 animate-pulse rounded-lg border border-border bg-card" />
		);
	}

	return (
		<Card className="overflow-hidden">
			<CardHeader className="pr-2.5">
				<CardTitle>{APP_CONFIG.dashboard.recentScansTitle}</CardTitle>
				<Button variant="ghost" size="sm" onClick={() => navigate("/scans")}>
					{APP_CONFIG.dashboard.viewAllLabel}
					<ArrowRight />
				</Button>
			</CardHeader>

			<div className="overflow-x-auto">
				<Table>
					<TableHeader>
						<TableRow className="hover:bg-transparent">
							<TableHead className="pl-5">Patient</TableHead>
							<TableHead className="hidden sm:table-cell">Date</TableHead>
							<TableHead className="hidden md:table-cell">Modalities</TableHead>
							<TableHead>Result</TableHead>
							<TableHead className="w-14 pr-3">
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
								<TableCell className="hidden whitespace-nowrap font-mono text-xs sm:table-cell text-muted-foreground tabular-nums">
									{scan.scanDate}
								</TableCell>
								<TableCell className="hidden md:table-cell">
									<ModalityBadges modalities={scan.modalities} />
								</TableCell>
								<TableCell className="whitespace-nowrap">
									<ResultStatus scan={scan} />
								</TableCell>
								<TableCell className="pr-3 text-right">
									<DropdownMenu>
										<DropdownMenuTrigger asChild>
											<button
												aria-label={`Actions for ${scan.patientName}`}
												onClick={(e) => e.stopPropagation()}
												className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
											>
												<MoreHorizontal className="h-4 w-4" />
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
					<div className="px-6 py-14 text-center">
						<p className="text-sm font-medium text-foreground">No scans yet</p>
						<p className="mx-auto mt-1 max-w-xs text-[13px] text-muted-foreground">
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
		</Card>
	);
};
