import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Pencil, Search, Trash2 } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { ModalityBadges } from "@/components/scan/ModalityBadges";
import { PatientCell } from "@/components/scan/PatientCell";
import { ResultStatus } from "@/components/scan/ResultStatus";
import {
	DeleteScanDialog,
	EditScanDialog,
} from "@/components/scan/ScanDialogs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { useScans } from "@/hooks/use-scans";
import { APP_CONFIG } from "@/lib/config";
import { matchesSearch } from "@/lib/scan-format";
import type { MRIScan } from "@/lib/types";

const config = APP_CONFIG.scanLibrary;

const ScanLibrary = () => {
	const navigate = useNavigate();
	const location = useLocation();
	const querySearch = new URLSearchParams(location.search).get("search") || "";
	const [search, setSearch] = useState(querySearch);
	const [syncedQuery, setSyncedQuery] = useState(querySearch);
	const [editingScan, setEditingScan] = useState<MRIScan | null>(null);
	const [deletingScan, setDeletingScan] = useState<MRIScan | null>(null);
	const { data: scans, isLoading } = useScans();

	// A new ?search= value replaces whatever was typed.
	if (querySearch !== syncedQuery) {
		setSyncedQuery(querySearch);
		setSearch(querySearch);
	}

	if (isLoading || !scans) {
		return (
			<div className="animate-pulse space-y-6">
				<div className="h-14 w-1/3 rounded-lg bg-muted" />
				<div className="h-9 w-72 rounded-md bg-muted" />
				<div className="h-96 rounded-lg bg-muted" />
			</div>
		);
	}

	const filtered = scans.filter((scan) => matchesSearch(scan, search));

	return (
		<div className="space-y-6">
			<PageHeader title={config.title} description={config.subtitle} />

			<Card className="overflow-hidden">
				<div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
					<div className="relative w-full sm:w-72">
						<Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
						<Input
							type="search"
							aria-label="Search scans"
							placeholder={config.searchPlaceholder}
							value={search}
							onChange={(e) => setSearch(e.target.value)}
							className="h-8 pl-8 text-[13px]"
						/>
					</div>
					<p className="font-mono text-xs text-muted-foreground tabular-nums">
						{filtered.length} of {scans.length} scans
					</p>
				</div>
				<Table>
					<TableHeader>
						<TableRow className="hover:bg-transparent">
							<TableHead className="pl-5">Patient</TableHead>
							<TableHead className="hidden md:table-cell">Modalities</TableHead>
							<TableHead className="hidden sm:table-cell">Date</TableHead>
							<TableHead>Result</TableHead>
							<TableHead className="pr-3 text-right">
								<span className="sr-only">Actions</span>
							</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{filtered.map((scan) => (
							<TableRow
								key={scan.id}
								className="cursor-pointer"
								onClick={() => navigate(`/scan/${scan.id}`)}
							>
								<PatientCell scan={scan} />
								<TableCell className="hidden md:table-cell">
									<ModalityBadges modalities={scan.modalities} />
								</TableCell>
								<TableCell className="hidden whitespace-nowrap font-mono text-xs sm:table-cell text-muted-foreground tabular-nums">
									{scan.scanDate}
								</TableCell>
								<TableCell className="whitespace-nowrap">
									<ResultStatus scan={scan} />
								</TableCell>
								<TableCell className="pr-3">
									<div className="flex justify-end gap-1">
										<Button
											variant="ghost"
											size="sm"
											aria-label={`Edit ${scan.patientName}`}
											onClick={(e) => {
												e.stopPropagation();
												setEditingScan(scan);
											}}
										>
											<Pencil />
											<span className="hidden sm:inline">Edit</span>
										</Button>
										<Button
											variant="ghost"
											size="sm"
											aria-label={`Delete ${scan.patientName}`}
											className="hover:bg-destructive/10 hover:text-destructive"
											onClick={(e) => {
												e.stopPropagation();
												setDeletingScan(scan);
											}}
										>
											<Trash2 />
											<span className="hidden sm:inline">Delete</span>
										</Button>
									</div>
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>

				{filtered.length === 0 && (
					<div className="px-6 py-16 text-center">
						<p className="text-sm font-medium text-foreground">
							{config.noScansFound}
						</p>
						<p className="mt-1 text-[13px] text-muted-foreground">
							{config.adjustFilters}
						</p>
					</div>
				)}
			</Card>

			<EditScanDialog scan={editingScan} onClose={() => setEditingScan(null)} />
			<DeleteScanDialog
				scan={deletingScan}
				onClose={() => setDeletingScan(null)}
			/>
		</div>
	);
};

export default ScanLibrary;
