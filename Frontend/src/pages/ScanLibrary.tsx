import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Pencil, Search, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { matchesSearch, resultLabel } from "@/lib/scan-format";
import type { MRIScan } from "@/lib/types";

const headClass =
	"px-6 py-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground";
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
				<div className="h-20 w-1/3 rounded-xl bg-muted" />
				<div className="grid grid-cols-3 gap-4">
					{[1, 2, 3].map((i) => (
						<div key={i} className="h-48 rounded-xl bg-muted" />
					))}
				</div>
			</div>
		);
	}

	const filtered = scans.filter((scan) => matchesSearch(scan, search));

	return (
		<div className="space-y-6">
			<div>
				<h1 className="text-2xl font-bold tracking-tight text-foreground">
					{config.title}
				</h1>
				<p className="mt-1 text-sm text-muted-foreground">{config.subtitle}</p>
			</div>

			<div className="relative w-72">
				<Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
				<Input
					aria-label="Search scans"
					placeholder={config.searchPlaceholder}
					value={search}
					onChange={(e) => setSearch(e.target.value)}
					className="h-9 border-none bg-muted/50 pl-9 text-sm"
				/>
			</div>

			<div className="overflow-x-auto">
				<Table>
					<TableHeader>
						<TableRow className="hover:bg-transparent">
							<TableHead className={headClass}>Patient</TableHead>
							<TableHead className={headClass}>Modalities</TableHead>
							<TableHead className={headClass}>Date</TableHead>
							<TableHead className={headClass}>Result</TableHead>
							<TableHead className={headClass}>Edit</TableHead>
							<TableHead className={headClass}>Delete</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{filtered.map((scan) => (
							<TableRow
								key={scan.id}
								className="group cursor-pointer hover:bg-muted/50"
								onClick={() => navigate(`/scan/${scan.id}`)}
							>
								<PatientCell scan={scan} />
								<TableCell className="px-6 py-4">
									<ModalityBadges modalities={scan.modalities} />
								</TableCell>
								<TableCell className="px-6 py-4 text-sm text-muted-foreground">
									{scan.scanDate}
								</TableCell>
								<TableCell className="px-6 py-4 text-sm text-muted-foreground">
									{resultLabel(scan)}
								</TableCell>
								<TableCell className="px-6 py-4">
									<Button
										variant="ghost"
										size="sm"
										className="text-xs text-muted-foreground"
										onClick={(e) => {
											e.stopPropagation();
											setEditingScan(scan);
										}}
									>
										<Pencil className="mr-1 h-3.5 w-3.5" />
										Edit
									</Button>
								</TableCell>
								<TableCell className="px-6 py-4">
									<Button
										variant="ghost"
										size="sm"
										className="text-xs text-destructive hover:text-destructive"
										onClick={(e) => {
											e.stopPropagation();
											setDeletingScan(scan);
										}}
									>
										<Trash2 className="mr-1 h-3.5 w-3.5" />
										Delete
									</Button>
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</div>

			{filtered.length === 0 && (
				<div className="py-16 text-center text-muted-foreground">
					<p className="text-sm font-medium">{config.noScansFound}</p>
					<p className="mt-1 text-xs">{config.adjustFilters}</p>
				</div>
			)}

			<EditScanDialog scan={editingScan} onClose={() => setEditingScan(null)} />
			<DeleteScanDialog
				scan={deletingScan}
				onClose={() => setDeletingScan(null)}
			/>
		</div>
	);
};

export default ScanLibrary;
