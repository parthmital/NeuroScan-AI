import { useNavigate, useLocation } from "react-router-dom";
import { Search, Eye, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useScans } from "@/hooks/use-data";
import { useConfig } from "@/hooks/use-config";
import { MRIScan, AppConfig } from "@/lib/types";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";

const ScanLibrary = () => {
	const navigate = useNavigate();
	const location = useLocation();
	const querySearch = new URLSearchParams(location.search).get("search") || "";
	const [search, setSearch] = useState(querySearch);
	const [filter, setFilter] = useState<string>("all");
	const [editingScan, setEditingScan] = useState<MRIScan | null>(null);
	const [editForm, setEditForm] = useState({ patientName: "", patientId: "" });
	const { data: scans, isLoading: scansLoading } = useScans();
	const { data: config, isLoading: configLoading } = useConfig();

	const handleEditClick = (scan: MRIScan) => {
		setEditingScan(scan);
		setEditForm({ patientName: scan.patientName, patientId: scan.patientId });
	};

	const handleSaveScan = async () => {
		if (!editingScan) return;
		try {
			const token = localStorage.getItem("token");
			await fetch(`http://localhost:8000/api/scans/${editingScan.id}`, {
				method: "PUT",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${token}`,
				},
				body: JSON.stringify(editForm),
			});
			setEditingScan(null);
			window.location.reload();
		} catch (error) {
			console.error("Failed to update scan", error);
		}
	};

	useEffect(() => {
		setSearch(querySearch);
	}, [querySearch]);

	if (scansLoading || configLoading || !scans || !config) {
		return (
			<div className="space-y-6">
				<div className="h-20 bg-muted rounded-xl w-1/3" />
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{[1, 2, 3].map((i) => (
						<div key={i} className="h-48 bg-muted rounded-xl" />
					))}
				</div>
			</div>
		);
	}

	// Filter scans based on search
	const filtered = (scans || []).filter((s) => {
		const matchesSearch =
			s.patientName.toLowerCase().includes(search.toLowerCase()) ||
			s.patientId.toLowerCase().includes(search.toLowerCase());
		return matchesSearch;
	});

	return (
		<div className="space-y-6">
			<div>
				<h1 className="text-2xl font-bold text-foreground tracking-tight">
					{config.scanLibrary.title}
				</h1>
				<p className="text-sm text-muted-foreground mt-1">
					{config.scanLibrary.subtitle}
				</p>
			</div>

			{/* Search */}
			<div className="flex items-center gap-4">
				<div className="relative w-72">
					<Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
					<Input
						placeholder={config.scanLibrary.searchPlaceholder}
						value={search}
						onChange={(e) => setSearch(e.target.value)}
						className="pl-9 h-9 bg-muted/50 border-none text-sm"
					/>
				</div>
			</div>

			<Dialog
				open={!!editingScan}
				onOpenChange={(open) => !open && setEditingScan(null)}
			>
				<DialogContent onClick={(e) => e.stopPropagation()}>
					<DialogHeader>
						<DialogTitle>Edit Scan Details</DialogTitle>
						<DialogDescription>
							Update patient information for this scan.
						</DialogDescription>
					</DialogHeader>
					<div className="space-y-4 py-2">
						<div className="space-y-2">
							<Label htmlFor="patientName">Patient Name</Label>
							<Input
								id="patientName"
								value={editForm.patientName}
								onChange={(e) =>
									setEditForm({ ...editForm, patientName: e.target.value })
								}
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="patientId">Patient ID</Label>
							<Input
								id="patientId"
								value={editForm.patientId}
								onChange={(e) =>
									setEditForm({ ...editForm, patientId: e.target.value })
								}
							/>
						</div>
					</div>
					<DialogFooter>
						<Button variant="outline" onClick={() => setEditingScan(null)}>
							Cancel
						</Button>
						<Button onClick={handleSaveScan} className="bg-medical text-white">
							Save Changes
						</Button>
					</DialogFooter>
				</DialogContent>
			</Dialog>

			{/* Desktop table view */}
			<div className="overflow-x-auto">
				<Table>
					<TableHeader>
						<TableRow className="hover:bg-transparent">
							<TableHead className="px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Patient
							</TableHead>
							<TableHead className="px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Modalities
							</TableHead>
							<TableHead className="px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Date
							</TableHead>
							<TableHead className="px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Result
							</TableHead>

							<TableHead className="px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Edit
							</TableHead>
							<TableHead className="px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Delete
							</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{filtered.map((scan: MRIScan) => {
							return (
								<TableRow
									key={scan.id}
									className="cursor-pointer group hover:bg-muted/50"
									onClick={() => navigate(`/scan/${scan.id}`)}
								>
									<TableCell className="px-6 py-4">
										<div>
											<p className="text-sm font-medium text-foreground">
												{scan.patientName}
											</p>
											<p className="text-xs text-muted-foreground">
												{scan.patientId}
											</p>
										</div>
									</TableCell>
									<TableCell className="px-6 py-4">
										<div className="flex gap-1">
											{scan.modalities.map((m) => (
												<span
													key={m}
													className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-muted text-muted-foreground"
												>
													{m}
												</span>
											))}
										</div>
									</TableCell>
									<TableCell className="px-6 py-4 text-sm text-muted-foreground">
										{scan.scanDate}
									</TableCell>
									<TableCell className="px-6 py-4">
										<span className="text-sm text-muted-foreground">
											{scan.results
												? scan.results.detected
													? scan.results.classification
													: "No Tumour"
												: "—"}
										</span>
									</TableCell>

									<TableCell className="px-6 py-4">
										<Button
											variant="ghost"
											size="sm"
											className="text-xs text-muted-foreground"
											onClick={(e) => {
												e.stopPropagation();
												handleEditClick(scan);
											}}
										>
											<Pencil className="w-3.5 h-3.5 mr-1" />
											Edit
										</Button>
									</TableCell>
									<TableCell className="px-6 py-4">
										<Button
											variant="ghost"
											size="sm"
											className="text-xs text-destructive hover:text-destructive"
											onClick={async (e) => {
												e.stopPropagation();
												if (
													confirm("Are you sure you want to delete this scan?")
												) {
													try {
														const token = localStorage.getItem("token");
														await fetch(
															`http://localhost:8000/api/scans/${scan.id}`,
															{
																method: "DELETE",
																headers: {
																	Authorization: `Bearer ${token}`,
																},
															},
														);
														window.location.reload(); // Simple reload to refresh data
													} catch (err) {
														console.error("Failed to delete", err);
													}
												}
											}}
										>
											<Trash2 className="w-3.5 h-3.5 mr-1" />
											Delete
										</Button>
									</TableCell>
								</TableRow>
							);
						})}
					</TableBody>
				</Table>
			</div>

			{filtered.length === 0 && (
				<div className="text-center py-16 text-muted-foreground">
					<p className="text-sm font-medium">
						{config.scanLibrary.noScansFound}
					</p>
					<p className="text-xs mt-1">{config.scanLibrary.adjustFilters}</p>
				</div>
			)}
		</div>
	);
};

export default ScanLibrary;
