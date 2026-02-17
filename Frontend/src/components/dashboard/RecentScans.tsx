import { useNavigate } from "react-router-dom";
import { Eye, RotateCcw, MoreHorizontal, Trash2, Pencil } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useScans } from "@/hooks/use-data";
import { useConfig } from "@/hooks/use-config";
import { MRIScan, AppConfig } from "@/lib/types";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
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
import { useState } from "react";

export const RecentScans = () => {
	const navigate = useNavigate();
	const { data: scans, isLoading: scansLoading } = useScans();
	const { data: config, isLoading: configLoading } = useConfig();
	const [editingScan, setEditingScan] = useState<MRIScan | null>(null);
	const [editForm, setEditForm] = useState({ patientName: "", patientId: "" });

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

	if (scansLoading || configLoading || !scans || !config) {
		return <div className="h-64 bg-card rounded-xl border border-border" />;
	}

	return (
		<div className="bg-card rounded-xl border border-border overflow-hidden">
			<div className="flex items-center justify-between px-4 sm:px-6 py-4 border-b border-border">
				<h2 className="text-sm font-semibold text-foreground">
					{(config as AppConfig).dashboard.recentScansTitle}
				</h2>
				<Button
					variant="ghost"
					size="sm"
					onClick={() => navigate("/scans")}
					className="text-xs text-muted-foreground"
				>
					{(config as AppConfig).dashboard.viewAllLabel}
				</Button>
			</div>

			{/* Mobile card view */}
			<div className="sm:hidden divide-y divide-border">
				{(scans as MRIScan[]).slice(0, 6).map((scan) => {
					return (
						<div
							key={scan.id}
							className="px-4 sm:px-6 py-4 cursor-pointer hover:bg-muted/30 transition-colors"
							onClick={() => navigate(`/scan/${scan.id}`)}
						>
							<div className="flex items-start justify-between mb-2">
								<div>
									<p className="text-sm font-medium text-foreground">
										{scan.patientName}
									</p>
									<p className="text-xs text-muted-foreground">
										{scan.patientId}
									</p>
								</div>
							</div>
							<div className="flex items-center gap-2 text-xs text-muted-foreground">
								<span>{scan.scanDate}</span>
								<span>·</span>
								<span>
									{scan.results
										? scan.results.detected
											? scan.results.classification
											: "No Tumour"
										: "Pending"}
								</span>
							</div>
							<div className="flex gap-2 mt-3">
								<Button
									variant="ghost"
									size="sm"
									className="text-xs text-muted-foreground"
									onClick={(e) => {
										e.stopPropagation();
										navigate(`/scan/${scan.id}`);
									}}
								>
									<Eye className="w-3.5 h-3.5 mr-1" />
									View
								</Button>
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
								<Button
									variant="ghost"
									size="sm"
									className="text-xs text-destructive hover:text-destructive"
									onClick={async (e) => {
										e.stopPropagation();
										if (confirm("Are you sure you want to delete this scan?")) {
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
							</div>
						</div>
					);
				})}
			</div>

			{/* Desktop table view */}
			<div className="hidden sm:block overflow-x-auto">
				<Table>
					<TableHeader>
						<TableRow className="hover:bg-transparent">
							<TableHead className="px-4 sm:px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground">
								Patient
							</TableHead>
							<TableHead className="px-4 sm:px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground hidden md:table-cell">
								Date
							</TableHead>
							<TableHead className="px-4 sm:px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground hidden lg:table-cell">
								Modalities
							</TableHead>
							<TableHead className="px-4 sm:px-6 py-4 text-xs font-semibold tracking-wider uppercase text-muted-foreground hidden md:table-cell">
								Result
							</TableHead>
							<TableHead className="px-4 sm:px-6 py-4 w-[80px]" />
						</TableRow>
					</TableHeader>
					<TableBody>
						{(scans as MRIScan[]).slice(0, 6).map((scan) => {
							return (
								<TableRow
									key={scan.id}
									className="cursor-pointer group"
									onClick={() => navigate(`/scan/${scan.id}`)}
								>
									<TableCell className="px-4 sm:px-6 py-4">
										<div>
											<p className="text-sm font-medium text-foreground">
												{scan.patientName}
											</p>
											<p className="text-xs text-muted-foreground">
												{scan.patientId}
											</p>
										</div>
									</TableCell>
									<TableCell className="px-4 sm:px-6 py-4 text-sm text-muted-foreground hidden md:table-cell">
										{scan.scanDate}
									</TableCell>
									<TableCell className="px-4 sm:px-6 py-4 hidden lg:table-cell">
										<div className="flex gap-1">
											{scan.modalities.map((m: string) => (
												<span
													key={m}
													className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-muted text-muted-foreground"
												>
													{m}
												</span>
											))}
										</div>
									</TableCell>
									<TableCell className="px-4 sm:px-6 py-4">
										<span className="text-sm text-muted-foreground">
											{scan.results
												? scan.results.detected
													? scan.results.classification
													: "No Tumour"
												: "—"}
										</span>
									</TableCell>
									<TableCell className="px-4 sm:px-6 py-4">
										<DropdownMenu>
											<DropdownMenuTrigger asChild>
												<button className="p-1.5 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors">
													<MoreHorizontal className="w-3.5 h-3.5" />
												</button>
											</DropdownMenuTrigger>
											<DropdownMenuContent align="end">
												<DropdownMenuItem
													onClick={(e) => {
														e.stopPropagation();
														navigate(`/scan/${scan.id}`);
													}}
												>
													<Eye className="w-3.5 h-3.5 mr-2" />
													View Details
												</DropdownMenuItem>
												<DropdownMenuItem
													onClick={(e) => {
														e.stopPropagation();
														handleEditClick(scan);
													}}
												>
													<Pencil className="w-3.5 h-3.5 mr-2" />
													Edit Details
												</DropdownMenuItem>
												<DropdownMenuItem
													className="text-destructive focus:text-destructive"
													onClick={async (e) => {
														e.stopPropagation();
														if (
															confirm(
																"Are you sure you want to delete this scan?",
															)
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
													<Trash2 className="w-3.5 h-3.5 mr-2" />
													Delete Scan
												</DropdownMenuItem>
											</DropdownMenuContent>
										</DropdownMenu>
									</TableCell>
								</TableRow>
							);
						})}
					</TableBody>
				</Table>
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
		</div>
	);
};
