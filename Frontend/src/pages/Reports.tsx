import { useNavigate } from "react-router-dom";
import { Download, FileText, Search } from "lucide-react";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useScans } from "@/hooks/use-data";
import { useConfig } from "@/hooks/use-config";
import { MRIScan, AppConfig } from "@/lib/types";

const Reports = () => {
	const navigate = useNavigate();
	const [search, setSearch] = useState("");
	const { data: scans, isLoading: scansLoading } = useScans();
	const { data: config, isLoading: configLoading } = useConfig();

	if (scansLoading || configLoading || !scans || !config) {
		return (
			<div className="space-y-6 animate-pulse">
				<div className="h-20 bg-muted rounded-xl w-1/2" />
				<div className="space-y-3">
					{[1, 2, 3].map((i) => (
						<div key={i} className="h-24 bg-muted rounded-xl" />
					))}
				</div>
			</div>
		);
	}

	const completedScans = scans.filter((s: MRIScan) => s.status === "completed");

	const filtered = completedScans.filter(
		(s: MRIScan) =>
			s.patientName.toLowerCase().includes(search.toLowerCase()) ||
			s.patientId.toLowerCase().includes(search.toLowerCase()),
	);

	return (
		<div className="space-y-6">
			<div className="flex items-center justify-between gap-4">
				<div>
					<h1 className="text-2xl font-bold text-foreground tracking-tight">
						{(config as AppConfig).reports.title}
					</h1>
					<p className="text-sm text-muted-foreground mt-1">
						{(config as AppConfig).reports.subtitle}
					</p>
				</div>
				<Button variant="outline" size="sm" className="gap-1.5 text-xs">
					<Download className="w-3.5 h-3.5" />
					{(config as AppConfig).reports.exportAll}
				</Button>
			</div>

			{/* Filters */}
			<div className="flex items-center gap-4 flex-wrap">
				<div className="relative w-72">
					<Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
					<Input
						placeholder={(config as AppConfig).reports.searchPlaceholder}
						value={search}
						onChange={(e) => setSearch(e.target.value)}
						className="pl-9 h-9 bg-muted/50 border-none text-sm"
					/>
				</div>
			</div>

			{/* Report cards */}
			<div className="space-y-3">
				{filtered.map((scan) => (
					<div
						key={scan.id}
						className="bg-card rounded-xl border border-border p-5 flex items-center gap-4 hover:border-medical/30 hover:shadow-md transition-all cursor-pointer"
						onClick={() => navigate(`/scan/${scan.id}`)}
					>
						<div className="flex items-center gap-3 flex-1 min-w-0">
							<div className="w-10 h-10 rounded-lg bg-medical/10 flex items-center justify-center shrink-0">
								<FileText className="w-5 h-5 text-medical" />
							</div>
							<div className="min-w-0">
								<p className="text-sm font-semibold text-foreground truncate">
									{scan.patientName} — AI Analysis Report
								</p>
								<div className="flex items-center gap-2 mt-0.5 flex-wrap">
									<span className="text-xs text-muted-foreground">
										{scan.patientId}
									</span>
									<span className="text-xs text-muted-foreground">·</span>
									<span className="text-xs text-muted-foreground">
										{scan.scanDate}
									</span>
									{scan.results && (
										<>
											<span className="text-xs text-muted-foreground mr-1">
												·
											</span>
											<span
												className={cn(
													"text-xs font-medium",
													scan.results.detected
														? "text-warning"
														: "text-success",
												)}
											>
												{scan.results.detected
													? scan.results.classification
													: "No Tumour"}
											</span>
										</>
									)}
								</div>
							</div>
						</div>

						<div className="flex items-center gap-3">
							{scan.modalities.map((m) => (
								<span
									key={m}
									className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-muted text-muted-foreground"
								>
									{m}
								</span>
							))}
						</div>

						<div className="flex items-center gap-2 shrink-0">
							<Button
								variant="outline"
								size="sm"
								className="gap-1.5 text-xs"
								onClick={(e) => {
									e.stopPropagation();
								}}
							>
								<Download className="w-3.5 h-3.5" />
								PDF
							</Button>
							<Button
								variant="outline"
								size="sm"
								className="gap-1.5 text-xs"
								onClick={(e) => {
									e.stopPropagation();
								}}
							>
								<Download className="w-3.5 h-3.5" />
								CSV
							</Button>
						</div>
					</div>
				))}
			</div>

			{filtered.length === 0 && (
				<div className="text-center py-16 text-muted-foreground">
					<FileText className="w-10 h-10 mx-auto mb-3 opacity-30" />
					<p className="text-sm font-medium">
						{(config as AppConfig).reports.noReportsFound}
					</p>
					<p className="text-xs mt-1">
						{(config as AppConfig).reports.autoGenerated}
					</p>
				</div>
			)}
		</div>
	);
};

export default Reports;
