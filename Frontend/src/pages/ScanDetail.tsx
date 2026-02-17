import { useParams, useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, Download, FileText, RotateCcw } from "lucide-react";
import { useState } from "react";
import { SliceViewer } from "@/components/scan/SliceViewer";
import { QuadViewer } from "@/components/scan/QuadViewer";
import { SegmentationControls } from "@/components/scan/SegmentationControls";
import { ProcessingPipeline } from "@/components/scan/ProcessingPipeline";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useConfig } from "@/hooks/use-config";
import { useScans } from "@/hooks/use-data";
import { MRIScan, AppConfig } from "@/lib/types";

const ScanDetail = () => {
	const { id } = useParams();
	const navigate = useNavigate();
	const location = useLocation();
	const { data: config, isLoading: configLoading } = useConfig();
	const { data: scans, isLoading: scansLoading } = useScans();

	const scan =
		location.state?.scanResult ||
		(scans as MRIScan[])?.find((s) => s.id === id);

	const [overlays, setOverlays] = useState({
		wt: true,
		tc: true,
		et: true,
		opacity: 0.6,
	});

	const [overlayColors] = useState(() => {
		const saved = localStorage.getItem("overlayColors");
		return saved
			? JSON.parse(saved)
			: {
					wt: "#eab308",
					tc: "#ef4444",
					et: "#3b82f6",
				};
	});

	const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");

	if (configLoading || scansLoading || !config) {
		return (
			<div className="space-y-6">
				<div className="h-10 bg-muted rounded-xl w-24" />
				<div className="h-20 bg-muted rounded-xl w-1/2" />
				<div className="h-64 bg-muted rounded-xl" />
			</div>
		);
	}

	if (!scan) {
		return (
			<div className="text-center py-20">
				<p className="text-lg font-semibold text-foreground">Scan not found</p>
				<p className="text-sm text-muted-foreground mt-1">
					The requested scan could not be located.
				</p>
				<Button variant="ghost" className="mt-4" onClick={() => navigate("/")}>
					Return to Dashboard
				</Button>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center gap-4">
				<Button
					variant="ghost"
					size="sm"
					onClick={() => navigate(-1)}
					className="text-muted-foreground"
				>
					<ArrowLeft className="w-4 h-4 mr-1" />
					{(config as AppConfig).scanDetail.backButton}
				</Button>
			</div>

			<div className="flex items-start justify-between gap-4">
				<div>
					<div className="flex items-center gap-3 mb-1">
						<h1 className="text-2xl font-bold text-foreground tracking-tight">
							{scan.patientName}
						</h1>
					</div>
					<div className="flex items-center gap-4 text-sm text-muted-foreground">
						<span>{scan.patientId}</span>
						<span>·</span>
						<span>{scan.scanDate}</span>
						<span>·</span>
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
					</div>
				</div>

				<div className="flex gap-2 shrink-0">
					{scan.results && (
						<>
							<Button variant="outline" size="sm" className="gap-1.5 text-xs">
								<Download className="w-3.5 h-3.5" />
								{(config as AppConfig).scanDetail.exportButton}
							</Button>
							<Button variant="outline" size="sm" className="gap-1.5 text-xs">
								<FileText className="w-3.5 h-3.5" />
								{(config as AppConfig).scanDetail.reportButton}
							</Button>
						</>
					)}
				</div>
			</div>

			{/* Main content */}
			<Tabs defaultValue="viewer" className="space-y-6">
				<TabsList className="bg-muted/50">
					<TabsTrigger value="viewer">
						{(config as AppConfig).scanDetail.viewerTab}
					</TabsTrigger>
					<TabsTrigger value="pipeline">
						{(config as AppConfig).scanDetail.pipelineTab}
					</TabsTrigger>
					<TabsTrigger value="report">
						{(config as AppConfig).scanDetail.reportTab}
					</TabsTrigger>
				</TabsList>

				<TabsContent value="viewer" className="space-y-0">
					<div className="grid grid-cols-3 gap-6">
						<div className="col-span-2">
							<div className="flex justify-between items-center mb-4">
								<h2 className="text-sm font-semibold text-foreground">
									Scan Visualisation
								</h2>
								<div className="bg-muted p-1 rounded-lg inline-flex">
									<button
										onClick={() => setViewMode("2d")}
										className={cn(
											"px-3 py-1.5 text-xs font-medium rounded-md transition-all",
											viewMode === "2d"
												? "bg-background text-foreground shadow-sm"
												: "text-muted-foreground hover:text-foreground",
										)}
									>
										2D Slices
									</button>
									<button
										onClick={() => setViewMode("3d")}
										className={cn(
											"px-3 py-1.5 text-xs font-medium rounded-md transition-all",
											viewMode === "3d"
												? "bg-background text-foreground shadow-sm"
												: "text-muted-foreground hover:text-foreground",
										)}
									>
										3D Volume
									</button>
								</div>
							</div>

							{viewMode === "2d" ? (
								<SliceViewer
									scanId={scan.id}
									modalities={scan.modalities}
									overlays={overlays}
									overlayColors={overlayColors}
								/>
							) : (
								<div className="aspect-square bg-black rounded-xl overflow-hidden border border-border relative">
									<QuadViewer
										scanId={scan.id}
										modality={scan.modalities[0]?.toLowerCase() || "flair"}
										overlays={overlays}
										overlayColors={overlayColors}
									/>
								</div>
							)}
						</div>
						<div className="space-y-6">
							<SegmentationControls
								overlays={overlays}
								onOverlaysChange={setOverlays}
								overlayColors={overlayColors}
							/>

							{/* Scan metrics */}
							{scan.results && (
								<div>
									<h3 className="text-xs font-semibold tracking-wider uppercase text-muted-foreground mb-3">
										{(config as AppConfig).scanDetail.measurementsTitle}
									</h3>
									<div className="grid grid-cols-2 gap-2">
										<div className="p-3 rounded-lg bg-muted/30 border border-border">
											<p className="text-xs text-muted-foreground">
												Classification
											</p>
											<p className="text-sm font-semibold text-foreground mt-0.5">
												{scan.results.classification}
											</p>
										</div>
										<div className="p-3 rounded-lg bg-muted/30 border border-border">
											<p className="text-xs text-muted-foreground">
												Confidence
											</p>
											<p className="text-sm font-semibold text-foreground mt-0.5">
												{(scan.results.confidence * 100).toFixed(1)}%
											</p>
										</div>
										<div className="p-3 rounded-lg bg-muted/30 border border-border">
											<p className="text-xs text-muted-foreground">
												Total Volume
											</p>
											<p className="text-sm font-semibold text-foreground mt-0.5">
												{scan.results.tumorVolume} cm³
											</p>
										</div>
										<div className="p-3 rounded-lg bg-muted/30 border border-border">
											<p className="text-xs text-muted-foreground">
												WT / TC / ET
											</p>
											<p className="text-sm font-semibold text-foreground mt-0.5">
												{scan.results.wtVolume} / {scan.results.tcVolume} /{" "}
												{scan.results.etVolume}
											</p>
										</div>
									</div>
								</div>
							)}
						</div>
					</div>
				</TabsContent>

				<TabsContent value="pipeline">
					<div className="max-w-xl">
						<ProcessingPipeline scan={scan} />
					</div>
				</TabsContent>

				<TabsContent value="report">
					{scan.results ? (
						<div className="max-w-2xl bg-card rounded-xl border border-border p-8 space-y-6">
							<div className="border-b border-border pb-4">
								<h2 className="text-lg font-bold text-foreground">
									{(config as AppConfig).scanDetail.reportTitle}
								</h2>
								<p className="text-xs text-muted-foreground mt-1">
									Generated on {scan.scanDate} · NeuroScan AI v2.4
								</p>
							</div>

							<div className="grid grid-cols-2 gap-4 text-sm">
								<div>
									<p className="text-muted-foreground text-xs font-semibold uppercase tracking-wider">
										Patient
									</p>
									<p className="font-medium text-foreground mt-1">
										{scan.patientName}
									</p>
									<p className="text-muted-foreground text-xs">
										{scan.patientId}
									</p>
								</div>
								<div>
									<p className="text-muted-foreground text-xs font-semibold uppercase tracking-wider">
										Scan Date
									</p>
									<p className="font-medium text-foreground mt-1">
										{scan.scanDate}
									</p>
									<p className="text-muted-foreground text-xs">
										Modalities: {scan.modalities.join(", ")}
									</p>
								</div>
							</div>

							<div className="space-y-3">
								<h3 className="text-sm font-semibold text-foreground">
									{(config as AppConfig).scanDetail.findingsTitle}
								</h3>
								<div className="p-4 rounded-lg bg-muted/30 border border-border space-y-2">
									<div className="flex items-center justify-between">
										<span className="text-sm text-muted-foreground">
											Tumour Detected
										</span>
										<span
											className={cn(
												"text-sm font-semibold",
												scan.results.detected ? "text-warning" : "text-success",
											)}
										>
											{scan.results.detected ? "Yes" : "No"}
										</span>
									</div>
									<div className="flex items-center justify-between">
										<span className="text-sm text-muted-foreground">
											Classification
										</span>
										<span className="text-sm font-semibold text-foreground">
											{scan.results.classification}
										</span>
									</div>
									<div className="flex items-center justify-between">
										<span className="text-sm text-muted-foreground">
											Confidence Score
										</span>
										<span className="text-sm font-semibold text-foreground">
											{(scan.results.confidence * 100).toFixed(1)}%
										</span>
									</div>
								</div>
							</div>

							{scan.results.detected && (
								<div className="space-y-3">
									<h3 className="text-sm font-semibold text-foreground">
										{(config as AppConfig).scanDetail.volumetricTitle}
									</h3>
									<div className="p-4 rounded-lg bg-muted/30 border border-border space-y-2">
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-2">
												<div
													className="w-2.5 h-2.5 rounded-sm"
													style={{ background: overlayColors.wt }}
												/>
												<span className="text-sm text-muted-foreground">
													Whole Tumour (WT)
												</span>
											</div>
											<span className="text-sm font-semibold text-foreground">
												{scan.results.wtVolume} cm³
											</span>
										</div>
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-2">
												<div
													className="w-2.5 h-2.5 rounded-sm"
													style={{ background: overlayColors.tc }}
												/>
												<span className="text-sm text-muted-foreground">
													Tumour Core (TC)
												</span>
											</div>
											<span className="text-sm font-semibold text-foreground">
												{scan.results.tcVolume} cm³
											</span>
										</div>
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-2">
												<div
													className="w-2.5 h-2.5 rounded-sm"
													style={{ background: overlayColors.et }}
												/>
												<span className="text-sm text-muted-foreground">
													Enhancing Tumour (ET)
												</span>
											</div>
											<span className="text-sm font-semibold text-foreground">
												{scan.results.etVolume} cm³
											</span>
										</div>
									</div>
								</div>
							)}

							<div className="border-t border-border pt-4 flex gap-3">
								<Button className="bg-medical text-medical-foreground hover:bg-medical/90 gap-2">
									<Download className="w-4 h-4" />
									{(config as AppConfig).scanDetail.downloadPDF}
								</Button>
								<Button variant="outline" className="gap-2">
									<Download className="w-4 h-4" />
									{(config as AppConfig).scanDetail.exportCSV}
								</Button>
							</div>
						</div>
					) : (
						<div className="text-center py-16 text-muted-foreground">
							<FileText className="w-10 h-10 mx-auto mb-3 opacity-30" />
							<p className="text-sm font-medium">
								{(config as AppConfig).scanDetail.notAvailable}
							</p>
							<p className="text-xs mt-1">
								{(config as AppConfig).scanDetail.processingRequired}
							</p>
						</div>
					)}
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default ScanDetail;
