import { useParams, useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, Download, FileText } from "lucide-react";
import { useState } from "react";
import { SliceViewer } from "@/components/scan/SliceViewer";
import { VolumeViewer3D } from "@/components/scan/VolumeViewer3D";
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

	const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");

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

	if (configLoading || scansLoading || !config) {
		return (
			<div className="space-y-6">
				<div className="h-10 w-24 rounded-xl bg-muted" />
				<div className="h-20 w-1/2 rounded-xl bg-muted" />
				<div className="h-64 rounded-xl bg-muted" />
			</div>
		);
	}

	if (!scan) {
		return (
			<div className="py-20 text-center">
				<p className="text-lg font-semibold text-foreground">Scan not found</p>
				<p className="mt-1 text-sm text-muted-foreground">
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
					<ArrowLeft className="mr-1 h-4 w-4" />
					{(config as AppConfig).scanDetail.backButton}
				</Button>
			</div>

			<div className="flex items-start justify-between gap-4">
				<div>
					<div className="mb-1 flex items-center gap-3">
						<h1 className="text-2xl font-bold tracking-tight text-foreground">
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
									className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground"
								>
									{m}
								</span>
							))}
						</div>
					</div>
				</div>

				<div className="flex shrink-0 gap-2">
					{scan.results && (
						<>
							<Button variant="outline" size="sm" className="gap-1.5 text-xs">
								<Download className="h-3.5 w-3.5" />
								{(config as AppConfig).scanDetail.exportButton}
							</Button>
							<Button variant="outline" size="sm" className="gap-1.5 text-xs">
								<FileText className="h-3.5 w-3.5" />
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
							<div className="mb-4 flex items-center justify-between">
								<h2 className="text-sm font-semibold text-foreground">
									Scan Visualisation
								</h2>
								{/* 2D / 3D view toggle */}
								<div className="flex gap-1.5">
									{(["2d", "3d"] as const).map((mode) => (
										<button
											key={mode}
											onClick={() => setViewMode(mode)}
											className={cn(
												"rounded-md border px-3 py-1.5 text-xs font-semibold uppercase tracking-wide transition-all",
												viewMode === mode
													? "border-medical bg-medical text-medical-foreground shadow-sm"
													: "border-border bg-card text-muted-foreground hover:border-medical/50 hover:bg-muted/50",
											)}
										>
											{mode}
										</button>
									))}
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
								<VolumeViewer3D
									scanId={scan.id}
									modalities={scan.modalities}
									overlays={overlays}
									overlayColors={overlayColors}
									hasSegmentation={!!scan.results?.detected}
								/>
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
									<h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
										{(config as AppConfig).scanDetail.measurementsTitle}
									</h3>
									<div className="grid grid-cols-2 gap-2">
										<div className="rounded-lg border border-border bg-muted/30 p-3">
											<p className="text-xs text-muted-foreground">
												Classification
											</p>
											<p className="mt-0.5 text-sm font-semibold text-foreground">
												{scan.results.classification}
											</p>
										</div>
										<div className="rounded-lg border border-border bg-muted/30 p-3">
											<p className="text-xs text-muted-foreground">
												Confidence
											</p>
											<p className="mt-0.5 text-sm font-semibold text-foreground">
												{(scan.results.confidence * 100).toFixed(1)}%
											</p>
										</div>
										<div className="rounded-lg border border-border bg-muted/30 p-3">
											<p className="text-xs text-muted-foreground">
												Total Volume
											</p>
											<p className="mt-0.5 text-sm font-semibold text-foreground">
												{scan.results.tumorVolume} cm³
											</p>
										</div>
										<div className="rounded-lg border border-border bg-muted/30 p-3">
											<p className="text-xs text-muted-foreground">
												WT / TC / ET
											</p>
											<p className="mt-0.5 text-sm font-semibold text-foreground">
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
						<div className="max-w-2xl space-y-6 rounded-xl border border-border bg-card p-8">
							<div className="border-b border-border pb-4">
								<h2 className="text-lg font-bold text-foreground">
									{(config as AppConfig).scanDetail.reportTitle}
								</h2>
								<p className="mt-1 text-xs text-muted-foreground">
									Generated on {scan.scanDate} · NeuroScan AI v2.4
								</p>
							</div>

							<div className="grid grid-cols-2 gap-4 text-sm">
								<div>
									<p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
										Patient
									</p>
									<p className="mt-1 font-medium text-foreground">
										{scan.patientName}
									</p>
									<p className="text-xs text-muted-foreground">
										{scan.patientId}
									</p>
								</div>
								<div>
									<p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
										Scan Date
									</p>
									<p className="mt-1 font-medium text-foreground">
										{scan.scanDate}
									</p>
									<p className="text-xs text-muted-foreground">
										Modalities: {scan.modalities.join(", ")}
									</p>
								</div>
							</div>

							<div className="space-y-3">
								<h3 className="text-sm font-semibold text-foreground">
									{(config as AppConfig).scanDetail.findingsTitle}
								</h3>
								<div className="space-y-2 rounded-lg border border-border bg-muted/30 p-4">
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
									<div className="space-y-2 rounded-lg border border-border bg-muted/30 p-4">
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-2">
												<div
													className="h-2.5 w-2.5 rounded-sm"
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
													className="h-2.5 w-2.5 rounded-sm"
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
													className="h-2.5 w-2.5 rounded-sm"
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

							<div className="flex gap-3 border-t border-border pt-4">
								<Button className="gap-2 bg-medical text-medical-foreground hover:bg-medical/90">
									<Download className="h-4 w-4" />
									{(config as AppConfig).scanDetail.downloadPDF}
								</Button>
								<Button variant="outline" className="gap-2">
									<Download className="h-4 w-4" />
									{(config as AppConfig).scanDetail.exportCSV}
								</Button>
							</div>
						</div>
					) : (
						<div className="py-16 text-center text-muted-foreground">
							<FileText className="mx-auto mb-3 h-10 w-10 opacity-30" />
							<p className="text-sm font-medium">
								{(config as AppConfig).scanDetail.notAvailable}
							</p>
							<p className="mt-1 text-xs">
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
