import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Download, FileText, RotateCcw } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { SliceViewer } from "@/components/scan/SliceViewer";
import { VolumeViewer3D } from "@/components/scan/VolumeViewer3D";
import { SegmentationControls } from "@/components/scan/SegmentationControls";
import { ProcessingPipeline } from "@/components/scan/ProcessingPipeline";
import { ModalityBadges } from "@/components/scan/ModalityBadges";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { APP_CONFIG } from "@/lib/config";
import { formatPercent, isInProgress } from "@/lib/scan-format";
import { useOverlayColors } from "@/hooks/use-overlay-colors";
import { useReportDownload } from "@/hooks/use-report-download";
import { useRetryScan, useScan } from "@/hooks/use-scans";
import type { Overlays } from "@/lib/types";

const config = APP_CONFIG.scanDetail;

const ScanDetail = () => {
	const { id } = useParams();
	const navigate = useNavigate();
	const { data: scan, isLoading, isError } = useScan(id);
	const { colors: overlayColors } = useOverlayColors();
	const { pending, downloadPdf, downloadCsv } = useReportDownload();
	const retryScan = useRetryScan();

	const [overlays, setOverlays] = useState<Overlays>({
		wt: true,
		tc: true,
		et: true,
		opacity: 0.6,
	});

	const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");

	if (isLoading) {
		return (
			<div className="space-y-6">
				<div className="h-10 w-24 rounded-xl bg-muted" />
				<div className="h-20 w-1/2 rounded-xl bg-muted" />
				<div className="h-64 rounded-xl bg-muted" />
			</div>
		);
	}

	if (isError || !scan) {
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

	const results = scan.status === "completed" ? scan.results : null;
	const completed = results !== null;

	const retry = async () => {
		try {
			await retryScan.mutateAsync(scan.id);
			toast.success("Scan queued for processing");
		} catch (error) {
			toast.error(error instanceof Error ? error.message : "Retry failed");
		}
	};

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
					{config.backButton}
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
						<ModalityBadges modalities={scan.modalities} />
					</div>
				</div>

				<div className="flex shrink-0 gap-2">
					{scan.status === "failed" && (
						<Button
							variant="outline"
							size="sm"
							className="gap-1.5 text-xs"
							disabled={retryScan.isPending}
							onClick={retry}
						>
							<RotateCcw className="h-3.5 w-3.5" />
							{config.retryButton}
						</Button>
					)}
					{completed && (
						<>
							<Button
								variant="outline"
								size="sm"
								className="gap-1.5 text-xs"
								disabled={pending !== null}
								onClick={() => downloadCsv(scan.id)}
							>
								<Download className="h-3.5 w-3.5" />
								{config.exportButton}
							</Button>
							<Button
								variant="outline"
								size="sm"
								className="gap-1.5 text-xs"
								disabled={pending !== null}
								onClick={() => downloadPdf(scan.id)}
							>
								<FileText className="h-3.5 w-3.5" />
								{config.reportButton}
							</Button>
						</>
					)}
				</div>
			</div>

			{isInProgress(scan) && (
				<div
					role="status"
					className="flex items-center gap-3 rounded-xl border border-medical/20 bg-medical/5 p-4 text-sm text-foreground"
				>
					<div className="h-4 w-4 animate-spin rounded-full border-2 border-medical border-t-transparent" />
					Processing: {scan.pipelineStep} ({scan.progress}%). Results appear
					here automatically.
				</div>
			)}
			{scan.status === "failed" && scan.errorMessage && (
				<div
					role="alert"
					className="rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive"
				>
					{scan.errorMessage}
				</div>
			)}

			{/* Main content */}
			<Tabs defaultValue="viewer" className="flex flex-col gap-6">
				<TabsList className="self-start bg-muted/50">
					<TabsTrigger value="viewer">{config.viewerTab}</TabsTrigger>
					<TabsTrigger value="pipeline">{config.pipelineTab}</TabsTrigger>
					<TabsTrigger value="report">{config.reportTab}</TabsTrigger>
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
													? "border-medical bg-medical text-medical-foreground shadow-xs"
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
									hasSegmentation={scan.hasSegmentation}
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
									hasSegmentation={scan.hasSegmentation}
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
										{config.measurementsTitle}
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
												{formatPercent(scan.results.confidence)}
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
					{results ? (
						<div className="max-w-2xl space-y-6 rounded-xl border border-border bg-card p-8">
							<div className="border-b border-border pb-4">
								<h2 className="text-lg font-bold text-foreground">
									{config.reportTitle}
								</h2>
								<p className="mt-1 text-xs text-muted-foreground">
									Generated on {scan.scanDate}
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
									{config.findingsTitle}
								</h3>
								<div className="space-y-2 rounded-lg border border-border bg-muted/30 p-4">
									<div className="flex items-center justify-between">
										<span className="text-sm text-muted-foreground">
											Tumour Detected
										</span>
										<span
											className={cn(
												"text-sm font-semibold",
												results.detected ? "text-warning" : "text-success",
											)}
										>
											{results.detected ? "Yes" : "No"}
										</span>
									</div>
									<div className="flex items-center justify-between">
										<span className="text-sm text-muted-foreground">
											Classification
										</span>
										<span className="text-sm font-semibold text-foreground">
											{results.classification}
										</span>
									</div>
									<div className="flex items-center justify-between">
										<span className="text-sm text-muted-foreground">
											Confidence Score
										</span>
										<span className="text-sm font-semibold text-foreground">
											{formatPercent(results.confidence)}
										</span>
									</div>
								</div>
							</div>

							{results.detected && (
								<div className="space-y-3">
									<h3 className="text-sm font-semibold text-foreground">
										{config.volumetricTitle}
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
												{results.wtVolume} cm³
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
												{results.tcVolume} cm³
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
												{results.etVolume} cm³
											</span>
										</div>
									</div>
								</div>
							)}

							<div className="flex gap-3 border-t border-border pt-4">
								<Button
									className="gap-2 bg-medical text-medical-foreground hover:bg-medical/90"
									disabled={pending !== null}
									onClick={() => downloadPdf(scan.id)}
								>
									<Download className="h-4 w-4" />
									{pending === "pdf" ? "Preparing..." : config.downloadPDF}
								</Button>
								<Button
									variant="outline"
									className="gap-2"
									disabled={pending !== null}
									onClick={() => downloadCsv(scan.id)}
								>
									<Download className="h-4 w-4" />
									{pending === "csv" ? "Preparing..." : config.exportCSV}
								</Button>
							</div>
						</div>
					) : (
						<div className="py-16 text-center text-muted-foreground">
							<FileText className="mx-auto mb-3 h-10 w-10 opacity-30" />
							<p className="text-sm font-medium">{config.notAvailable}</p>
							<p className="mt-1 text-xs">{config.processingRequired}</p>
						</div>
					)}
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default ScanDetail;
