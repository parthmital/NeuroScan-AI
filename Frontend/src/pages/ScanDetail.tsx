import { useParams, useNavigate } from "react-router-dom";
import { Download, FileText, Loader2, RotateCcw } from "lucide-react";
import { useState, type ReactNode } from "react";
import { toast } from "sonner";
import { ScanViewer } from "@/components/scan/viewer/ScanViewer";
import { SegmentationControls } from "@/components/scan/SegmentationControls";
import { ProcessingPipeline } from "@/components/scan/ProcessingPipeline";
import { ModalityBadges } from "@/components/scan/ModalityBadges";
import { ResultStatus } from "@/components/scan/ResultStatus";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { APP_CONFIG } from "@/lib/config";
import { formatPercent, isInProgress } from "@/lib/scan-format";
import { useOverlayColors } from "@/hooks/use-overlay-colors";
import { useReportDownload } from "@/hooks/use-report-download";
import { useRetryScan, useScan } from "@/hooks/use-scans";
import type { Overlays } from "@/lib/types";

const config = APP_CONFIG.scanDetail;

/** One labelled value in a measurement or report list. */
const Row = ({
	label,
	children,
	swatch,
}: {
	label: string;
	children: ReactNode;
	swatch?: string;
}) => (
	<div className="flex items-baseline justify-between gap-4 py-2.5">
		<dt className="flex items-center gap-2 text-[13px] text-muted-foreground">
			{swatch && (
				<span
					className="h-2 w-2 shrink-0 rounded-[2px]"
					style={{ background: swatch }}
				/>
			)}
			{label}
		</dt>
		<dd className="text-right font-mono text-[13px] font-medium text-foreground tabular-nums">
			{children}
		</dd>
	</div>
);

const sectionLabel =
	"text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground";

const ScanDetail = () => {
	const { id } = useParams();
	const navigate = useNavigate();
	const { data: scan, isLoading, isError } = useScan(id);
	const { colors: overlayColors } = useOverlayColors();
	const { pending, openPdf, downloadCsv } = useReportDownload();
	const retryScan = useRetryScan();

	const [overlays, setOverlays] = useState<Overlays>({
		wt: true,
		tc: true,
		et: true,
		opacity: 0.6,
	});

	if (isLoading) {
		return (
			<div
				role="status"
				aria-label="Loading scan"
				className="animate-pulse space-y-6"
			>
				<div className="h-8 w-20 rounded-md bg-muted" />
				<div className="h-14 w-1/2 rounded-lg bg-muted" />
				<div className="h-96 rounded-lg bg-muted" />
			</div>
		);
	}

	if (isError || !scan) {
		return (
			<div className="py-24 text-center">
				<p className="text-base font-semibold text-foreground">
					Scan not found
				</p>
				<p className="mt-1 text-sm text-muted-foreground">
					The requested scan could not be located.
				</p>
				<Button
					variant="outline"
					className="mt-5"
					onClick={() => navigate("/")}
				>
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
			<PageHeader
				back={{ label: config.backButton, onClick: () => navigate(-1) }}
				title={scan.patientName}
				description={
					<div className="flex flex-wrap items-center gap-x-4 gap-y-2">
						<span className="font-mono text-xs tabular-nums">
							{scan.patientId}
						</span>
						<span className="font-mono text-xs tabular-nums">
							{scan.scanDate}
						</span>
						<ModalityBadges modalities={scan.modalities} />
						<ResultStatus scan={scan} />
					</div>
				}
				actions={
					<>
						{scan.status === "failed" && (
							<Button
								variant="outline"
								size="sm"
								disabled={retryScan.isPending}
								onClick={retry}
							>
								<RotateCcw />
								{config.retryButton}
							</Button>
						)}
						{completed && (
							<>
								<Button
									variant="outline"
									size="sm"
									disabled={pending !== null}
									onClick={() => downloadCsv(scan.id)}
								>
									{pending === "csv" ? (
										<Loader2 className="animate-spin" />
									) : (
										<Download />
									)}
									{config.exportButton}
								</Button>
								<Button size="sm" onClick={() => openPdf(scan.id)}>
									<FileText />
									{config.reportButton}
								</Button>
							</>
						)}
					</>
				}
			/>

			{isInProgress(scan) && (
				<div
					role="status"
					className="flex items-center gap-3 rounded-md border border-medical/25 bg-medical-light px-4 py-3 text-[13px] text-foreground"
				>
					<Loader2 className="h-4 w-4 shrink-0 animate-spin text-medical" />
					<span>
						Processing: {scan.pipelineStep}{" "}
						<span className="font-mono tabular-nums">({scan.progress}%)</span>.
						Results appear here automatically.
					</span>
				</div>
			)}
			{scan.status === "failed" && scan.errorMessage && (
				<div
					role="alert"
					className="rounded-md border border-destructive/30 bg-destructive/[0.06] px-4 py-3 text-[13px] text-destructive"
				>
					{scan.errorMessage}
				</div>
			)}

			<Tabs defaultValue="viewer" className="flex flex-col gap-6">
				<TabsList>
					<TabsTrigger value="viewer">{config.viewerTab}</TabsTrigger>
					<TabsTrigger value="pipeline">{config.pipelineTab}</TabsTrigger>
					<TabsTrigger value="report">{config.reportTab}</TabsTrigger>
				</TabsList>

				<TabsContent value="viewer">
					<ScanViewer
						scanId={scan.id}
						modalities={scan.modalities}
						hasSegmentation={scan.hasSegmentation}
						overlays={overlays}
						overlayColors={overlayColors}
						aside={
							<>
								<SegmentationControls
									overlays={overlays}
									onOverlaysChange={setOverlays}
									overlayColors={overlayColors}
								/>
								{scan.results && (
									<Card>
										<CardHeader>
											<CardTitle>{config.measurementsTitle}</CardTitle>
										</CardHeader>
										<dl className="divide-y divide-border px-5 py-1">
											<Row label="Classification">
												<span className="font-sans">
													{scan.results.classification}
												</span>
											</Row>
											<Row label="Confidence">
												{formatPercent(scan.results.confidence)}
											</Row>
											<Row label="Total volume">
												{scan.results.tumorVolume} cm³
											</Row>
											<Row label="WT / TC / ET">
												{scan.results.wtVolume} / {scan.results.tcVolume} /{" "}
												{scan.results.etVolume}
											</Row>
										</dl>
									</Card>
								)}
							</>
						}
					/>
				</TabsContent>

				<TabsContent value="pipeline">
					<div className="max-w-xl">
						<ProcessingPipeline scan={scan} />
					</div>
				</TabsContent>

				<TabsContent value="report">
					{results ? (
						<Card className="max-w-3xl">
							<div className="flex flex-col gap-1 border-b border-border px-8 py-6">
								<p className={sectionLabel}>NeuroScan AI</p>
								<h2 className="text-lg font-semibold tracking-[-0.01em] text-foreground">
									{config.reportTitle}
								</h2>
								<p className="font-mono text-xs text-muted-foreground">
									Generated on {scan.scanDate}
								</p>
							</div>

							<div className="space-y-8 px-8 py-6">
								<div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
									<div>
										<p className={sectionLabel}>Patient</p>
										<p className="mt-2 text-sm font-medium text-foreground">
											{scan.patientName}
										</p>
										<p className="font-mono text-xs text-muted-foreground">
											{scan.patientId}
										</p>
									</div>
									<div>
										<p className={sectionLabel}>Scan Date</p>
										<p className="mt-2 font-mono text-sm font-medium text-foreground">
											{scan.scanDate}
										</p>
										<p className="text-xs text-muted-foreground">
											Modalities:{" "}
											<span className="font-mono">
												{scan.modalities.join(", ")}
											</span>
										</p>
									</div>
								</div>

								<section>
									<h3 className={cn(sectionLabel, "mb-1")}>
										{config.findingsTitle}
									</h3>
									<dl className="divide-y divide-border border-y border-border">
										<Row label="Tumour detected">
											<span
												className={cn(
													"font-sans font-semibold",
													results.detected ? "text-warning" : "text-success",
												)}
											>
												{results.detected ? "Yes" : "No"}
											</span>
										</Row>
										<Row label="Classification">
											<span className="font-sans">
												{results.classification}
											</span>
										</Row>
										<Row label="Confidence score">
											{formatPercent(results.confidence)}
										</Row>
									</dl>
								</section>

								{results.detected && (
									<section>
										<h3 className={cn(sectionLabel, "mb-1")}>
											{config.volumetricTitle}
										</h3>
										<dl className="divide-y divide-border border-y border-border">
											<Row label="Whole Tumour (WT)" swatch={overlayColors.wt}>
												{results.wtVolume} cm³
											</Row>
											<Row label="Tumour Core (TC)" swatch={overlayColors.tc}>
												{results.tcVolume} cm³
											</Row>
											<Row
												label="Enhancing Tumour (ET)"
												swatch={overlayColors.et}
											>
												{results.etVolume} cm³
											</Row>
										</dl>
									</section>
								)}
							</div>

							<div className="flex flex-wrap gap-2 border-t border-border px-8 py-4">
								<Button onClick={() => openPdf(scan.id)}>
									<FileText />
									{config.openPDF}
								</Button>
								<Button
									variant="outline"
									disabled={pending !== null}
									onClick={() => downloadCsv(scan.id)}
								>
									{pending === "csv" ? (
										<Loader2 className="animate-spin" />
									) : (
										<Download />
									)}
									{pending === "csv" ? "Preparing..." : config.exportCSV}
								</Button>
							</div>
						</Card>
					) : (
						<div className="rounded-lg border border-dashed border-border px-6 py-16 text-center">
							<p className="text-sm font-medium text-foreground">
								{config.notAvailable}
							</p>
							<p className="mt-1 text-[13px] text-muted-foreground">
								{config.processingRequired}
							</p>
						</div>
					)}
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default ScanDetail;
