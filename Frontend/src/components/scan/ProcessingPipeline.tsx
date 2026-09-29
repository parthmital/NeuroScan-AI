import { Check, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PipelineStep, MRIScan } from "@/lib/types";

interface ProcessingPipelineProps {
	scan: MRIScan;
}

type StepStatus = "complete" | "processing" | "pending" | "failed";

interface PipelineStepData {
	key: PipelineStep;
	label: string;
	description: string;
	result?: string;
}

const getStepStatus = (
	stepKey: PipelineStep,
	currentStep: PipelineStep,
	scanStatus: string,
): StepStatus => {
	const order: PipelineStep[] = [
		"detection",
		"classification",
		"segmentation",
		"complete",
	];
	const currentIdx = order.indexOf(currentStep);
	const stepIdx = order.indexOf(stepKey);

	if (scanStatus === "failed" && stepIdx === currentIdx) return "failed";
	if (stepIdx < currentIdx) return "complete";
	if (stepIdx === currentIdx && scanStatus === "processing")
		return "processing";
	return "pending";
};

const statusIcon = (status: StepStatus, step: number) => {
	switch (status) {
		case "complete":
			return <Check className="h-3.5 w-3.5" strokeWidth={2.25} />;
		case "processing":
			return <Loader2 className="h-3.5 w-3.5 animate-spin" />;
		case "failed":
			return <X className="h-3.5 w-3.5" strokeWidth={2.25} />;
		default:
			return <span className="font-mono text-[11px]">{step}</span>;
	}
};

const statusColors: Record<StepStatus, string> = {
	complete: "bg-foreground text-background",
	processing: "border border-medical bg-medical-light text-medical",
	failed: "bg-destructive text-destructive-foreground",
	pending: "border border-input bg-card text-muted-foreground",
};

const lineColors: Record<StepStatus, string> = {
	complete: "bg-foreground/70",
	processing: "bg-border",
	failed: "bg-border",
	pending: "bg-border",
};

export const ProcessingPipeline = ({ scan }: ProcessingPipelineProps) => {
	const steps: PipelineStepData[] = [
		{
			key: "detection",
			label: "Detection",
			description: "Analysing MRI for tumour presence",
			result:
				scan.results?.detected !== undefined
					? scan.results.detected
						? `Tumour detected (${(scan.results.confidence * 100).toFixed(0)}% confidence)`
						: `No tumour detected (${(scan.results.confidence * 100).toFixed(0)}% confidence)`
					: undefined,
		},
		{
			key: "classification",
			label: "Classification",
			description: "Determining tumour type and grade",
			result: scan.results?.classification,
		},
		{
			key: "segmentation",
			label: "Segmentation",
			description: "Delineating tumour regions (WT, TC, ET)",
			result: scan.results
				? `WT: ${scan.results.wtVolume}cm³ · TC: ${scan.results.tcVolume}cm³ · ET: ${scan.results.etVolume}cm³`
				: undefined,
		},
	];

	return (
		<div>
			<h3 className="mb-5 text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
				Processing Pipeline
			</h3>
			<ol>
				{steps.map((step, idx) => {
					const status = getStepStatus(
						step.key,
						scan.pipelineStep,
						scan.status,
					);
					return (
						<li key={step.key} className="flex gap-4">
							{/* Timeline */}
							<div className="flex flex-col items-center">
								<div
									className={cn(
										"flex h-6 w-6 shrink-0 items-center justify-center rounded-full transition-colors",
										statusColors[status],
									)}
								>
									{statusIcon(status, idx + 1)}
								</div>
								{idx < steps.length - 1 && (
									<div
										className={cn(
											"my-1 min-h-[36px] w-px flex-1 transition-colors",
											lineColors[status],
										)}
									/>
								)}
							</div>

							{/* Content */}
							<div className="min-w-0 pb-7">
								<p
									className={cn(
										"text-sm font-medium leading-6",
										status === "pending"
											? "text-muted-foreground"
											: "text-foreground",
									)}
								>
									{step.label}
								</p>
								<p className="mt-0.5 text-xs text-muted-foreground">
									{step.description}
								</p>
								{step.result && status === "complete" && (
									<div className="mt-2.5 rounded-md border border-border bg-card px-3 py-2">
										<p className="font-mono text-xs text-foreground">
											{step.result}
										</p>
									</div>
								)}
								{status === "failed" && (
									<div className="mt-2.5 rounded-md border border-destructive/30 bg-destructive/[0.06] px-3 py-2">
										<p className="text-xs font-medium text-destructive">
											Processing failed. Check input data and retry.
										</p>
									</div>
								)}
							</div>
						</li>
					);
				})}
			</ol>
		</div>
	);
};
