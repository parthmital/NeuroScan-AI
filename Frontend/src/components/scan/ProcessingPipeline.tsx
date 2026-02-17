import { Check, Loader2, AlertCircle, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PipelineStep, MRIScan } from "@/lib/mock-data";

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

const statusIcon = (status: StepStatus) => {
	switch (status) {
		case "complete":
			return <Check className="w-4 h-4" />;
		case "processing":
			return <Loader2 className="w-4 h-4 animate-spin" />;
		case "failed":
			return <AlertCircle className="w-4 h-4" />;
		default:
			return <Clock className="w-4 h-4" />;
	}
};

const statusColors: Record<StepStatus, string> = {
	complete: "bg-success text-success-foreground",
	processing: "bg-medical text-medical-foreground",
	failed: "bg-destructive text-destructive-foreground",
	pending: "bg-muted text-muted-foreground",
};

const lineColors: Record<StepStatus, string> = {
	complete: "bg-success",
	processing: "bg-medical animate-pulse-medical",
	failed: "bg-destructive",
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
		<div className="space-y-1">
			<h3 className="text-xs font-semibold tracking-wider uppercase text-muted-foreground mb-4">
				Processing Pipeline
			</h3>
			<div className="space-y-0">
				{steps.map((step, idx) => {
					const status = getStepStatus(
						step.key,
						scan.pipelineStep,
						scan.status,
					);
					return (
						<div key={step.key} className="flex gap-4">
							{/* Timeline */}
							<div className="flex flex-col items-center">
								<div
									className={cn(
										"w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-colors",
										statusColors[status],
									)}
								>
									{statusIcon(status)}
								</div>
								{idx < steps.length - 1 && (
									<div
										className={cn(
											"w-0.5 flex-1 min-h-[40px] transition-colors",
											lineColors[status],
										)}
									/>
								)}
							</div>

							{/* Content */}
							<div className="pb-6 pt-1">
								<p
									className={cn(
										"text-sm font-semibold",
										status === "pending"
											? "text-muted-foreground"
											: "text-foreground",
									)}
								>
									{step.label}
								</p>
								<p className="text-xs text-muted-foreground mt-0.5">
									{step.description}
								</p>
								{step.result && status === "complete" && (
									<div className="mt-2 px-3 py-2 rounded-lg bg-muted/50 border border-border">
										<p className="text-xs font-medium text-foreground">
											{step.result}
										</p>
									</div>
								)}
								{status === "failed" && (
									<div className="mt-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/20">
										<p className="text-xs font-medium text-destructive">
											Processing failed. Check input data and retry.
										</p>
									</div>
								)}
							</div>
						</div>
					);
				})}
			</div>
		</div>
	);
};
