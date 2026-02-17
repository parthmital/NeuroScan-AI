export type ScanStatus =
	| "uploading"
	| "queued"
	| "processing"
	| "completed"
	| "failed";
export type Modality = "FLAIR" | "T1" | "T1CE" | "T2";
export type PipelineStep =
	| "detection"
	| "classification"
	| "segmentation"
	| "complete";

export interface ScanResult {
	detected: boolean;
	classification: string;
	confidence: number;
	tumorVolume: number;
	wtVolume: number;
	tcVolume: number;
	etVolume: number;
}

export interface MRIScan {
	id: string;
	patientId: string;
	patientName: string;
	scanDate: string;
	modalities: Modality[];
	status: ScanStatus;
	progress: number;
	pipelineStep: PipelineStep;
	results?: ScanResult;
}

export const mockScans: MRIScan[] = [];

export const getStatusColor = (status: ScanStatus) => {
	switch (status) {
		case "completed":
			return "success";
		case "processing":
			return "medical";
		case "queued":
			return "info";
		case "failed":
			return "destructive";
		case "uploading":
			return "warning";
		default:
			return "muted";
	}
};

export const getStatusLabel = (status: ScanStatus) => {
	switch (status) {
		case "completed":
			return "Completed";
		case "processing":
			return "Processing";
		case "queued":
			return "Queued";
		case "failed":
			return "Failed";
		case "uploading":
			return "Uploading";
		default:
			return status;
	}
};
