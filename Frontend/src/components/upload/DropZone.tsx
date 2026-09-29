import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
	AlertCircle,
	CheckCircle2,
	FileImage,
	Loader2,
	Upload,
	X,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useUploadScan } from "@/hooks/use-scans";
import type { Modality } from "@/lib/types";

interface UploadedFile {
	id: string;
	name: string;
	size: number;
	modality?: Modality;
	status: "ready" | "processing" | "done" | "error";
	file: File;
}

const MODALITIES: Modality[] = ["FLAIR", "T1", "T1CE", "T2"];
const ACCEPTED = /\.nii(\.gz)?$/i;

/** Same file-name rules the server uses to assign modalities. */
const detectModality = (filename: string): Modality | undefined => {
	const upper = filename.toUpperCase();
	if (upper.includes("FLAIR")) return "FLAIR";
	if (upper.includes("T1CE") || upper.includes("T1C")) return "T1CE";
	if (upper.includes("T2")) return "T2";
	if (upper.includes("T1")) return "T1";
	return undefined;
};

export const DropZone = () => {
	const navigate = useNavigate();
	const fileInputRef = useRef<HTMLInputElement>(null);
	const uploadScan = useUploadScan();
	const [isDragging, setIsDragging] = useState(false);
	const [files, setFiles] = useState<UploadedFile[]>([]);
	const [selectedModalities, setSelectedModalities] = useState<Set<Modality>>(
		new Set(MODALITIES),
	);

	const handleDragOver = useCallback((e: React.DragEvent) => {
		e.preventDefault();
		setIsDragging(true);
	}, []);

	const handleDragLeave = useCallback(() => {
		setIsDragging(false);
	}, []);

	const addFiles = useCallback((newFiles: File[]) => {
		const rejected = newFiles.filter((f) => !ACCEPTED.test(f.name));
		if (rejected.length > 0) {
			toast.error(
				`Only NIfTI files (.nii, .nii.gz) are supported: ${rejected.map((f) => f.name).join(", ")}`,
			);
		}
		const uploadFiles: UploadedFile[] = newFiles
			.filter((f) => ACCEPTED.test(f.name))
			.map((f) => ({
				id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
				name: f.name,
				size: f.size,
				modality: detectModality(f.name),
				status: "ready" as const,
				file: f,
			}));
		setFiles((prev) => [...prev, ...uploadFiles]);
	}, []);

	const handleDrop = useCallback(
		(e: React.DragEvent) => {
			e.preventDefault();
			setIsDragging(false);
			const droppedFiles = Array.from(e.dataTransfer.files);
			addFiles(droppedFiles);
		},
		[addFiles],
	);

	const handleFileInput = useCallback(
		(e: React.ChangeEvent<HTMLInputElement>) => {
			if (e.target.files) {
				addFiles(Array.from(e.target.files));
			}
			// Allow choosing the same file again after removing it.
			e.target.value = "";
		},
		[addFiles],
	);

	const removeFile = (id: string) => {
		setFiles((prev) => prev.filter((f) => f.id !== id));
	};

	const toggleModality = (mod: Modality) => {
		setSelectedModalities((prev) => {
			const next = new Set(prev);
			if (next.has(mod)) next.delete(mod);
			else next.add(mod);
			return next;
		});
	};

	const formatSize = (bytes: number) => {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	};

	const [patientName, setPatientName] = useState("");
	const [patientId, setPatientId] = useState("");

	const handleStartProcessing = async () => {
		if (files.length === 0) return;
		setFiles((prev) => prev.map((f) => ({ ...f, status: "processing" })));
		try {
			const scan = await uploadScan.mutateAsync({
				files: files.map((f) => f.file),
				patientName: patientName.trim() || "Uploaded Scan",
				patientId: patientId.trim(),
			});
			setFiles((prev) => prev.map((f) => ({ ...f, status: "done" })));
			toast.success("Upload complete. Processing has started.");
			navigate(`/scan/${scan.id}`);
		} catch (error) {
			setFiles((prev) => prev.map((f) => ({ ...f, status: "error" })));
			toast.error(
				error instanceof Error ? error.message : "Upload failed. Try again.",
			);
		}
	};

	const missingModalities = MODALITIES.filter(
		(m) => selectedModalities.has(m) && !files.some((f) => f.modality === m),
	);

	return (
		<div className="space-y-6">
			{/* Drop area */}
			<div
				onDragOver={handleDragOver}
				onDragLeave={handleDragLeave}
				onDrop={handleDrop}
				className={cn(
					"group relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-200",
					isDragging
						? "scale-[1.01] border-medical bg-medical-light/50"
						: "border-border hover:border-medical/40 hover:bg-muted/30",
				)}
				onClick={() => fileInputRef.current?.click()}
				onKeyDown={(e) => {
					if (e.key === "Enter" || e.key === " ") {
						e.preventDefault();
						fileInputRef.current?.click();
					}
				}}
				role="button"
				tabIndex={0}
				aria-label="Choose MRI files to upload"
			>
				<input
					ref={fileInputRef}
					type="file"
					multiple
					accept=".nii,.gz"
					className="hidden"
					onChange={handleFileInput}
				/>
				<div
					className={cn(
						"mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl transition-colors",
						isDragging
							? "bg-medical text-medical-foreground"
							: "bg-muted text-muted-foreground group-hover:bg-medical/10 group-hover:text-medical",
					)}
				>
					<Upload className="h-7 w-7" />
				</div>
				<h3 className="mb-1 text-lg font-semibold text-foreground">
					Drop your MRI files here
				</h3>
				<p className="mb-3 text-sm text-muted-foreground">
					or click to browse. Supports NIfTI volumes (.nii, .nii.gz).
				</p>
				<p className="text-xs text-muted-foreground/70">
					Batch upload supported · Multi-modal MRI (FLAIR, T1, T1CE, T2)
				</p>
			</div>

			{/* Patient Details */}
			{files.length > 0 && (
				<div className="grid animate-fade-in grid-cols-1 gap-4 rounded-2xl border border-border bg-muted/30 p-5 sm:grid-cols-2">
					<div className="space-y-2">
						<label
							htmlFor="patient-name"
							className="text-xs font-semibold uppercase tracking-wider text-muted-foreground"
						>
							Patient Name
						</label>
						<Input
							id="patient-name"
							maxLength={120}
							placeholder="e.g. John Doe"
							value={patientName}
							onChange={(e) => setPatientName(e.target.value)}
							className="border-border/50 bg-background"
						/>
					</div>
					<div className="space-y-2">
						<label
							htmlFor="patient-id"
							className="text-xs font-semibold uppercase tracking-wider text-muted-foreground"
						>
							Patient ID (Optional)
						</label>
						<Input
							id="patient-id"
							maxLength={64}
							placeholder="e.g. PT-10492"
							value={patientId}
							onChange={(e) => setPatientId(e.target.value)}
							className="border-border/50 bg-background"
						/>
					</div>
				</div>
			)}

			{/* Modality selector */}
			<div>
				<p className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground">
					Expected Modalities
				</p>
				<div className="flex gap-2">
					{MODALITIES.map((mod) => (
						<button
							key={mod}
							onClick={() => toggleModality(mod)}
							aria-pressed={selectedModalities.has(mod)}
							className={cn(
								"rounded-lg px-4 py-2 text-sm font-medium transition-all",
								selectedModalities.has(mod)
									? "bg-medical text-medical-foreground shadow-xs"
									: "bg-muted text-muted-foreground hover:bg-muted/80",
							)}
						>
							{mod}
						</button>
					))}
				</div>
			</div>

			{/* Missing modality warning */}
			{files.length > 0 && missingModalities.length > 0 && (
				<div className="flex animate-fade-in items-start gap-3 rounded-xl border border-warning/20 bg-warning/10 p-4">
					<AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
					<div>
						<p className="text-sm font-medium text-foreground">
							Missing modalities detected
						</p>
						<p className="mt-0.5 text-xs text-muted-foreground">
							The following expected modalities have not been uploaded:{" "}
							<strong>{missingModalities.join(", ")}</strong>. This may affect
							processing accuracy.
						</p>
					</div>
				</div>
			)}

			{/* File list */}
			{files.length > 0 && (
				<div className="space-y-2">
					<div className="flex items-center justify-between">
						<p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
							Uploaded Files ({files.length})
						</p>
						<Button
							variant="ghost"
							size="sm"
							className="text-xs text-muted-foreground"
							onClick={() => setFiles([])}
						>
							Clear All
						</Button>
					</div>
					<div className="space-y-1.5">
						{files.map((file) => (
							<div
								key={file.id}
								className="group flex animate-fade-in items-center gap-3 rounded-lg border border-border bg-muted/30 p-3"
							>
								<FileImage className="h-4 w-4 shrink-0 text-medical" />
								<div className="min-w-0 flex-1">
									<p className="truncate text-sm font-medium text-foreground">
										{file.name}
									</p>
									<p className="text-xs text-muted-foreground">
										{formatSize(file.size)}
										{file.modality && (
											<span className="ml-2 rounded bg-medical/10 px-1.5 py-0.5 text-[10px] font-semibold text-medical">
												{file.modality}
											</span>
										)}
									</p>
								</div>
								{file.status === "done" && (
									<CheckCircle2 className="h-4 w-4 text-success" />
								)}
								{file.status === "error" && (
									<AlertCircle
										className="h-4 w-4 text-destructive"
										aria-label="Upload failed"
									/>
								)}
								<button
									onClick={() => removeFile(file.id)}
									aria-label={`Remove ${file.name}`}
									className="rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100"
								>
									<X className="h-3.5 w-3.5" />
								</button>
							</div>
						))}
					</div>
				</div>
			)}

			{/* Action */}
			{files.length > 0 && (
				<div className="flex animate-fade-in justify-end">
					<Button
						onClick={handleStartProcessing}
						disabled={files.some((f) => f.status === "processing")}
						className="gap-2 bg-medical px-8 text-medical-foreground hover:bg-medical/90"
					>
						{files.some((f) => f.status === "processing") ? (
							<Loader2 className="h-4 w-4 animate-spin" />
						) : (
							<Upload className="h-4 w-4" />
						)}
						{files.some((f) => f.status === "processing")
							? "Processing..."
							: `Start Processing (${files.length} file${files.length > 1 ? "s" : ""})`}
					</Button>
				</div>
			)}
		</div>
	);
};
