import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
	AlertTriangle,
	ArrowUpFromLine,
	Check,
	CheckCircle2,
	CircleAlert,
	FileBox,
	Loader2,
	X,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
const sectionLabel =
	"text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground";
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
		<div className="space-y-8">
			{/* Drop area */}
			<div
				onDragOver={handleDragOver}
				onDragLeave={handleDragLeave}
				onDrop={handleDrop}
				className={cn(
					"group relative cursor-pointer rounded-lg border border-dashed px-8 py-12 text-center transition-colors duration-150 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
					isDragging
						? "border-medical bg-medical-light"
						: "border-input bg-card hover:border-foreground/30",
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
						"mx-auto mb-4 flex h-10 w-10 items-center justify-center rounded-md border transition-colors",
						isDragging
							? "border-medical/40 text-medical"
							: "border-border bg-muted/60 text-muted-foreground group-hover:text-foreground",
					)}
				>
					<ArrowUpFromLine className="h-5 w-5" />
				</div>
				<h3 className="text-[15px] font-semibold text-foreground">
					Drop MRI volumes here, or{" "}
					<span className="text-medical underline-offset-4 group-hover:underline">
						browse files
					</span>
				</h3>
				<p className="mt-1.5 text-[13px] text-muted-foreground">
					NIfTI volumes (<span className="font-mono">.nii</span>,{" "}
					<span className="font-mono">.nii.gz</span>). Batch upload of FLAIR,
					T1, T1CE, and T2 is supported.
				</p>
			</div>

			{/* Patient Details */}
			{files.length > 0 && (
				<div className="animate-fade-in">
					<p className={cn(sectionLabel, "mb-3")}>Patient</p>
					<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
						<div className="space-y-1.5">
							<Label htmlFor="patient-name">Patient name</Label>
							<Input
								id="patient-name"
								maxLength={120}
								placeholder="e.g. John Doe"
								value={patientName}
								onChange={(e) => setPatientName(e.target.value)}
							/>
						</div>
						<div className="space-y-1.5">
							<Label htmlFor="patient-id">
								Patient ID
								<span className="ml-1.5 font-normal text-muted-foreground">
									Optional
								</span>
							</Label>
							<Input
								id="patient-id"
								maxLength={64}
								placeholder="e.g. PT-10492"
								value={patientId}
								onChange={(e) => setPatientId(e.target.value)}
								className="font-mono"
							/>
						</div>
					</div>
				</div>
			)}

			{/* Modality selector */}
			<div>
				<p className={cn(sectionLabel, "mb-3")}>Expected modalities</p>
				<div className="flex flex-wrap gap-2">
					{MODALITIES.map((mod) => {
						const selected = selectedModalities.has(mod);
						return (
							<button
								key={mod}
								type="button"
								onClick={() => toggleModality(mod)}
								aria-pressed={selected}
								className={cn(
									"inline-flex h-8 items-center gap-1.5 rounded-md border px-3 font-mono text-xs font-medium transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring",
									selected
										? "border-medical/40 bg-medical-light text-medical"
										: "border-input bg-card text-muted-foreground hover:border-foreground/25 hover:text-foreground",
								)}
							>
								<Check
									className={cn(
										"h-3.5 w-3.5 transition-opacity",
										selected ? "opacity-100" : "opacity-0",
									)}
								/>
								{mod}
							</button>
						);
					})}
				</div>
			</div>

			{/* Missing modality warning */}
			{files.length > 0 && missingModalities.length > 0 && (
				<div className="flex animate-fade-in items-start gap-3 rounded-md border border-warning/30 bg-warning/[0.06] px-4 py-3">
					<AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
					<div>
						<p className="text-[13px] font-medium text-foreground">
							Missing modalities
						</p>
						<p className="mt-0.5 text-[13px] text-muted-foreground">
							The following expected modalities have not been uploaded:{" "}
							<strong className="font-mono font-medium text-foreground">
								{missingModalities.join(", ")}
							</strong>
							. This may affect processing accuracy.
						</p>
					</div>
				</div>
			)}

			{/* File list */}
			{files.length > 0 && (
				<div>
					<div className="mb-2 flex items-center justify-between">
						<p className={sectionLabel}>
							Files{" "}
							<span className="font-mono tabular-nums">({files.length})</span>
						</p>
						<Button
							variant="ghost"
							size="sm"
							className="-mr-2.5"
							onClick={() => setFiles([])}
						>
							Clear all
						</Button>
					</div>
					<ul className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-card">
						{files.map((file) => (
							<li
								key={file.id}
								className="flex animate-fade-in items-center gap-3 px-4 py-2.5"
							>
								<FileBox className="h-4 w-4 shrink-0 text-muted-foreground" />
								<p className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-foreground">
									{file.name}
								</p>
								{file.modality ? (
									<span className="rounded-[4px] border border-medical/30 bg-medical-light px-1.5 py-px font-mono text-[10.5px] font-medium text-medical">
										{file.modality}
									</span>
								) : (
									<span className="font-mono text-[10.5px] text-muted-foreground">
										Unassigned
									</span>
								)}
								<span className="hidden w-16 shrink-0 text-right font-mono text-[11px] text-muted-foreground tabular-nums sm:inline">
									{formatSize(file.size)}
								</span>
								{file.status === "done" && (
									<CheckCircle2 className="h-4 w-4 text-success" />
								)}
								{file.status === "error" && (
									<CircleAlert
										className="h-4 w-4 text-destructive"
										aria-label="Upload failed"
									/>
								)}
								<button
									onClick={() => removeFile(file.id)}
									aria-label={`Remove ${file.name}`}
									className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
								>
									<X className="h-3.5 w-3.5" />
								</button>
							</li>
						))}
					</ul>
				</div>
			)}

			{/* Action */}
			{files.length > 0 && (
				<div className="flex animate-fade-in justify-end border-t border-border pt-6">
					<Button
						size="lg"
						onClick={handleStartProcessing}
						disabled={files.some((f) => f.status === "processing")}
					>
						{files.some((f) => f.status === "processing") ? (
							<Loader2 className="animate-spin" />
						) : (
							<ArrowUpFromLine />
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
