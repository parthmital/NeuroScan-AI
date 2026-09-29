import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useDeleteScan, useUpdateScan } from "@/hooks/use-scans";
import type { MRIScan } from "@/lib/types";

interface ScanDialogProps {
	scan: MRIScan | null;
	onClose: () => void;
}

const errorText = (error: unknown) =>
	error instanceof Error ? error.message : "Something went wrong. Try again.";

export const EditScanDialog = ({ scan, onClose }: ScanDialogProps) => {
	const [form, setForm] = useState({ patientName: "", patientId: "" });
	const [error, setError] = useState<string | null>(null);
	const updateScan = useUpdateScan();
	const [syncedScan, setSyncedScan] = useState<MRIScan | null>(null);

	if (scan !== syncedScan) {
		setSyncedScan(scan);
		if (scan) {
			setForm({ patientName: scan.patientName, patientId: scan.patientId });
			setError(null);
		}
	}

	const save = async (event: React.FormEvent) => {
		event.preventDefault();
		if (!scan) return;
		if (!form.patientName.trim() || !form.patientId.trim()) {
			setError("Patient name and patient ID are required.");
			return;
		}
		try {
			await updateScan.mutateAsync({ id: scan.id, ...form });
			toast.success("Scan details updated");
			onClose();
		} catch (err) {
			setError(errorText(err));
		}
	};

	return (
		<Dialog open={scan !== null} onOpenChange={(open) => !open && onClose()}>
			<DialogContent onClick={(e) => e.stopPropagation()}>
				<form onSubmit={save} noValidate className="grid gap-4">
					<DialogHeader>
						<DialogTitle>Edit Scan Details</DialogTitle>
						<DialogDescription>
							Update patient information for this scan.
						</DialogDescription>
					</DialogHeader>
					<div className="space-y-4">
						<div className="space-y-1.5">
							<Label htmlFor="edit-patientName">Patient Name</Label>
							<Input
								id="edit-patientName"
								value={form.patientName}
								maxLength={120}
								onChange={(e) =>
									setForm({ ...form, patientName: e.target.value })
								}
							/>
						</div>
						<div className="space-y-1.5">
							<Label htmlFor="edit-patientId">Patient ID</Label>
							<Input
								id="edit-patientId"
								value={form.patientId}
								maxLength={64}
								onChange={(e) =>
									setForm({ ...form, patientId: e.target.value })
								}
							/>
						</div>
						{error && (
							<p
								role="alert"
								className="animate-fade-in text-[13px] text-destructive"
							>
								{error}
							</p>
						)}
					</div>
					<DialogFooter>
						<Button type="button" variant="outline" onClick={onClose}>
							Cancel
						</Button>
						<Button type="submit" disabled={updateScan.isPending}>
							{updateScan.isPending ? "Saving..." : "Save Changes"}
						</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
};

export const DeleteScanDialog = ({ scan, onClose }: ScanDialogProps) => {
	const deleteScan = useDeleteScan();

	const confirmDelete = async () => {
		if (!scan) return;
		try {
			await deleteScan.mutateAsync(scan.id);
			toast.success("Scan deleted");
			onClose();
		} catch (err) {
			toast.error(errorText(err));
		}
	};

	return (
		<Dialog open={scan !== null} onOpenChange={(open) => !open && onClose()}>
			<DialogContent onClick={(e) => e.stopPropagation()}>
				<DialogHeader>
					<DialogTitle>Delete this scan?</DialogTitle>
					<DialogDescription>
						{scan
							? `The MRI files and results for ${scan.patientName} (${scan.patientId}) will be permanently deleted.`
							: ""}
					</DialogDescription>
				</DialogHeader>
				<DialogFooter>
					<Button variant="outline" onClick={onClose}>
						Cancel
					</Button>
					<Button
						variant="destructive"
						onClick={confirmDelete}
						disabled={deleteScan.isPending}
					>
						{deleteScan.isPending ? "Deleting..." : "Delete Scan"}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
};
