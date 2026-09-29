import { useState } from "react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authApi } from "@/lib/api";
import { passwordProblem } from "@/lib/password";

const EMPTY = { current: "", next: "", confirm: "" };

export const PasswordForm = () => {
	const { setUser } = useAuth();
	const [form, setForm] = useState(EMPTY);
	const [error, setError] = useState<string | null>(null);
	const [saving, setSaving] = useState(false);

	const submit = async (event: React.FormEvent) => {
		event.preventDefault();
		const problem = !form.current
			? "Enter your current password."
			: (passwordProblem(form.next) ??
				(form.next !== form.confirm ? "New passwords do not match." : null));
		setError(problem);
		if (problem) return;
		setSaving(true);
		try {
			setUser(await authApi.changePassword(form.current, form.next));
			setForm(EMPTY);
			toast.success("Password changed. Other sessions were signed out.");
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Could not change password.",
			);
		} finally {
			setSaving(false);
		}
	};

	const field = (
		key: keyof typeof EMPTY,
		label: string,
		autoComplete: string,
	) => (
		<div className="space-y-1.5">
			<Label htmlFor={`password-${key}`}>{label}</Label>
			<Input
				id={`password-${key}`}
				type="password"
				autoComplete={autoComplete}
				maxLength={key === "current" ? 256 : 72}
				value={form[key]}
				onChange={(e) => setForm({ ...form, [key]: e.target.value })}
			/>
		</div>
	);

	return (
		<form onSubmit={submit} noValidate className="space-y-4">
			<div>
				<h2 className="mb-1 text-sm font-semibold text-foreground">
					Change Password
				</h2>
				<p className="text-xs text-muted-foreground">
					At least 12 characters. Changing it signs out your other sessions.
				</p>
			</div>
			<div className="space-y-3">
				{field("current", "Current Password", "current-password")}
				{field("next", "New Password", "new-password")}
				{field("confirm", "Confirm New Password", "new-password")}
			</div>
			{error && (
				<p role="alert" className="animate-fade-in text-xs text-destructive">
					{error}
				</p>
			)}
			<div className="flex justify-end">
				<Button type="submit" disabled={saving}>
					{saving ? "Updating..." : "Update Password"}
				</Button>
			</div>
		</form>
	);
};
