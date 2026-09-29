import { useState } from "react";
import { Save } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authApi } from "@/lib/api";
import type { User } from "@/lib/types";

type ProfileField =
	| "firstName"
	| "lastName"
	| "email"
	| "title"
	| "department"
	| "institution";

const FIELDS: {
	id: ProfileField;
	label: string;
	autoComplete: string;
	placeholder?: string;
	wide?: boolean;
}[] = [
	{ id: "firstName", label: "First Name", autoComplete: "given-name" },
	{ id: "lastName", label: "Last Name", autoComplete: "family-name" },
	{ id: "email", label: "Email", autoComplete: "email" },
	{
		id: "title",
		label: "Professional Title",
		autoComplete: "organization-title",
		placeholder: "e.g. Chief Neuroradiologist",
	},
	{ id: "department", label: "Department", autoComplete: "off" },
	{
		id: "institution",
		label: "Institution",
		autoComplete: "organization",
		wide: true,
	},
];

const fromUser = (user: User): Record<ProfileField, string> => ({
	firstName: user.fullName.split(" ")[0] || "",
	lastName: user.fullName.split(" ").slice(1).join(" ") || "",
	email: user.email || "",
	title: user.title || "",
	department: user.department || "",
	institution: user.institution || "",
});

export const ProfileForm = ({ user }: { user: User }) => {
	const { setUser } = useAuth();
	const [form, setForm] = useState(() => fromUser(user));
	const [error, setError] = useState<string | null>(null);
	const [saving, setSaving] = useState(false);
	const [syncedUser, setSyncedUser] = useState(user);

	if (user !== syncedUser) {
		setSyncedUser(user);
		setForm(fromUser(user));
	}

	const save = async (event: React.FormEvent) => {
		event.preventDefault();
		const fullName = `${form.firstName} ${form.lastName}`.trim();
		if (!fullName || !form.email.trim()) {
			setError("Name and email are required.");
			return;
		}
		setError(null);
		setSaving(true);
		try {
			setUser(
				await authApi.updateProfile({
					fullName,
					email: form.email.trim(),
					title: form.title.trim(),
					department: form.department.trim(),
					institution: form.institution.trim(),
				}),
			);
			toast.success("Your profile has been updated.");
		} catch (err) {
			setError(err instanceof Error ? err.message : "Could not save changes.");
		} finally {
			setSaving(false);
		}
	};

	return (
		<form
			onSubmit={save}
			noValidate
			className="space-y-6 rounded-xl border border-border bg-card p-6"
		>
			<h2 className="text-sm font-semibold text-foreground">
				Profile Information
			</h2>
			<div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
				{FIELDS.map((field) => (
					<div
						key={field.id}
						className={field.wide ? "space-y-2 sm:col-span-2" : "space-y-2"}
					>
						<Label
							htmlFor={`profile-${field.id}`}
							className="text-xs leading-6 font-medium text-muted-foreground"
						>
							{field.label}
						</Label>
						<Input
							id={`profile-${field.id}`}
							type={field.id === "email" ? "email" : "text"}
							value={form[field.id]}
							maxLength={120}
							autoComplete={field.autoComplete}
							placeholder={field.placeholder}
							onChange={(e) => setForm({ ...form, [field.id]: e.target.value })}
							className="h-9 text-sm"
						/>
					</div>
				))}
			</div>
			{error && (
				<p role="alert" className="animate-fade-in text-xs text-destructive">
					{error}
				</p>
			)}
			<div className="flex justify-end border-t border-border pt-4">
				<Button
					type="submit"
					disabled={saving}
					className="gap-1.5 bg-medical text-xs text-medical-foreground hover:bg-medical/90"
				>
					<Save className="h-3.5 w-3.5" />
					{saving ? "Saving..." : "Save Changes"}
				</Button>
			</div>
		</form>
	);
};
