import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Loader2, UserPlus } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth/useAuth";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { passwordProblem } from "@/lib/password";

const inputClass = "border-border/50 bg-background/50 focus:border-medical/50";

type Field =
	| "fullName"
	| "email"
	| "username"
	| "password"
	| "title"
	| "department"
	| "institution";

const FIELDS: {
	id: Field;
	label: string;
	placeholder?: string;
	type?: string;
	autoComplete: string;
	required?: boolean;
	wide?: boolean;
}[] = [
	{
		id: "fullName",
		label: "Full Name",
		placeholder: "Dr. Alex Rivera",
		autoComplete: "name",
		required: true,
	},
	{
		id: "email",
		label: "Email Address",
		placeholder: "alex.rivera@hospital.org",
		type: "email",
		autoComplete: "email",
		required: true,
	},
	{
		id: "username",
		label: "Username",
		placeholder: "arivera",
		autoComplete: "username",
		required: true,
	},
	{
		id: "password",
		label: "Password",
		type: "password",
		autoComplete: "new-password",
		required: true,
	},
	{
		id: "title",
		label: "Professional Title",
		placeholder: "Chief Radiologist",
		autoComplete: "organization-title",
	},
	{
		id: "department",
		label: "Department",
		placeholder: "Neuroradiology",
		autoComplete: "off",
	},
	{
		id: "institution",
		label: "Medical Institution",
		placeholder: "General Hospital Center",
		autoComplete: "organization",
		wide: true,
	},
];

const validate = (form: Record<Field, string>): string | null => {
	if (!form.fullName.trim() || !form.email.trim() || !form.username.trim()) {
		return "Full name, email, and username are required.";
	}
	if (!/^[A-Za-z0-9_.-]{3,32}$/.test(form.username.trim())) {
		return "Username must be 3 to 32 letters, digits, dots, dashes, or underscores.";
	}
	if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) {
		return "Enter a valid email address.";
	}
	return passwordProblem(form.password);
};

const Register = () => {
	const [form, setForm] = useState<Record<Field, string>>({
		username: "",
		email: "",
		password: "",
		fullName: "",
		title: "",
		department: "",
		institution: "",
	});
	const [error, setError] = useState<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const { register } = useAuth();
	const navigate = useNavigate();

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		const problem = validate(form);
		setError(problem);
		if (problem) return;
		setIsLoading(true);
		try {
			await register({
				username: form.username.trim(),
				email: form.email.trim(),
				password: form.password,
				fullName: form.fullName.trim(),
				title: form.title.trim() || undefined,
				department: form.department.trim() || undefined,
				institution: form.institution.trim() || undefined,
			});
			toast.success("Account created successfully!");
			navigate("/");
		} catch (err) {
			setError(err instanceof Error ? err.message : "Registration failed.");
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background/50 px-4 py-12 sm:px-8">
			<Card className="w-full max-w-2xl border-border/40 bg-card/50 shadow-2xl backdrop-blur-xl">
				<CardHeader className="space-y-3 text-center">
					<div className="mb-5 flex justify-center">
						<div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-medical/20 text-medical shadow-lg shadow-medical/10">
							<UserPlus className="h-6 w-6" />
						</div>
					</div>
					<CardTitle className="text-3xl font-bold tracking-tight">
						Create Medical Account
					</CardTitle>
					<CardDescription className="text-base text-muted-foreground">
						Join the network of neuroradiology professionals
					</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} noValidate className="space-y-6">
						<div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
							{FIELDS.map((field) => (
								<div
									key={field.id}
									className={
										field.wide ? "space-y-2 sm:col-span-2" : "space-y-2"
									}
								>
									<Label htmlFor={field.id}>
										{field.label}
										{!field.required && (
											<span className="ml-1 text-xs font-normal text-muted-foreground">
												(optional)
											</span>
										)}
									</Label>
									<Input
										id={field.id}
										type={field.type ?? "text"}
										placeholder={field.placeholder}
										value={form[field.id]}
										onChange={(e) =>
											setForm({ ...form, [field.id]: e.target.value })
										}
										autoComplete={field.autoComplete}
										spellCheck={false}
										maxLength={field.id === "password" ? 72 : 120}
										className={inputClass}
									/>
									{field.id === "password" && (
										<p className="text-xs text-muted-foreground">
											At least 12 characters.
										</p>
									)}
								</div>
							))}
						</div>
						{error && (
							<p
								role="alert"
								className="animate-fade-in text-sm text-destructive"
							>
								{error}
							</p>
						)}
						<Button
							type="submit"
							className="h-11 w-full bg-medical text-base font-medium text-white shadow-lg shadow-medical/20 hover:bg-medical/90"
							disabled={isLoading}
						>
							{isLoading ? (
								<Loader2 className="h-5 w-5 animate-spin" />
							) : (
								<>
									Complete Registration
									<ArrowRight className="ml-2 h-4 w-4" />
								</>
							)}
						</Button>
					</form>
				</CardContent>
				<CardFooter className="flex flex-col space-y-4 pt-2">
					<div className="text-center text-sm text-muted-foreground">
						Already have an account?{" "}
						<Link
							to="/login"
							className="font-medium text-medical hover:underline"
						>
							Sign in instead
						</Link>
					</div>
				</CardFooter>
			</Card>
		</div>
	);
};

export default Register;
