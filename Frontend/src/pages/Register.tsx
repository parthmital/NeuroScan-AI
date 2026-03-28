import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
	Card,
	CardHeader,
	CardTitle,
	CardDescription,
	CardContent,
	CardFooter,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
	BrainCircuit,
	Loader2,
	ArrowRight,
	ShieldCheck,
	UserPlus,
} from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth/AuthContext";

const Register = () => {
	const [formData, setFormData] = useState({
		username: "",
		email: "",
		password: "",
		fullName: "",
		title: "",
		department: "",
		institution: "",
	});
	const [isLoading, setIsLoading] = useState(false);
	const { login } = useAuth();
	const navigate = useNavigate();

	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		setFormData({ ...formData, [e.target.id]: e.target.value });
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		setIsLoading(true);

		try {
			const response = await fetch("http://localhost:8000/api/auth/register", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(formData),
			});

			if (!response.ok) {
				const error = await response.json();
				throw new Error(error.detail || "Registration failed");
			}

			const data = await response.json();
			login(data.access_token, data.user);

			toast.success("Account created successfully!");
			navigate("/");
		} catch (error: unknown) {
			const errorMessage =
				error instanceof Error ? error.message : "An unknown error occurred";
			toast.error(errorMessage);
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background/50 px-8 py-12">
			{/* Background Decorations */}
			<div className="absolute right-[-10%] top-[-10%] h-[40%] w-[40%] rounded-full bg-medical/5 blur-[120px]" />
			<div className="absolute bottom-[-10%] left-[-10%] h-[40%] w-[40%] rounded-full bg-blue-500/5 blur-[120px]" />

			<Card className="w-full max-w-2xl border-border/40 bg-card/50 shadow-2xl backdrop-blur-xl">
				<CardHeader className="space-y-3 text-center">
					<div className="mb-2 flex justify-center">
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
					<form onSubmit={handleSubmit} className="space-y-6">
						<div className="grid grid-cols-2 gap-6">
							<div className="space-y-2">
								<Label htmlFor="fullName">Full Name</Label>
								<Input
									id="fullName"
									placeholder="Dr. Alex Rivera"
									value={formData.fullName}
									onChange={handleChange}
									required
									autoComplete="off"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="email">Email Address</Label>
								<Input
									id="email"
									type="email"
									placeholder="alex.rivera@hospital.org"
									value={formData.email}
									onChange={handleChange}
									required
									autoComplete="off"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="username">Username</Label>
								<Input
									id="username"
									placeholder="arivera"
									value={formData.username}
									onChange={handleChange}
									required
									autoComplete="off"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="password">Password</Label>
								<Input
									id="password"
									type="password"
									placeholder="••••••••"
									value={formData.password}
									onChange={handleChange}
									required
									autoComplete="new-password"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="title">Professional Title</Label>
								<Input
									id="title"
									placeholder="Chief Radiologist"
									value={formData.title}
									onChange={handleChange}
									autoComplete="off"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="department">Department</Label>
								<Input
									id="department"
									placeholder="Neuroradiology"
									value={formData.department}
									onChange={handleChange}
									autoComplete="off"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
							<div className="col-span-2 space-y-2">
								<Label htmlFor="institution">Medical Institution</Label>
								<Input
									id="institution"
									placeholder="General Hospital Center"
									value={formData.institution}
									onChange={handleChange}
									autoComplete="off"
									spellCheck={false}
									className="border-border/50 bg-background/50 focus:border-medical/50"
								/>
							</div>
						</div>

						<Button
							type="submit"
							className="h-11 w-full bg-medical text-base font-medium text-white shadow-lg shadow-medical/20 hover:bg-medical/90"
							disabled={isLoading}
						>
							{isLoading ? (
								<Loader2 className="h-5 w-5" />
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
					<div className="flex w-full items-center justify-center space-x-2 border-t border-border/50 pt-4 text-[10px] uppercase tracking-widest text-muted-foreground">
						<ShieldCheck className="h-3 w-3" />
						<span>Secure Medical-Grade Authentication</span>
					</div>
				</CardFooter>
			</Card>
		</div>
	);
};

export default Register;
