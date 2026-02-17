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
		} catch (error: any) {
			toast.error(error.message);
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="min-h-screen flex items-center justify-center bg-background/50 relative overflow-hidden py-12 px-4">
			{/* Background Decorations */}
			<div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] bg-medical/5 rounded-full blur-[120px]" />
			<div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 rounded-full blur-[120px]" />

			<Card className="w-full max-w-2xl border-border/40 bg-card/50 backdrop-blur-xl shadow-2xl">
				<CardHeader className="space-y-3 text-center">
					<div className="flex justify-center mb-2">
						<div className="w-12 h-12 rounded-2xl bg-medical/20 flex items-center justify-center text-medical shadow-lg shadow-medical/10">
							<UserPlus className="w-6 h-6" />
						</div>
					</div>
					<CardTitle className="text-3xl font-bold tracking-tight">
						Create Medical Account
					</CardTitle>
					<CardDescription className="text-muted-foreground text-base">
						Join the network of neuroradiology professionals
					</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} className="space-y-6">
						<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
							<div className="space-y-2">
								<Label htmlFor="fullName">Full Name</Label>
								<Input
									id="fullName"
									placeholder="Dr. Alex Rivera"
									value={formData.fullName}
									onChange={handleChange}
									required
									className="bg-background/50 border-border/50 focus:border-medical/50"
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
									className="bg-background/50 border-border/50 focus:border-medical/50"
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
									className="bg-background/50 border-border/50 focus:border-medical/50"
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
									className="bg-background/50 border-border/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="title">Professional Title</Label>
								<Input
									id="title"
									placeholder="Chief Radiologist"
									value={formData.title}
									onChange={handleChange}
									className="bg-background/50 border-border/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2">
								<Label htmlFor="department">Department</Label>
								<Input
									id="department"
									placeholder="Neuroradiology"
									value={formData.department}
									onChange={handleChange}
									className="bg-background/50 border-border/50 focus:border-medical/50"
								/>
							</div>
							<div className="space-y-2 md:col-span-2">
								<Label htmlFor="institution">Medical Institution</Label>
								<Input
									id="institution"
									placeholder="General Hospital Center"
									value={formData.institution}
									onChange={handleChange}
									className="bg-background/50 border-border/50 focus:border-medical/50"
								/>
							</div>
						</div>

						<Button
							type="submit"
							className="w-full bg-medical hover:bg-medical/90 text-white shadow-lg shadow-medical/20 h-11 text-base font-medium"
							disabled={isLoading}
						>
							{isLoading ? (
								<Loader2 className="w-5 h-5" />
							) : (
								<>
									Complete Registration
									<ArrowRight className="ml-2 w-4 h-4" />
								</>
							)}
						</Button>
					</form>
				</CardContent>
				<CardFooter className="flex flex-col space-y-4 pt-2">
					<div className="text-sm text-center text-muted-foreground">
						Already have an account?{" "}
						<Link
							to="/login"
							className="text-medical font-medium hover:underline"
						>
							Sign in instead
						</Link>
					</div>
					<div className="border-t border-border/50 w-full pt-4 flex items-center justify-center space-x-2 text-[10px] text-muted-foreground uppercase tracking-widest">
						<ShieldCheck className="w-3 h-3" />
						<span>Secure Medical-Grade Authentication</span>
					</div>
				</CardFooter>
			</Card>
		</div>
	);
};

export default Register;
