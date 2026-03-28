import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/components/auth/AuthContext";
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
import { BrainCircuit, Loader2, ArrowRight, ShieldCheck } from "lucide-react";
import { toast } from "sonner";

const Login = () => {
	const [username, setUsername] = useState("");
	const [password, setPassword] = useState("");
	const [isLoading, setIsLoading] = useState(false);
	const { login } = useAuth();
	const navigate = useNavigate();

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		setIsLoading(true);

		try {
			const formData = new URLSearchParams();
			formData.append("username", username);
			formData.append("password", password);

			const response = await fetch("http://localhost:8000/api/auth/login", {
				method: "POST",
				headers: {
					"Content-Type": "application/x-www-form-urlencoded",
				},
				body: formData,
			});

			if (!response.ok) {
				const error = await response.json();
				throw new Error(error.detail || "Login failed");
			}

			const data = await response.json();
			login(data.access_token, data.user);
			toast.success("Welcome back, " + data.user.fullName);
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
		<div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background/50">
			{/* Background Decorations */}
			<div className="absolute right-[-10%] top-[-10%] h-[40%] w-[40%] rounded-full bg-medical/5 blur-[120px]" />
			<div className="absolute bottom-[-10%] left-[-10%] h-[40%] w-[40%] rounded-full bg-blue-500/5 blur-[120px]" />

			<Card className="w-full max-w-md border-border/40 bg-card/50 shadow-2xl backdrop-blur-xl">
				<CardHeader className="space-y-3 text-center">
					<div className="mb-2 flex justify-center">
						<div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-medical/20 text-medical shadow-lg shadow-medical/10">
							<BrainCircuit className="h-6 w-6" />
						</div>
					</div>
					<CardTitle className="text-3xl font-bold tracking-tight">
						NeuroScan AI
					</CardTitle>
					<CardDescription className="text-base text-muted-foreground">
						Advanced medical imaging analysis platform
					</CardDescription>
				</CardHeader>
				<CardContent>
					<form
						onSubmit={handleSubmit}
						className="space-y-5"
						autoComplete="off"
					>
						<div className="space-y-2">
							<Label htmlFor="username">Username</Label>
							<Input
								id="username"
								type="text"
								placeholder="drrebecca"
								value={username}
								onChange={(e) => setUsername(e.target.value)}
								required
								autoComplete="off"
								spellCheck={false}
								className="border-border/50 bg-background/50 focus:border-medical/50"
							/>
						</div>
						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<Label htmlFor="password">Password</Label>
								<a href="#" className="text-xs text-medical hover:underline">
									Forgot password?
								</a>
							</div>
							<Input
								id="password"
								type="password"
								placeholder="••••••••"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								required
								autoComplete="new-password"
								spellCheck={false}
								className="border-border/50 bg-background/50 focus:border-medical/50"
							/>
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
									Sign In
									<ArrowRight className="ml-2 h-4 w-4" />
								</>
							)}
						</Button>
					</form>
				</CardContent>
				<CardFooter className="flex flex-col space-y-4 pt-2">
					<div className="text-center text-sm text-muted-foreground">
						Don't have an account?{" "}
						<Link
							to="/register"
							className="font-medium text-medical hover:underline"
						>
							Create an account
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

export default Login;
