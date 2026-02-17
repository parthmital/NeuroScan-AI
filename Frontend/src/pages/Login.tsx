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
		} catch (error: any) {
			toast.error(error.message);
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="min-h-screen flex items-center justify-center bg-background/50 relative overflow-hidden">
			{/* Background Decorations */}
			<div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] bg-medical/5 rounded-full blur-[120px]" />
			<div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 rounded-full blur-[120px]" />

			<Card className="w-full max-w-md border-border/40 bg-card/50 backdrop-blur-xl shadow-2xl">
				<CardHeader className="space-y-3 text-center">
					<div className="flex justify-center mb-2">
						<div className="w-12 h-12 rounded-2xl bg-medical/20 flex items-center justify-center text-medical shadow-lg shadow-medical/10">
							<BrainCircuit className="w-6 h-6" />
						</div>
					</div>
					<CardTitle className="text-3xl font-bold tracking-tight">
						NeuroScan AI
					</CardTitle>
					<CardDescription className="text-muted-foreground text-base">
						Advanced medical imaging analysis platform
					</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} className="space-y-5">
						<div className="space-y-2">
							<Label htmlFor="username">Username</Label>
							<Input
								id="username"
								type="text"
								placeholder="drrebecca"
								value={username}
								onChange={(e) => setUsername(e.target.value)}
								required
								className="bg-background/50 border-border/50 focus:border-medical/50"
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
								className="bg-background/50 border-border/50 focus:border-medical/50"
							/>
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
									Sign In
									<ArrowRight className="ml-2 w-4 h-4" />
								</>
							)}
						</Button>
					</form>
				</CardContent>
				<CardFooter className="flex flex-col space-y-4 pt-2">
					<div className="text-sm text-center text-muted-foreground">
						Don't have an account?{" "}
						<Link
							to="/register"
							className="text-medical font-medium hover:underline"
						>
							Create an account
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

export default Login;
