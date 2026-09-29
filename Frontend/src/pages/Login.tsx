import React, { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, BrainCircuit, Loader2, ShieldCheck } from "lucide-react";
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

const inputClass = "border-border/50 bg-background/50 focus:border-medical/50";

/** Only same-app paths are followed after login, never external URLs. */
const safeRedirect = (state: unknown): string => {
	const path = (state as { from?: { pathname?: string } } | null)?.from
		?.pathname;
	return path && path.startsWith("/") && !path.startsWith("//") ? path : "/";
};

const Login = () => {
	const [username, setUsername] = useState("");
	const [password, setPassword] = useState("");
	const [otp, setOtp] = useState("");
	const [needsOtp, setNeedsOtp] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const { login } = useAuth();
	const navigate = useNavigate();
	const location = useLocation();

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		setError(null);
		if (!username.trim() || !password) {
			setError("Enter your username and password.");
			return;
		}
		if (needsOtp && !/^\d{6}$/.test(otp)) {
			setError("Enter the 6-digit code from your authenticator app.");
			return;
		}
		setIsLoading(true);
		try {
			const result = await login(
				username.trim(),
				password,
				needsOtp ? otp : undefined,
			);
			if (result.status === "otp-required") {
				setNeedsOtp(true);
				return;
			}
			toast.success(`Welcome back, ${result.user.fullName}`);
			navigate(safeRedirect(location.state), { replace: true });
		} catch (err) {
			setError(err instanceof Error ? err.message : "Sign-in failed.");
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background/50 px-4">
			<Card className="w-full max-w-md border-border/40 bg-card/50 shadow-2xl backdrop-blur-xl">
				<CardHeader className="space-y-3 text-center">
					<div className="mb-5 flex justify-center">
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
					<form onSubmit={handleSubmit} noValidate className="space-y-5">
						<div className="space-y-2">
							<Label htmlFor="username">Username</Label>
							<Input
								id="username"
								type="text"
								placeholder="drrebecca"
								value={username}
								onChange={(e) => setUsername(e.target.value)}
								autoComplete="username"
								spellCheck={false}
								disabled={needsOtp}
								className={inputClass}
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="password">Password</Label>
							<Input
								id="password"
								type="password"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								autoComplete="current-password"
								disabled={needsOtp}
								className={inputClass}
							/>
						</div>
						{needsOtp && (
							<div className="animate-fade-in space-y-2">
								<Label htmlFor="otp">Authentication code</Label>
								<Input
									id="otp"
									inputMode="numeric"
									autoComplete="one-time-code"
									maxLength={6}
									placeholder="123456"
									value={otp}
									onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))}
									autoFocus
									className={inputClass}
								/>
								<p className="text-xs text-muted-foreground">
									Enter the 6-digit code from your authenticator app.
								</p>
							</div>
						)}
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
									{needsOtp ? "Verify and Sign In" : "Sign In"}
									<ArrowRight className="ml-2 h-4 w-4" />
								</>
							)}
						</Button>
						{needsOtp && (
							<Button
								type="button"
								variant="ghost"
								className="w-full text-xs text-muted-foreground"
								onClick={() => {
									setNeedsOtp(false);
									setOtp("");
									setError(null);
								}}
							>
								Use a different account
							</Button>
						)}
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
					<div className="flex w-full items-center justify-center gap-2 border-t border-border/50 pt-4 text-xs text-muted-foreground">
						<ShieldCheck className="h-5 w-5 shrink-0" />
						<span>
							Forgot your password or lost your authenticator? Ask your
							NeuroScan administrator to reset it.
						</span>
					</div>
				</CardFooter>
			</Card>
		</div>
	);
};

export default Login;
