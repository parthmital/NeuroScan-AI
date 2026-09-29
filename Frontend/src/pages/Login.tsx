import React, { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth/useAuth";
import { AuthShell } from "@/components/layout/AuthShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

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
		<AuthShell
			title={needsOtp ? "Two-step verification" : "Sign in"}
			description={
				needsOtp
					? "Enter the code from your authenticator app to continue."
					: "Use your NeuroScan account to continue."
			}
			footer="Forgot your password or lost your authenticator? Ask your NeuroScan administrator to reset it."
		>
			<form onSubmit={handleSubmit} noValidate className="space-y-4">
				<div className="space-y-1.5">
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
					/>
				</div>
				<div className="space-y-1.5">
					<Label htmlFor="password">Password</Label>
					<Input
						id="password"
						type="password"
						value={password}
						onChange={(e) => setPassword(e.target.value)}
						autoComplete="current-password"
						disabled={needsOtp}
					/>
				</div>
				{needsOtp && (
					<div className="animate-fade-in space-y-1.5">
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
							className="font-mono tracking-[0.3em]"
						/>
						<p className="text-xs text-muted-foreground">
							Enter the 6-digit code from your authenticator app.
						</p>
					</div>
				)}
				{error && (
					<p
						role="alert"
						className="animate-fade-in rounded-md border border-destructive/30 bg-destructive/[0.06] px-3 py-2 text-[13px] text-destructive"
					>
						{error}
					</p>
				)}
				<Button
					type="submit"
					size="lg"
					className="mt-2 w-full"
					disabled={isLoading}
				>
					{isLoading ? (
						<Loader2 className="animate-spin" />
					) : (
						<>
							{needsOtp ? "Verify and Sign In" : "Sign In"}
							<ArrowRight />
						</>
					)}
				</Button>
				{needsOtp && (
					<Button
						type="button"
						variant="ghost"
						className="w-full"
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
			<p className="mt-8 border-t border-border pt-6 text-sm text-muted-foreground">
				Don't have an account?{" "}
				<Link
					to="/register"
					className="font-medium text-foreground underline decoration-border underline-offset-4 transition-colors hover:decoration-foreground"
				>
					Create an account
				</Link>
			</p>
		</AuthShell>
	);
};

export default Login;
