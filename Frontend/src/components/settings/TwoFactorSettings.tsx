import { useState } from "react";
import { Shield, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/components/auth/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authApi } from "@/lib/api";
import type { TwoFactorSetup, User } from "@/lib/types";

type Step = "idle" | "confirm-password" | "scan-code" | "disable";

const errorText = (error: unknown) =>
	error instanceof Error ? error.message : "Something went wrong. Try again.";

const CodeInput = ({
	id,
	value,
	onChange,
}: {
	id: string;
	value: string;
	onChange: (value: string) => void;
}) => (
	<Input
		id={id}
		inputMode="numeric"
		autoComplete="one-time-code"
		maxLength={6}
		placeholder="123456"
		value={value}
		onChange={(e) => onChange(e.target.value.replace(/\D/g, ""))}
		className="h-9 w-40 text-sm tracking-widest"
	/>
);

export const TwoFactorSettings = ({ user }: { user: User }) => {
	const { setUser } = useAuth();
	const [step, setStep] = useState<Step>("idle");
	const [password, setPassword] = useState("");
	const [code, setCode] = useState("");
	const [setup, setSetup] = useState<TwoFactorSetup | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [busy, setBusy] = useState(false);

	const reset = () => {
		setStep("idle");
		setPassword("");
		setCode("");
		setSetup(null);
		setError(null);
	};

	const run = async (action: () => Promise<void>) => {
		setError(null);
		setBusy(true);
		try {
			await action();
		} catch (err) {
			setError(errorText(err));
		} finally {
			setBusy(false);
		}
	};

	const startSetup = (event: React.FormEvent) => {
		event.preventDefault();
		if (!password) return setError("Enter your password.");
		return run(async () => {
			setSetup(await authApi.startTwoFactorSetup(password));
			setPassword("");
			setStep("scan-code");
		});
	};

	const enable = (event: React.FormEvent) => {
		event.preventDefault();
		if (!/^\d{6}$/.test(code)) return setError("Enter the 6-digit code.");
		return run(async () => {
			setUser(await authApi.enableTwoFactor(code));
			toast.success("Two-factor authentication is on.");
			reset();
		});
	};

	const disable = (event: React.FormEvent) => {
		event.preventDefault();
		if (!password || !/^\d{6}$/.test(code)) {
			return setError("Enter your password and a current 6-digit code.");
		}
		return run(async () => {
			setUser(await authApi.disableTwoFactor(password, code));
			toast.success("Two-factor authentication is off.");
			reset();
		});
	};

	const errorLine = error && (
		<p role="alert" className="animate-fade-in text-xs text-destructive">
			{error}
		</p>
	);

	return (
		<div className="space-y-4">
			<div>
				<h2 className="mb-1 text-sm font-semibold text-foreground">
					Two-Factor Authentication
				</h2>
				<p className="text-xs text-muted-foreground">
					Require a code from an authenticator app when signing in.
				</p>
			</div>

			<div className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-3">
				{user.totpEnabled ? (
					<ShieldCheck className="h-5 w-5 shrink-0 text-success" />
				) : (
					<Shield className="h-5 w-5 shrink-0 text-muted-foreground" />
				)}
				<p className="flex-1 text-sm font-medium text-foreground">
					{user.totpEnabled ? "On" : "Off"}
				</p>
				{step === "idle" && (
					<Button
						type="button"
						variant="outline"
						size="sm"
						className="text-xs"
						onClick={() =>
							setStep(user.totpEnabled ? "disable" : "confirm-password")
						}
					>
						{user.totpEnabled ? "Turn off" : "Set up"}
					</Button>
				)}
			</div>

			{step === "confirm-password" && (
				<form
					onSubmit={startSetup}
					noValidate
					className="animate-fade-in space-y-3"
				>
					<div className="space-y-2">
						<Label
							htmlFor="twofa-password"
							className="text-xs leading-6 text-muted-foreground"
						>
							Confirm your password to continue
						</Label>
						<Input
							id="twofa-password"
							type="password"
							autoComplete="current-password"
							value={password}
							onChange={(e) => setPassword(e.target.value)}
							className="h-9 text-sm"
						/>
					</div>
					{errorLine}
					<div className="flex gap-2">
						<Button
							type="submit"
							size="sm"
							disabled={busy}
							className="bg-medical text-xs text-medical-foreground"
						>
							Continue
						</Button>
						<Button
							type="button"
							size="sm"
							variant="ghost"
							className="text-xs"
							onClick={reset}
						>
							Cancel
						</Button>
					</div>
				</form>
			)}

			{step === "scan-code" && setup && (
				<form
					onSubmit={enable}
					noValidate
					className="animate-fade-in space-y-4"
				>
					<div className="flex flex-col gap-4 sm:flex-row sm:items-start">
						<img
							src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(setup.qrSvg)}`}
							alt="QR code for your authenticator app"
							className="h-40 w-40 rounded-lg bg-white p-2"
						/>
						<div className="min-w-0 space-y-2 text-xs text-muted-foreground">
							<p>
								Scan the QR code with an authenticator app, or enter this key
								manually:
							</p>
							<code className="block break-all rounded bg-muted px-2 py-1 font-mono text-sm text-foreground">
								{setup.secret}
							</code>
						</div>
					</div>
					<div className="space-y-2">
						<Label
							htmlFor="twofa-enable-code"
							className="text-xs leading-6 text-muted-foreground"
						>
							Code from the app
						</Label>
						<CodeInput id="twofa-enable-code" value={code} onChange={setCode} />
					</div>
					{errorLine}
					<div className="flex gap-2">
						<Button
							type="submit"
							size="sm"
							disabled={busy}
							className="bg-medical text-xs text-medical-foreground"
						>
							Turn on
						</Button>
						<Button
							type="button"
							size="sm"
							variant="ghost"
							className="text-xs"
							onClick={reset}
						>
							Cancel
						</Button>
					</div>
				</form>
			)}

			{step === "disable" && (
				<form
					onSubmit={disable}
					noValidate
					className="animate-fade-in space-y-3"
				>
					<div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
						<div className="space-y-2">
							<Label
								htmlFor="twofa-disable-password"
								className="text-xs leading-6 text-muted-foreground"
							>
								Password
							</Label>
							<Input
								id="twofa-disable-password"
								type="password"
								autoComplete="current-password"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								className="h-9 text-sm"
							/>
						</div>
						<div className="space-y-2">
							<Label
								htmlFor="twofa-disable-code"
								className="text-xs leading-6 text-muted-foreground"
							>
								Code from the app
							</Label>
							<CodeInput
								id="twofa-disable-code"
								value={code}
								onChange={setCode}
							/>
						</div>
					</div>
					{errorLine}
					<div className="flex gap-2">
						<Button
							type="submit"
							size="sm"
							variant="destructive"
							disabled={busy}
							className="text-xs"
						>
							Turn off
						</Button>
						<Button
							type="button"
							size="sm"
							variant="ghost"
							className="text-xs"
							onClick={reset}
						>
							Cancel
						</Button>
					</div>
				</form>
			)}
		</div>
	);
};
