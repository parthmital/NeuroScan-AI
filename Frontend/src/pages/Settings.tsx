import { useState, useEffect } from "react";
import { User, Palette, Shield, Monitor, Moon, Sun, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useConfig } from "@/hooks/use-config";
import { useAuth } from "@/components/auth/AuthContext";
import { toast } from "@/components/ui/use-toast";

const Settings = () => {
	const { data: config, isLoading: configLoading } = useConfig();
	const { user, token, login } = useAuth();

	const [formData, setFormData] = useState({
		firstName: "",
		lastName: "",
		email: "",
		title: "",
		department: "",
		institution: "",
	});

	const [darkMode, setDarkMode] = useState(() => {
		const theme = localStorage.getItem("theme");
		return (
			theme === "dark" ||
			(!theme && window.matchMedia("(prefers-color-scheme: dark)").matches)
		);
	});

	const [overlayColors, setOverlayColors] = useState(() => {
		const saved = localStorage.getItem("overlayColors");
		return saved
			? JSON.parse(saved)
			: {
					wt: "#eab308",
					tc: "#ef4444",
					et: "#3b82f6",
				};
	});

	useEffect(() => {
		if (user) {
			setFormData({
				firstName: user.fullName.split(" ")[0] || "",
				lastName: user.fullName.split(" ").slice(1).join(" ") || "",
				email: user.email || "",
				title: user.title || "",
				department: user.department || "",
				institution: user.institution || "",
			});
		}
	}, [user]);

	useEffect(() => {
		document.documentElement.classList.toggle("dark", darkMode);
		localStorage.setItem("theme", darkMode ? "dark" : "light");
	}, [darkMode]);

	useEffect(() => {
		localStorage.setItem("overlayColors", JSON.stringify(overlayColors));
	}, [overlayColors]);

	if (configLoading || !config || !user) {
		return (
			<div className="space-y-6 max-w-3xl">
				<div className="h-20 bg-muted rounded-xl w-1/3" />
				<div className="h-64 bg-muted rounded-xl" />
			</div>
		);
	}

	const handleSaveChanges = async () => {
		try {
			const fullName = `${formData.firstName} ${formData.lastName}`.trim();
			const response = await fetch("http://localhost:8000/api/auth/me", {
				method: "PUT",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${token}`,
				},
				body: JSON.stringify({
					fullName,
					email: formData.email,
					title: formData.title,
					department: formData.department,
					institution: formData.institution,
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || "Failed to update profile");
			}

			const updatedUser = await response.json();
			login(token!, updatedUser);

			toast({
				title: "Settings Saved",
				description: "Your profile has been updated successfully.",
			});
		} catch (error: unknown) {
			const message =
				error instanceof Error
					? error.message
					: "There was an error saving your changes.";
			toast({
				title: "Update Failed",
				description: message,
				variant: "destructive",
			});
		}
	};

	return (
		<div className="space-y-6 w-full">
			<div>
				<h1 className="text-2xl font-bold text-foreground tracking-tight">
					Settings
				</h1>
				<p className="text-sm text-muted-foreground mt-1">
					Manage your account, preferences, and application settings.
				</p>
			</div>

			<Tabs defaultValue="profile" className="space-y-6">
				<TabsList className="bg-muted/50 gap-1 p-1">
					<TabsTrigger value="profile" className="gap-1.5 text-xs">
						<User className="w-3.5 h-3.5" />
						Profile
					</TabsTrigger>
					<TabsTrigger value="appearance" className="gap-1.5 text-xs">
						<Palette className="w-3.5 h-3.5" />
						Appearance
					</TabsTrigger>
					<TabsTrigger value="security" className="gap-1.5 text-xs">
						<Shield className="w-3.5 h-3.5" />
						Security
					</TabsTrigger>
				</TabsList>

				{/* Profile */}
				<TabsContent value="profile">
					<div className="bg-card rounded-xl border border-border p-6 space-y-6">
						<div>
							<h2 className="text-sm font-semibold text-foreground mb-4">
								Profile Information
							</h2>
							<div className="grid grid-cols-2 gap-4">
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										First Name
									</label>
									<Input
										value={formData.firstName}
										onChange={(e) =>
											setFormData((prev) => ({
												...prev,
												firstName: e.target.value,
											}))
										}
										className="h-9 text-sm"
									/>
								</div>
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										Last Name
									</label>
									<Input
										value={formData.lastName}
										onChange={(e) =>
											setFormData((prev) => ({
												...prev,
												lastName: e.target.value,
											}))
										}
										className="h-9 text-sm"
									/>
								</div>
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										Email
									</label>
									<Input
										value={formData.email}
										onChange={(e) =>
											setFormData((prev) => ({
												...prev,
												email: e.target.value,
											}))
										}
										className="h-9 text-sm"
									/>
								</div>
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										Professional Title
									</label>
									<Input
										value={formData.title}
										onChange={(e) =>
											setFormData((prev) => ({
												...prev,
												title: e.target.value,
											}))
										}
										className="h-9 text-sm"
										placeholder="e.g. Chief Neuroradiologist"
									/>
								</div>
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										Department
									</label>
									<Input
										value={formData.department}
										onChange={(e) =>
											setFormData((prev) => ({
												...prev,
												department: e.target.value,
											}))
										}
										className="h-9 text-sm"
									/>
								</div>
								<div className="space-y-2 col-span-2">
									<label className="text-xs font-medium text-muted-foreground">
										Institution
									</label>
									<Input
										value={formData.institution}
										onChange={(e) =>
											setFormData((prev) => ({
												...prev,
												institution: e.target.value,
											}))
										}
										className="h-9 text-sm"
									/>
								</div>
							</div>
						</div>

						<div className="flex justify-end pt-2 border-t border-border">
							<Button
								onClick={handleSaveChanges}
								className="bg-medical text-medical-foreground hover:bg-medical/90 text-xs gap-1.5"
							>
								<Save className="w-3.5 h-3.5" />
								Save Changes
							</Button>
						</div>
					</div>
				</TabsContent>

				{/* Appearance */}
				<TabsContent value="appearance">
					<div className="bg-card rounded-xl border border-border p-6 space-y-6">
						<div>
							<h2 className="text-sm font-semibold text-foreground mb-1">
								Theme
							</h2>
							<p className="text-xs text-muted-foreground mb-4">
								Customise the application appearance.
							</p>
							<div className="grid grid-cols-2 gap-3">
								<button
									onClick={() => setDarkMode(false)}
									className={cn(
										"flex items-center gap-3 p-4 rounded-lg border-2",
										!darkMode
											? "border-medical bg-medical/5"
											: "border-border hover:border-muted-foreground/30",
									)}
								>
									<Sun className="w-5 h-5 text-warning" />
									<div className="text-left">
										<p className="text-sm font-medium text-foreground">Light</p>
										<p className="text-xs text-muted-foreground">
											Clean and bright interface
										</p>
									</div>
								</button>
								<button
									onClick={() => setDarkMode(true)}
									className={cn(
										"flex items-center gap-3 p-4 rounded-lg border-2",
										darkMode
											? "border-medical bg-medical/5"
											: "border-border hover:border-muted-foreground/30",
									)}
								>
									<Moon className="w-5 h-5 text-info" />
									<div className="text-left">
										<p className="text-sm font-medium text-foreground">Dark</p>
										<p className="text-xs text-muted-foreground">
											Optimised for low-light reading
										</p>
									</div>
								</button>
							</div>
						</div>

						<div>
							<h2 className="text-sm font-semibold text-foreground mb-1">
								Overlay Defaults
							</h2>
							<p className="text-xs text-muted-foreground mb-4">
								Set default segmentation overlay colours.
							</p>
							<div className="space-y-3">
								{[
									{
										id: "wt",
										label: "Whole Tumour (WT)",
										color: overlayColors.wt,
									},
									{
										id: "tc",
										label: "Tumour Core (TC)",
										color: overlayColors.tc,
									},
									{
										id: "et",
										label: "Enhancing Tumour (ET)",
										color: overlayColors.et,
									},
								].map((item) => (
									<div
										key={item.id}
										className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border border-border"
									>
										<input
											type="color"
											value={item.color}
											onChange={(e) =>
												setOverlayColors((prev) => ({
													...prev,
													[item.id]: e.target.value,
												}))
											}
											className="w-8 h-8 rounded shrink-0 cursor-pointer bg-transparent border-none"
										/>
										<span className="text-sm text-foreground flex-1">
											{item.label}
										</span>
										<Input
											value={item.color}
											className="h-7 w-24 text-xs font-mono uppercase"
											readOnly
										/>
									</div>
								))}
							</div>
						</div>
					</div>
				</TabsContent>

				{/* Security */}
				<TabsContent value="security">
					<div className="bg-card rounded-xl border border-border p-6 space-y-6">
						<div>
							<h2 className="text-sm font-semibold text-foreground mb-1">
								Change Password
							</h2>
							<p className="text-xs text-muted-foreground mb-4">
								Update your account password.
							</p>
							<div className="space-y-3">
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										Current Password
									</label>
									<Input type="password" className="h-9 text-sm" />
								</div>
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										New Password
									</label>
									<Input type="password" className="h-9 text-sm" />
								</div>
								<div className="space-y-2">
									<label className="text-xs font-medium text-muted-foreground">
										Confirm New Password
									</label>
									<Input type="password" className="h-9 text-sm" />
								</div>
							</div>
						</div>

						<div>
							<h2 className="text-sm font-semibold text-foreground mb-1">
								Two-Factor Authentication
							</h2>
							<p className="text-xs text-muted-foreground mb-4">
								Add an extra layer of security to your account.
							</p>
							<div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border border-border">
								<Shield className="w-5 h-5 text-medical shrink-0" />
								<div className="flex-1 min-w-0">
									<p className="text-sm font-medium text-foreground">
										Enable 2FA
									</p>
									<p className="text-xs text-muted-foreground">
										Secure your account with TOTP authentication
									</p>
								</div>
								<Switch className="shrink-0" />
							</div>
						</div>

						<div className="flex justify-end pt-2 border-t border-border">
							<Button className="bg-medical text-medical-foreground hover:bg-medical/90 text-xs gap-1.5">
								Update Security
							</Button>
						</div>
					</div>
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default Settings;
