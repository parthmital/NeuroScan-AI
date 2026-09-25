import { Palette, Shield, User as UserIcon } from "lucide-react";
import { useAuth } from "@/components/auth/useAuth";
import { AppearanceSettings } from "@/components/settings/AppearanceSettings";
import { PasswordForm } from "@/components/settings/PasswordForm";
import { ProfileForm } from "@/components/settings/ProfileForm";
import { TwoFactorSettings } from "@/components/settings/TwoFactorSettings";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Settings = () => {
	const { user } = useAuth();

	if (!user) {
		return (
			<div className="max-w-3xl space-y-6">
				<div className="h-20 w-1/3 rounded-xl bg-muted" />
				<div className="h-64 rounded-xl bg-muted" />
			</div>
		);
	}

	return (
		<div className="w-full space-y-6">
			<div>
				<h1 className="text-2xl font-bold tracking-tight text-foreground">
					Settings
				</h1>
				<p className="mt-1 text-sm text-muted-foreground">
					Manage your account, preferences, and application settings.
				</p>
			</div>

			<Tabs defaultValue="profile" className="space-y-6">
				<TabsList className="gap-1 bg-muted/50 p-1">
					<TabsTrigger value="profile" className="gap-1.5 text-xs">
						<UserIcon className="h-3.5 w-3.5" />
						Profile
					</TabsTrigger>
					<TabsTrigger value="appearance" className="gap-1.5 text-xs">
						<Palette className="h-3.5 w-3.5" />
						Appearance
					</TabsTrigger>
					<TabsTrigger value="security" className="gap-1.5 text-xs">
						<Shield className="h-3.5 w-3.5" />
						Security
					</TabsTrigger>
				</TabsList>

				<TabsContent value="profile">
					<ProfileForm user={user} />
				</TabsContent>

				<TabsContent value="appearance">
					<AppearanceSettings />
				</TabsContent>

				<TabsContent value="security">
					<div className="space-y-8 rounded-xl border border-border bg-card p-6">
						<PasswordForm />
						<div className="border-t border-border pt-6">
							<TwoFactorSettings user={user} />
						</div>
					</div>
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default Settings;
