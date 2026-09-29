import { KeyRound, Palette, UserRound } from "lucide-react";
import { useAuth } from "@/components/auth/useAuth";
import { PageHeader } from "@/components/layout/PageHeader";
import { AppearanceSettings } from "@/components/settings/AppearanceSettings";
import { PasswordForm } from "@/components/settings/PasswordForm";
import { ProfileForm } from "@/components/settings/ProfileForm";
import { TwoFactorSettings } from "@/components/settings/TwoFactorSettings";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Settings = () => {
	const { user } = useAuth();

	if (!user) {
		return (
			<div className="max-w-3xl animate-pulse space-y-6">
				<div className="h-14 w-1/3 rounded-lg bg-muted" />
				<div className="h-64 rounded-lg bg-muted" />
			</div>
		);
	}

	return (
		<div className="max-w-3xl space-y-6">
			<PageHeader
				title="Settings"
				description="Manage your account, preferences, and application settings."
			/>

			<Tabs defaultValue="profile" className="flex flex-col gap-6">
				<TabsList>
					<TabsTrigger value="profile">
						<UserRound />
						Profile
					</TabsTrigger>
					<TabsTrigger value="appearance">
						<Palette />
						Appearance
					</TabsTrigger>
					<TabsTrigger value="security">
						<KeyRound />
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
					<Card className="divide-y divide-border">
						<div className="p-5">
							<PasswordForm />
						</div>
						<div className="p-5">
							<TwoFactorSettings user={user} />
						</div>
					</Card>
				</TabsContent>
			</Tabs>
		</div>
	);
};

export default Settings;
