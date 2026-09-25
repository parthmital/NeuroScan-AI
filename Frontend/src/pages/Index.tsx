import { StatsCards } from "@/components/dashboard/StatsCards";
import { RecentScans } from "@/components/dashboard/RecentScans";
import { VolumeChart } from "@/components/dashboard/VolumeChart";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { APP_CONFIG as config } from "@/lib/config";

import { useAuth } from "@/components/auth/useAuth";

const Index = () => {
	const navigate = useNavigate();
	const { user } = useAuth();

	if (!user) {
		return (
			<div className="space-y-8">
				<div className="h-20 w-1/3 rounded-xl bg-muted" />
				<div className="grid grid-cols-4 gap-4">
					{[1, 2, 3, 4].map((i) => (
						<div key={i} className="h-32 rounded-xl bg-muted" />
					))}
				</div>
			</div>
		);
	}

	const welcomeMessage = config.dashboard.welcomeMessage.replace(
		"{userName}",
		user.fullName.split(" ")[0],
	);

	return (
		<div className="space-y-8">
			{/* Header */}
			<div className="flex items-center justify-between gap-4">
				<div>
					<h1 className="text-2xl font-bold tracking-tight text-foreground">
						Dashboard
					</h1>
					<p className="mt-1 text-sm text-muted-foreground">{welcomeMessage}</p>
				</div>
				<Button
					onClick={() => navigate("/upload")}
					className="gap-2 bg-medical text-medical-foreground hover:bg-medical/90"
				>
					<Upload className="h-4 w-4" />
					{config.dashboard.newScanButtonLabel}
				</Button>
			</div>

			{/* Stats */}
			<StatsCards />

			{/* Main grid */}
			<div className="grid grid-cols-3 gap-6">
				<div className="col-span-2">
					<RecentScans />
				</div>
				<div className="space-y-6">
					<VolumeChart />
				</div>
			</div>
		</div>
	);
};

export default Index;
