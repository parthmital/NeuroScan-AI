import { StatsCards } from "@/components/dashboard/StatsCards";
import { RecentScans } from "@/components/dashboard/RecentScans";
import { VolumeChart } from "@/components/dashboard/VolumeChart";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useConfig } from "@/hooks/use-config";

import { useAuth } from "@/components/auth/AuthContext";

const Index = () => {
	const navigate = useNavigate();
	const { data: config, isLoading: configLoading } = useConfig();
	const { user } = useAuth();

	if (configLoading || !config || !user) {
		return (
			<div className="space-y-8">
				<div className="h-20 bg-muted rounded-xl w-1/3" />
				<div className="grid grid-cols-4 gap-4">
					{[1, 2, 3, 4].map((i) => (
						<div key={i} className="h-32 bg-muted rounded-xl" />
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
					<h1 className="text-2xl font-bold text-foreground tracking-tight">
						Dashboard
					</h1>
					<p className="text-sm text-muted-foreground mt-1">{welcomeMessage}</p>
				</div>
				<Button
					onClick={() => navigate("/upload")}
					className="bg-medical text-medical-foreground hover:bg-medical/90 gap-2"
				>
					<Upload className="w-4 h-4" />
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
