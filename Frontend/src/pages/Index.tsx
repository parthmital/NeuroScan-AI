import { StatsCards } from "@/components/dashboard/StatsCards";
import { RecentScans } from "@/components/dashboard/RecentScans";
import { VolumeChart } from "@/components/dashboard/VolumeChart";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { ArrowUpFromLine } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { APP_CONFIG as config } from "@/lib/config";

import { useAuth } from "@/components/auth/useAuth";

const Index = () => {
	const navigate = useNavigate();
	const { user } = useAuth();

	if (!user) {
		return (
			<div className="animate-pulse space-y-8">
				<div className="h-14 w-1/3 rounded-lg bg-muted" />
				<div className="h-[92px] rounded-lg bg-muted" />
			</div>
		);
	}

	const welcomeMessage = config.dashboard.welcomeMessage.replace(
		"{userName}",
		user.fullName.split(" ")[0],
	);

	return (
		<div className="space-y-8">
			<PageHeader
				title="Dashboard"
				description={welcomeMessage}
				actions={
					<Button onClick={() => navigate("/upload")}>
						<ArrowUpFromLine />
						{config.dashboard.newScanButtonLabel}
					</Button>
				}
			/>

			<StatsCards />

			<div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
				<div className="min-w-0 xl:col-span-2">
					<RecentScans />
				</div>
				<div className="min-w-0">
					<VolumeChart />
				</div>
			</div>
		</div>
	);
};

export default Index;
