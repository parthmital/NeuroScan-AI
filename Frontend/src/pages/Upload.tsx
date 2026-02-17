import { DropZone } from "@/components/upload/DropZone";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useConfig } from "@/hooks/use-config";
import { AppConfig } from "@/lib/types";

const Upload = () => {
	const navigate = useNavigate();
	const { data: config, isLoading } = useConfig();

	if (isLoading || !config) {
		return (
			<div className="space-y-6 animate-pulse">
				<div className="h-10 bg-muted rounded-xl w-24" />
				<div className="h-20 bg-muted rounded-xl w-3/4" />
			</div>
		);
	}

	return (
		<div className="space-y-6">
			<div className="flex items-center gap-4">
				<Button
					variant="ghost"
					size="sm"
					onClick={() => navigate("/")}
					className="text-muted-foreground"
				>
					<ArrowLeft className="w-4 h-4 mr-1" />
					{(config as AppConfig).upload.backButton}
				</Button>
			</div>

			<div>
				<h1 className="text-2xl font-bold text-foreground tracking-tight">
					{(config as AppConfig).upload.title}
				</h1>
				<p className="text-sm text-muted-foreground mt-1">
					{(config as AppConfig).upload.subtitle}
				</p>
			</div>

			<DropZone />
		</div>
	);
};

export default Upload;
