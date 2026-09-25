import { DropZone } from "@/components/upload/DropZone";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { APP_CONFIG as config } from "@/lib/config";

const Upload = () => {
	const navigate = useNavigate();

	return (
		<div className="space-y-6">
			<div className="flex items-center gap-4">
				<Button
					variant="ghost"
					size="sm"
					onClick={() => navigate("/")}
					className="text-muted-foreground"
				>
					<ArrowLeft className="mr-1 h-4 w-4" />
					{config.upload.backButton}
				</Button>
			</div>

			<div>
				<h1 className="text-2xl font-bold tracking-tight text-foreground">
					{config.upload.title}
				</h1>
				<p className="mt-1 text-sm text-muted-foreground">
					{config.upload.subtitle}
				</p>
			</div>

			<DropZone />
		</div>
	);
};

export default Upload;
