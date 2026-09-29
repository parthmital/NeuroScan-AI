import { DropZone } from "@/components/upload/DropZone";
import { PageHeader } from "@/components/layout/PageHeader";
import { useNavigate } from "react-router-dom";
import { APP_CONFIG as config } from "@/lib/config";

const Upload = () => {
	const navigate = useNavigate();

	return (
		<div className="max-w-3xl space-y-8">
			<PageHeader
				back={{ label: config.upload.backButton, onClick: () => navigate("/") }}
				title={config.upload.title}
				description={config.upload.subtitle}
			/>

			<DropZone />
		</div>
	);
};

export default Upload;
