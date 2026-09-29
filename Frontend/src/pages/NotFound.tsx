import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { AuthShell } from "@/components/layout/AuthShell";

const NotFound = () => (
	<AuthShell
		title="Page not found"
		description="Error 404. The address may be mistyped, or the page has moved."
	>
		<Link to="/" className={buttonVariants({ variant: "outline" })}>
			<ArrowLeft />
			Return to Dashboard
		</Link>
	</AuthShell>
);

export default NotFound;
