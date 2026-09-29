import type { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PageHeaderProps {
	title: ReactNode;
	description?: ReactNode;
	actions?: ReactNode;
	back?: { label: string; onClick: () => void };
}

export const PageHeader = ({
	title,
	description,
	actions,
	back,
}: PageHeaderProps) => (
	<header className="space-y-3">
		{back && (
			<Button
				variant="ghost"
				size="sm"
				onClick={back.onClick}
				className="-ml-2.5"
			>
				<ArrowLeft />
				{back.label}
			</Button>
		)}
		<div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
			<div className="min-w-0">
				<h1 className="truncate text-2xl font-semibold tracking-[-0.02em] text-foreground">
					{title}
				</h1>
				{description && (
					<div className="mt-1 text-sm text-muted-foreground">
						{description}
					</div>
				)}
			</div>
			{actions && (
				<div className="flex shrink-0 flex-wrap items-center gap-2">
					{actions}
				</div>
			)}
		</div>
	</header>
);
