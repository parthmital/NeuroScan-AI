import type { ReactNode } from "react";
import { APP_CONFIG as config } from "@/lib/config";
import { cn } from "@/lib/utils";
import { BrandMark } from "./BrandMark";

interface AuthShellProps {
	title: string;
	description: string;
	children: ReactNode;
	footer?: ReactNode;
	className?: string;
}

/** Signed-out page frame: brand bar, a centred form column, and a footer note. */
export const AuthShell = ({
	title,
	description,
	children,
	footer,
	className,
}: AuthShellProps) => (
	<div className="flex min-h-screen flex-col bg-background">
		<header className="flex h-16 items-center gap-2.5 px-6 sm:px-10">
			<BrandMark />
			<span className="text-[14px] font-semibold tracking-[-0.01em] text-foreground">
				{config.appName}
			</span>
		</header>

		<main className="flex flex-1 items-center justify-center px-4 py-10">
			<div className={cn("w-full max-w-[380px] animate-fade-in", className)}>
				<h1 className="text-2xl font-semibold tracking-[-0.02em] text-foreground">
					{title}
				</h1>
				<p className="mt-1.5 text-sm text-muted-foreground">{description}</p>
				<div className="mt-8">{children}</div>
			</div>
		</main>

		{footer && (
			<footer className="px-6 pb-8 text-center text-xs text-muted-foreground sm:px-10">
				{footer}
			</footer>
		)}
	</div>
);
