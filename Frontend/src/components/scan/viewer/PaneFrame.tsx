import type { ReactNode } from "react";
import { Maximize2, Minimize2 } from "lucide-react";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface PaneFrameProps {
	title: string;
	colour: string;
	active: boolean;
	maximised: boolean;
	/** Omitted where only one view fits, so there is nothing to switch to. */
	onToggleMaximise?: () => void;
	/** Right side of the header, before the maximise button. */
	info?: ReactNode;
	footer?: ReactNode;
	children: ReactNode;
}

/** The dark frame shared by every viewer pane: title bar, viewport, and footer. */
export const PaneFrame = ({
	title,
	colour,
	active,
	maximised,
	onToggleMaximise,
	info,
	footer,
	children,
}: PaneFrameProps) => {
	const label = maximised
		? `Show all views`
		: `Enlarge ${title.toLowerCase()} view`;
	return (
		<section
			aria-label={`${title} view`}
			className={cn(
				"flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-[#0a0a0a] text-white transition-[border-color,box-shadow] duration-150",
				active
					? "border-white/35 shadow-[0_0_0_1px_rgba(255,255,255,0.12)]"
					: "border-white/10",
			)}
		>
			<header className="flex h-9 shrink-0 items-center gap-2 border-b border-white/[0.07] pl-3 pr-3 has-[button]:pr-1">
				<span
					aria-hidden
					className="h-2 w-2 shrink-0 rounded-full"
					style={{ backgroundColor: colour }}
				/>
				<h3 className="font-mono text-[11px] font-medium uppercase tracking-[0.1em] text-white/85">
					{title}
				</h3>
				<div className="ml-auto flex min-w-0 items-center gap-2 font-mono text-[11px] text-white/60 tabular-nums">
					{info}
				</div>
				{onToggleMaximise && (
					<Tooltip>
						<TooltipTrigger asChild>
							<button
								type="button"
								onClick={onToggleMaximise}
								aria-label={label}
								aria-pressed={maximised}
								className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-white/55 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-white/60"
							>
								{maximised ? (
									<Minimize2 className="h-3.5 w-3.5" />
								) : (
									<Maximize2 className="h-3.5 w-3.5" />
								)}
							</button>
						</TooltipTrigger>
						<TooltipContent>{label}</TooltipContent>
					</Tooltip>
				)}
			</header>
			<div className="relative min-h-0 flex-1">{children}</div>
			{footer && (
				<div className="flex h-10 shrink-0 items-center gap-3 border-t border-white/[0.07] px-3">
					{footer}
				</div>
			)}
		</section>
	);
};
