import { cn } from "@/lib/utils";

/** Axial slice outline and midline: the NeuroScan mark. */
export const BrandMark = ({ className }: { className?: string }) => (
	<span
		aria-hidden="true"
		className={cn(
			"flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-foreground text-background",
			className,
		)}
	>
		<svg viewBox="0 0 24 24" fill="none" className="h-[22px] w-[22px]">
			<ellipse
				cx="12"
				cy="12"
				rx="7.5"
				ry="9.25"
				stroke="currentColor"
				strokeWidth="1.6"
			/>
			<path
				d="M12 3.75v16.5"
				stroke="currentColor"
				strokeWidth="1.2"
				strokeDasharray="1.6 1.8"
			/>
		</svg>
	</span>
);
