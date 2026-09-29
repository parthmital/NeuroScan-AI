import { cn } from "@/lib/utils";

interface SegmentedProps<T extends string> {
	options: readonly { value: T; label: string }[];
	value: T;
	onChange: (value: T) => void;
	"aria-label": string;
	className?: string;
}

/** Single-choice toggle group for switching views or modalities. */
export const Segmented = <T extends string>({
	options,
	value,
	onChange,
	className,
	...props
}: SegmentedProps<T>) => (
	<div
		role="group"
		aria-label={props["aria-label"]}
		className={cn(
			"inline-flex h-8 items-center rounded-md border border-border bg-muted/60 p-0.5",
			className,
		)}
	>
		{options.map((option) => (
			<button
				key={option.value}
				type="button"
				onClick={() => onChange(option.value)}
				aria-pressed={value === option.value}
				className={cn(
					"h-full rounded-[5px] px-3 font-mono text-[11px] font-medium uppercase tracking-[0.04em] transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring",
					value === option.value
						? "bg-card text-foreground shadow-[0_1px_2px_hsl(220_20%_10%/0.12),0_0_0_1px_hsl(var(--border))]"
						: "text-muted-foreground hover:text-foreground",
				)}
			>
				{option.label}
			</button>
		))}
	</div>
);
