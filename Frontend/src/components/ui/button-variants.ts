import { cva } from "class-variance-authority";

export const buttonVariants = cva(
	"inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-[color,background-color,border-color,box-shadow] focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-45 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
	{
		variants: {
			variant: {
				default:
					"bg-primary text-primary-foreground hover:bg-primary/88 active:bg-primary/80",
				destructive:
					"bg-destructive text-destructive-foreground hover:bg-destructive/90",
				outline:
					"border border-input bg-card text-foreground hover:border-foreground/25 hover:bg-accent active:bg-accent/70",
				secondary:
					"bg-secondary text-secondary-foreground hover:bg-secondary/80",
				ghost:
					"text-muted-foreground hover:bg-accent hover:text-accent-foreground",
				link: "text-medical underline-offset-4 hover:underline",
			},
			size: {
				default: "h-9 px-3.5",
				sm: "h-8 gap-1.5 px-2.5 text-[13px] [&_svg]:size-3.5",
				lg: "h-10 px-5",
				icon: "h-9 w-9",
			},
		},
		defaultVariants: {
			variant: "default",
			size: "default",
		},
	},
);
