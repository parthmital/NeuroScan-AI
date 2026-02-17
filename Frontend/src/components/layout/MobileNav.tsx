import { NavLink } from "react-router-dom";
import * as Icons from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
	Sheet,
	SheetContent,
	SheetTrigger,
	SheetTitle,
} from "@/components/ui/sheet";
import { useConfig } from "@/hooks/use-config";
import { useAuth } from "../auth/AuthContext";

export const MobileNav = () => {
	const [open, setOpen] = useState(false);
	const { data: config, isLoading } = useConfig();
	const { user } = useAuth();

	if (isLoading || !config || !user) {
		return null; // Side nav is hidden while loading
	}

	const navItems = config.navItems.map((item) => ({
		...item,
		icon:
			(Icons[item.icon as keyof typeof Icons] as Icons.LucideIcon) ||
			Icons.HelpCircle,
	}));

	return (
		<Sheet open={open} onOpenChange={setOpen}>
			<SheetTrigger asChild>
				<button className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors lg:hidden">
					<Icons.Menu className="w-5 h-5" />
				</button>
			</SheetTrigger>
			<SheetContent
				side="left"
				className="w-[280px] p-0 sidebar-gradient border-sidebar-border"
			>
				<SheetTitle className="sr-only">Navigation</SheetTitle>

				{/* Brand */}
				<div className="flex items-center gap-3 px-5 h-16 border-b border-sidebar-border">
					<div className="flex items-center justify-center w-9 h-9 rounded-lg medical-gradient shrink-0">
						<Icons.Brain className="w-5 h-5 text-medical-foreground" />
					</div>
					<div>
						<h1 className="text-sm font-bold text-sidebar-primary-foreground tracking-tight">
							{config.appName}
						</h1>
						<p className="text-[10px] text-sidebar-foreground/60 tracking-wider uppercase">
							{config.appSubtitle}
						</p>
					</div>
				</div>

				{/* Navigation */}
				<nav className="py-4 px-3 space-y-1">
					{navItems.map((item) => (
						<NavLink
							key={item.to}
							to={item.to}
							end={item.to === "/"}
							onClick={() => setOpen(false)}
							className={({ isActive }) =>
								cn(
									"flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group",
									isActive
										? "bg-sidebar-accent text-sidebar-primary shadow-sm"
										: "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
								)
							}
						>
							{({ isActive }) => {
								const Icon = item.icon;
								return (
									<>
										<Icon
											className={cn(
												"w-[18px] h-[18px] shrink-0 transition-colors",
												isActive
													? "text-sidebar-primary"
													: "text-sidebar-foreground/50 group-hover:text-sidebar-foreground/80",
											)}
										/>
										<span>{item.label}</span>
										{isActive && (
											<div className="ml-auto w-1.5 h-1.5 rounded-full bg-sidebar-primary" />
										)}
									</>
								);
							}}
						</NavLink>
					))}
				</nav>

				{/* User section */}
				<div className="absolute bottom-0 left-0 right-0 border-t border-sidebar-border p-3">
					<div className="flex items-center gap-3 px-3 py-2">
						<div className="w-8 h-8 rounded-full bg-sidebar-accent flex items-center justify-center shrink-0">
							<span className="text-xs font-semibold text-sidebar-primary">
								{user.fullName
									.split(" ")
									.map((n) => n[0])
									.join("")}
							</span>
						</div>
						<div className="flex-1 min-w-0">
							<p className="text-xs font-medium text-sidebar-foreground truncate">
								{user.fullName}
							</p>
							<p className="text-[10px] text-sidebar-foreground/50 truncate">
								{user.title || "Physician"}
							</p>
						</div>
					</div>
				</div>
			</SheetContent>
		</Sheet>
	);
};
