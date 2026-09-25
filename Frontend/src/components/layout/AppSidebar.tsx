import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
	Brain,
	FileText,
	HelpCircle,
	Images,
	LayoutDashboard,
	LogOut,
	Settings,
	Upload,
	type LucideIcon,
} from "lucide-react";
import { APP_CONFIG as config } from "@/lib/config";
import { initials } from "@/lib/user-format";

import { useAuth } from "../auth/useAuth";

// Explicit imports keep the bundle to the icons actually used.
const NAV_ICONS: Record<string, LucideIcon> = {
	LayoutDashboard,
	Upload,
	Images,
	FileText,
	Settings,
};

export const AppSidebar = () => {
	const { user, logout } = useAuth();

	if (!user) {
		return (
			<aside
				className={cn(
					"sidebar-gradient fixed left-0 top-0 z-40 h-screen w-[260px] border-r border-sidebar-border",
				)}
			/>
		);
	}

	const navItems = config.navItems.map((item) => ({
		...item,
		icon: NAV_ICONS[item.icon] ?? HelpCircle,
	}));

	return (
		<aside
			className={cn(
				"sidebar-gradient fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-sidebar-border transition-all duration-300",
				"w-[260px]",
			)}
		>
			{/* Brand */}
			<div className="flex h-16 items-center gap-3 border-b border-sidebar-border px-5">
				<div className="medical-gradient flex h-9 w-9 shrink-0 items-center justify-center rounded-lg">
					<Brain className="h-5 w-5 text-medical-foreground" />
				</div>
				<div className="animate-slide-in-left">
					<h1 className="text-sm font-bold tracking-tight text-sidebar-primary-foreground">
						{config.appName}
					</h1>
					<p className="text-[10px] uppercase tracking-wider text-sidebar-foreground/60">
						{config.appSubtitle}
					</p>
				</div>
			</div>

			{/* Navigation */}
			<nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
				{navItems.map((item) => (
					<NavLink
						key={item.to}
						to={item.to}
						end={item.to === "/"}
						className={({ isActive }) =>
							cn(
								"group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
								isActive
									? "bg-sidebar-accent text-sidebar-primary shadow-sm"
									: "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
							)
						}
					>
						{({ isActive }) => {
							const Icon = item.icon;
							return (
								<>
									<Icon
										className={cn(
											"h-[18px] w-[18px] shrink-0 transition-colors",
											isActive
												? "text-sidebar-primary"
												: "text-sidebar-foreground/50 group-hover:text-sidebar-foreground/80",
										)}
									/>
									<span>{item.label}</span>
									{isActive && (
										<div className="ml-auto h-1.5 w-1.5 rounded-full bg-sidebar-primary" />
									)}
								</>
							);
						}}
					</NavLink>
				))}
			</nav>

			{/* User section */}
			<div className="border-t border-sidebar-border p-3">
				<div className="flex items-center gap-3 px-3 py-2">
					<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-sidebar-accent">
						<span className="text-xs font-semibold text-sidebar-primary">
							{initials(user.fullName)}
						</span>
					</div>
					<div className="animate-slide-in-left min-w-0 flex-1">
						<p className="truncate text-xs font-medium text-sidebar-foreground">
							{user.fullName}
						</p>
						<p className="truncate text-[10px] text-sidebar-foreground/50">
							{user.username}
						</p>
					</div>
					<button
						onClick={logout}
						aria-label="Log out"
						className="rounded p-1 text-sidebar-foreground/40 transition-colors hover:text-sidebar-foreground/70"
					>
						<LogOut className="h-4 w-4" />
					</button>
				</div>
			</div>
		</aside>
	);
};
