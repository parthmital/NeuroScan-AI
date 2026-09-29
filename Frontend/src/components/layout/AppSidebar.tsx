import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
	ArrowUpFromLine,
	FileText,
	HelpCircle,
	Layers,
	LayoutGrid,
	LogOut,
	Settings2,
	type LucideIcon,
} from "lucide-react";
import { APP_CONFIG as config } from "@/lib/config";
import { initials } from "@/lib/user-format";

import { useAuth } from "../auth/useAuth";
import { BrandMark } from "./BrandMark";

// Explicit imports keep the bundle to the icons actually used.
const NAV_ICONS: Record<string, LucideIcon> = {
	LayoutGrid,
	ArrowUpFromLine,
	Layers,
	FileText,
	Settings2,
};

/** Navigation column; `onNavigate` lets the mobile drawer close after a choice. */
export const AppSidebar = ({ onNavigate }: { onNavigate?: () => void }) => {
	const { user, logout } = useAuth();

	const navItems = config.navItems.map((item) => ({
		...item,
		icon: NAV_ICONS[item.icon] ?? HelpCircle,
	}));

	return (
		<aside className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
			{/* Brand */}
			<div className="flex h-14 shrink-0 items-center gap-2.5 px-4">
				<BrandMark />
				<div className="min-w-0 leading-none">
					<p className="text-[14px] font-semibold tracking-[-0.01em] text-foreground">
						{config.appName}
					</p>
					<p className="mt-1 font-mono text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
						{config.appSubtitle}
					</p>
				</div>
			</div>

			{/* Navigation */}
			<nav aria-label="Main" className="flex-1 overflow-y-auto px-2.5 py-3">
				<ul className="space-y-0.5">
					{navItems.map((item) => {
						const Icon = item.icon;
						return (
							<li key={item.to}>
								<NavLink
									to={item.to}
									end={item.to === "/"}
									onClick={onNavigate}
									className={({ isActive }) =>
										cn(
											"flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] font-medium transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-sidebar-ring",
											isActive
												? "bg-sidebar-accent text-sidebar-accent-foreground shadow-[0_1px_2px_hsl(220_20%_10%/0.06),0_0_0_1px_hsl(var(--sidebar-border))] [&_svg]:text-sidebar-primary"
												: "hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground [&_svg]:text-muted-foreground",
										)
									}
								>
									<Icon className="h-4 w-4 shrink-0" />
									<span>{item.label}</span>
								</NavLink>
							</li>
						);
					})}
				</ul>
			</nav>

			{/* Signed-in user */}
			{user && (
				<div className="border-t border-sidebar-border p-2.5">
					<div className="flex items-center gap-2.5 rounded-md px-1.5 py-1.5">
						<span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-sidebar-border bg-sidebar-accent font-mono text-[11px] font-medium text-foreground">
							{initials(user.fullName)}
						</span>
						<div className="min-w-0 flex-1 leading-tight">
							<p className="truncate text-[13px] font-medium text-foreground">
								{user.fullName}
							</p>
							<p className="truncate text-[11px] text-muted-foreground">
								{user.title || user.username}
							</p>
						</div>
						<button
							onClick={logout}
							aria-label="Log out"
							className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-sidebar-ring"
						>
							<LogOut className="h-4 w-4" />
						</button>
					</div>
				</div>
			)}
		</aside>
	);
};
