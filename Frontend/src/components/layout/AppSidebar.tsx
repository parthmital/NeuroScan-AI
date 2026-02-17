import { NavLink } from "react-router-dom";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { useConfig } from "@/hooks/use-config";
import * as Icons from "lucide-react";
import { AppConfig } from "@/lib/types";

import { useAuth } from "../auth/AuthContext";

export const AppSidebar = () => {
	const [collapsed, setCollapsed] = useState(false);
	const { data: config, isLoading: configLoading } = useConfig();
	const { user, logout } = useAuth();

	if (configLoading || !config || !user) {
		return (
			<aside
				className={cn(
					"fixed left-0 top-0 z-40 h-screen sidebar-gradient border-r border-sidebar-border w-[260px]",
				)}
			/>
		);
	}

	const navItems = config.navItems.map((item) => ({
		...item,
		icon:
			(Icons[item.icon as keyof typeof Icons] as Icons.LucideIcon) ||
			Icons.HelpCircle,
	}));

	return (
		<aside
			className={cn(
				"fixed left-0 top-0 z-40 h-screen sidebar-gradient border-r border-sidebar-border flex flex-col transition-all duration-300",
				collapsed ? "w-[72px]" : "w-[260px]",
			)}
		>
			{/* Brand */}
			<div className="flex items-center gap-3 px-5 h-16 border-b border-sidebar-border">
				<div className="flex items-center justify-center w-9 h-9 rounded-lg medical-gradient shrink-0">
					<Icons.Brain className="w-5 h-5 text-medical-foreground" />
				</div>
				{!collapsed && (
					<div className="animate-slide-in-left">
						<h1 className="text-sm font-bold text-sidebar-primary-foreground tracking-tight">
							{config.appName}
						</h1>
						<p className="text-[10px] text-sidebar-foreground/60 tracking-wider uppercase">
							{config.appSubtitle}
						</p>
					</div>
				)}
			</div>

			{/* Navigation */}
			<nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
				{navItems.map((item) => (
					<NavLink
						key={item.to}
						to={item.to}
						end={item.to === "/"}
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
									{!collapsed && <span>{item.label}</span>}
									{isActive && !collapsed && (
										<div className="ml-auto w-1.5 h-1.5 rounded-full bg-sidebar-primary" />
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
					<div className="w-8 h-8 rounded-full bg-sidebar-accent flex items-center justify-center shrink-0">
						<span className="text-xs font-semibold text-sidebar-primary">
							{user.fullName
								.split(" ")
								.map((n) => n[0])
								.join("")}
						</span>
					</div>
					{!collapsed && (
						<div className="flex-1 min-w-0 animate-slide-in-left">
							<p className="text-xs font-medium text-sidebar-foreground truncate">
								{user.fullName}
							</p>
							<p className="text-[10px] text-sidebar-foreground/50 truncate">
								{user.username}
							</p>
						</div>
					)}
					{!collapsed && (
						<button
							onClick={logout}
							className="p-1 rounded text-sidebar-foreground/40 hover:text-sidebar-foreground/70 transition-colors"
						>
							<Icons.LogOut className="w-4 h-4" />
						</button>
					)}
				</div>
			</div>
		</aside>
	);
};
