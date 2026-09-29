import { useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { LogOut, Menu, Search, Settings2 } from "lucide-react";
import { AppSidebar } from "./AppSidebar";
import { useAuth } from "../auth/useAuth";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { APP_CONFIG } from "@/lib/config";
import { initials } from "@/lib/user-format";

export const AppLayout = () => {
	const { user, logout } = useAuth();
	const navigate = useNavigate();
	const { pathname } = useLocation();
	const [searchQuery, setSearchQuery] = useState("");
	const [menuOpen, setMenuOpen] = useState(false);

	return (
		<div className="min-h-screen bg-background">
			{/* Sidebar: fixed column on large screens */}
			<div className="fixed inset-y-0 left-0 z-40 hidden w-60 border-r border-sidebar-border lg:block">
				<AppSidebar />
			</div>

			{/* Sidebar: drawer on smaller screens */}
			<DialogPrimitive.Root open={menuOpen} onOpenChange={setMenuOpen}>
				<DialogPrimitive.Portal>
					<DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-[hsl(220_20%_6%/0.45)] data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 lg:hidden" />
					<DialogPrimitive.Content className="fixed inset-y-0 left-0 z-50 w-64 max-w-[85vw] border-r border-sidebar-border shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left lg:hidden">
						<DialogPrimitive.Title className="sr-only">
							Navigation
						</DialogPrimitive.Title>
						<DialogPrimitive.Description className="sr-only">
							Move between NeuroScan pages.
						</DialogPrimitive.Description>
						<AppSidebar onNavigate={() => setMenuOpen(false)} />
					</DialogPrimitive.Content>
				</DialogPrimitive.Portal>
			</DialogPrimitive.Root>

			<div className="flex min-w-0 flex-col lg:pl-60">
				{/* Top bar */}
				<header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-background/85 px-4 backdrop-blur-md sm:px-6 lg:px-10">
					<button
						onClick={() => setMenuOpen(true)}
						aria-label="Open navigation"
						className="-ml-1.5 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring lg:hidden"
					>
						<Menu className="h-5 w-5" />
					</button>
					<div className="relative w-full max-w-md">
						<Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
						<Input
							type="search"
							aria-label="Search scans"
							placeholder={APP_CONFIG.dashboard.searchPlaceholder}
							className="h-8 bg-card pl-8 text-[13px]"
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							onKeyDown={(e) => {
								if (e.key === "Enter" && searchQuery.trim()) {
									navigate(
										`/scans?search=${encodeURIComponent(searchQuery.trim())}`,
									);
								}
							}}
						/>
					</div>
					<div className="ml-auto">
						<DropdownMenu>
							<DropdownMenuTrigger asChild>
								<button
									aria-label="Account menu"
									className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-card font-mono text-[11px] font-medium text-foreground transition-colors hover:border-foreground/25 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
								>
									{initials(user?.fullName ?? "")}
								</button>
							</DropdownMenuTrigger>
							<DropdownMenuContent align="end" className="w-60">
								<DropdownMenuLabel className="font-normal">
									<p className="truncate text-[13px] font-medium text-foreground">
										{user?.fullName}
									</p>
									<p className="truncate text-xs text-muted-foreground">
										{user?.email || user?.username}
									</p>
								</DropdownMenuLabel>
								<DropdownMenuSeparator />
								<DropdownMenuItem onClick={() => navigate("/settings")}>
									<Settings2 className="mr-2 h-4 w-4" />
									<span>Settings</span>
								</DropdownMenuItem>
								<DropdownMenuItem
									onClick={logout}
									className="text-destructive focus:text-destructive [&_svg]:text-destructive"
								>
									<LogOut className="mr-2 h-4 w-4" />
									<span>Log out</span>
								</DropdownMenuItem>
							</DropdownMenuContent>
						</DropdownMenu>
					</div>
				</header>

				{/* Main content */}
				{/* Keyed by path so each page fades in when navigated to. */}
				<main
					key={pathname}
					className="mx-auto w-full max-w-[1320px] animate-fade-in px-4 py-8 sm:px-6 lg:px-10"
				>
					<Outlet />
				</main>
			</div>
		</div>
	);
};
