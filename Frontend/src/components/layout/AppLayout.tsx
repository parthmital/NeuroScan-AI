import { useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import { AppSidebar } from "./AppSidebar";
import { useAuth } from "../auth/useAuth";
import { LogOut, Settings, Search } from "lucide-react";
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
	const [searchQuery, setSearchQuery] = useState("");

	return (
		<div className="flex min-h-screen bg-background">
			{/* Sidebar */}
			<div className="w-[260px] shrink-0 border-r border-border">
				<AppSidebar />
			</div>

			<div className="flex min-w-0 flex-1 flex-col">
				{/* Top bar */}
				<header className="sticky top-0 z-30 flex h-16 items-center border-b border-border bg-background/80 px-8 backdrop-blur-md">
					<div className="mr-8 flex flex-1 items-center gap-8">
						<div className="relative flex-1">
							<Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
							<Input
								placeholder={APP_CONFIG.dashboard.searchPlaceholder}
								className="h-9 border-none bg-muted/50 pl-9 text-sm"
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
					</div>
					<div className="flex items-center gap-3">
						<div className="h-8 w-px bg-border" />
						<div className="flex items-center gap-3 pl-1">
							<div className="flex flex-col items-end">
								<span className="text-sm font-medium leading-none">
									{user?.fullName}
								</span>
								<span className="text-[10px] uppercase tracking-tight text-muted-foreground">
									{user?.title || "Physician"}
								</span>
							</div>
							<DropdownMenu>
								<DropdownMenuTrigger asChild>
									<button
										aria-label="Account menu"
										className="flex h-9 w-9 items-center justify-center rounded-full border border-medical/20 bg-medical/10 font-bold text-medical transition-all hover:bg-medical/20 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
									>
										{initials(user?.fullName ?? "")}
									</button>
								</DropdownMenuTrigger>
								<DropdownMenuContent align="end" className="w-56">
									<DropdownMenuLabel>My Account</DropdownMenuLabel>
									<DropdownMenuSeparator />
									<DropdownMenuItem onClick={() => navigate("/settings")}>
										<Settings className="mr-2 h-4 w-4" />
										<span>Settings</span>
									</DropdownMenuItem>
									<DropdownMenuSeparator />
									<DropdownMenuItem
										onClick={logout}
										className="text-destructive focus:text-destructive"
									>
										<LogOut className="mr-2 h-4 w-4" />
										<span>Log out</span>
									</DropdownMenuItem>
								</DropdownMenuContent>
							</DropdownMenu>
						</div>
					</div>
				</header>

				{/* Main content */}
				<main className="p-8">
					<Outlet />
				</main>
			</div>
		</div>
	);
};
