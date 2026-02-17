import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import Index from "./pages/Index";
import Upload from "./pages/Upload";
import ScanDetail from "./pages/ScanDetail";
import ScanLibrary from "./pages/ScanLibrary";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";

import { useNavigate, Navigate, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./components/auth/AuthContext";
import Login from "./pages/Login";
import Register from "./pages/Register";
import { useEffect } from "react";

const queryClient = new QueryClient();

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
	const { isAuthenticated, isLoading } = useAuth();
	const location = useLocation();

	if (isLoading) {
		return (
			<div className="h-screen w-screen flex items-center justify-center bg-black">
				<div className="flex flex-col items-center gap-4">
					<Loader2 className="w-10 h-10 text-medical" />
					<p className="text-white/50 text-sm tracking-widest uppercase">
						Initialising Security
					</p>
				</div>
			</div>
		);
	}

	if (!isAuthenticated) {
		return <Navigate to="/login" state={{ from: location }} replace />;
	}

	return <>{children}</>;
};

import { Loader2 } from "lucide-react";

const App = () => {
	useEffect(() => {
		const theme = localStorage.getItem("theme");
		if (
			theme === "dark" ||
			(!theme && window.matchMedia("(prefers-color-scheme: dark)").matches)
		) {
			document.documentElement.classList.add("dark");
			if (!theme) localStorage.setItem("theme", "dark");
		} else {
			document.documentElement.classList.remove("dark");
		}
	}, []);

	return (
		<QueryClientProvider client={queryClient}>
			<TooltipProvider>
				<AuthProvider>
					<Toaster />
					<Sonner />
					<BrowserRouter
						future={{
							v7_startTransition: true,
							v7_relativeSplatPath: true,
						}}
					>
						<Routes>
							<Route path="/login" element={<Login />} />
							<Route path="/register" element={<Register />} />
							<Route
								element={
									<ProtectedRoute>
										<AppLayout />
									</ProtectedRoute>
								}
							>
								<Route path="/" element={<Index />} />
								<Route path="/upload" element={<Upload />} />
								<Route path="/scan/:id" element={<ScanDetail />} />
								<Route path="/scans" element={<ScanLibrary />} />
								<Route path="/reports" element={<Reports />} />
								<Route path="/settings" element={<Settings />} />
							</Route>
							<Route path="*" element={<NotFound />} />
						</Routes>
					</BrowserRouter>
				</AuthProvider>
			</TooltipProvider>
		</QueryClientProvider>
	);
};

export default App;
