import { lazy, Suspense } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
	BrowserRouter,
	Navigate,
	Route,
	Routes,
	useLocation,
} from "react-router-dom";
import { Loader2 } from "lucide-react";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "./components/layout/AppLayout";
import { AuthProvider } from "./components/auth/AuthContext";
import { useAuth } from "./components/auth/useAuth";
import { ApiError } from "./lib/api";
import Login from "./pages/Login";
import Register from "./pages/Register";

// Signed-in pages load on demand, keeping the login screen's bundle small.
const Index = lazy(() => import("./pages/Index"));
const Upload = lazy(() => import("./pages/Upload"));
const ScanDetail = lazy(() => import("./pages/ScanDetail"));
const ScanLibrary = lazy(() => import("./pages/ScanLibrary"));
const Reports = lazy(() => import("./pages/Reports"));
const Settings = lazy(() => import("./pages/Settings"));
const NotFound = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient({
	defaultOptions: {
		queries: {
			// Client errors (missing scan, expired session) will not succeed on retry.
			retry: (failureCount, error) =>
				!(error instanceof ApiError && error.status < 500) && failureCount < 2,
		},
	},
});

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
	const { isAuthenticated, isLoading } = useAuth();
	const location = useLocation();

	if (isLoading) {
		return (
			<div className="flex h-screen w-screen items-center justify-center bg-background">
				<div role="status" className="flex flex-col items-center gap-3">
					<Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
					<p className="font-mono text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
						Checking session
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

const App = () => (
	<QueryClientProvider client={queryClient}>
		<TooltipProvider>
			<AuthProvider>
				<Toaster />
				<BrowserRouter>
					<Suspense fallback={null}>
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
					</Suspense>
				</BrowserRouter>
			</AuthProvider>
		</TooltipProvider>
	</QueryClientProvider>
);

export default App;
