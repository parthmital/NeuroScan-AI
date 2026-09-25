import React, { createContext, useCallback, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { authApi, setUnauthorizedHandler, type LoginResult } from "@/lib/api";
import type { User } from "@/lib/types";

interface AuthContextType {
	user: User | null;
	isAuthenticated: boolean;
	isLoading: boolean;
	login: (
		username: string,
		password: string,
		otp?: string,
	) => Promise<LoginResult>;
	register: (data: Parameters<typeof authApi.register>[0]) => Promise<User>;
	logout: () => Promise<void>;
	setUser: (user: User) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export { AuthContext };

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
	children,
}) => {
	const [user, setUserState] = useState<User | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const queryClient = useQueryClient();

	const clearSession = useCallback(() => {
		setUserState(null);
		queryClient.clear();
	}, [queryClient]);

	useEffect(() => {
		// Earlier versions kept the bearer token in localStorage; remove any leftovers.
		try {
			localStorage.removeItem("token");
			localStorage.removeItem("user");
		} catch {
			// Storage may be unavailable.
		}
		setUnauthorizedHandler(clearSession);
		authApi
			.me()
			.then(setUserState)
			.catch(() => setUserState(null))
			.finally(() => setIsLoading(false));
	}, [clearSession]);

	const login = async (username: string, password: string, otp?: string) => {
		const result = await authApi.login(username, password, otp);
		if (result.status === "ok") setUserState(result.user);
		return result;
	};

	const register = async (data: Parameters<typeof authApi.register>[0]) => {
		const created = await authApi.register(data);
		setUserState(created);
		return created;
	};

	const logout = async () => {
		try {
			await authApi.logout();
		} finally {
			clearSession();
		}
	};

	return (
		<AuthContext.Provider
			value={{
				user,
				isAuthenticated: user !== null,
				isLoading,
				login,
				register,
				logout,
				setUser: setUserState,
			}}
		>
			{children}
		</AuthContext.Provider>
	);
};
