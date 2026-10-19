import { createContext } from "react";
import type { authApi, LoginResult } from "@/lib/api";
import type { User } from "@/lib/types";

export interface AuthContextType {
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

export const AuthContext = createContext<AuthContextType | undefined>(
	undefined,
);
