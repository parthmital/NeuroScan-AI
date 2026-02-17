import { APP_CONFIG } from "@/lib/config";
import { AppConfig } from "@/lib/types";

export const useConfig = () => {
	return {
		data: APP_CONFIG,
		isLoading: false,
		error: null,
	};
};
