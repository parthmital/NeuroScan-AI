import type { MRIScan, TwoFactorSetup, User } from "./types";

/**
 * The single HTTP client for the backend. The session lives in an HttpOnly cookie that
 * JavaScript never sees; every request sends it and the CSRF header the API requires.
 */
const API_BASE_URL = (import.meta.env.VITE_API_URL ?? "/api").replace(
	/\/$/,
	"",
);
const CSRF_HEADERS = { "X-Requested-With": "NeuroScan" };
const PAGE_SIZE = 500;

export class ApiError extends Error {
	constructor(
		message: string,
		readonly status: number,
		readonly body: unknown,
	) {
		super(message);
		this.name = "ApiError";
	}
}

let onUnauthorized: () => void = () => {};

/** Called whenever an authenticated request is rejected because the session ended. */
export const setUnauthorizedHandler = (handler: () => void) => {
	onUnauthorized = handler;
};

const errorMessage = (body: unknown, fallback: string): string => {
	const detail = (body as { detail?: unknown } | null)?.detail;
	if (typeof detail === "string") return detail;
	if (Array.isArray(detail) && detail.length > 0) {
		return detail
			.map((item: { loc?: unknown[]; msg?: string }) => {
				const field = item.loc?.[item.loc.length - 1];
				return field ? `${field}: ${item.msg}` : item.msg;
			})
			.join("; ");
	}
	return fallback;
};

const send = async (
	path: string,
	init: RequestInit = {},
	{ authRequired = true } = {},
): Promise<Response> => {
	const response = await fetch(`${API_BASE_URL}${path}`, {
		...init,
		credentials: "include",
		headers: { ...CSRF_HEADERS, ...init.headers },
	});
	if (response.ok) return response;
	const body = await response.json().catch(() => null);
	if (response.status === 401 && authRequired) onUnauthorized();
	throw new ApiError(
		errorMessage(body, response.statusText || "Request failed"),
		response.status,
		body,
	);
};

const json = async <T>(
	path: string,
	init: RequestInit = {},
	options?: { authRequired?: boolean },
): Promise<T> => {
	const response = await send(path, init, options);
	return (response.status === 204 ? undefined : await response.json()) as T;
};

const jsonBody = (method: string, body: unknown): RequestInit => ({
	method,
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify(body),
});

const saveBlob = (blob: Blob, filename: string) => {
	const url = URL.createObjectURL(blob);
	const link = document.createElement("a");
	link.href = url;
	link.download = filename;
	document.body.appendChild(link);
	link.click();
	link.remove();
	URL.revokeObjectURL(url);
};

const download = async (path: string, fallbackName: string) => {
	const response = await send(path);
	const disposition = response.headers.get("Content-Disposition") ?? "";
	const filename = /filename="([^"]+)"/.exec(disposition)?.[1] ?? fallbackName;
	saveBlob(await response.blob(), filename);
};

interface AuthResponse {
	user: User;
}

export type LoginResult =
	{ status: "ok"; user: User } | { status: "otp-required" };

export const authApi = {
	me: () => json<User>("/auth/me", {}, { authRequired: false }),

	async login(
		username: string,
		password: string,
		otp?: string,
	): Promise<LoginResult> {
		try {
			const { user } = await json<AuthResponse>(
				"/auth/login",
				jsonBody("POST", { username, password, ...(otp ? { otp } : {}) }),
				{ authRequired: false },
			);
			return { status: "ok", user };
		} catch (error) {
			const otpRequired =
				error instanceof ApiError &&
				(error.body as { otpRequired?: boolean } | null)?.otpRequired;
			if (otpRequired) return { status: "otp-required" };
			throw error;
		}
	},

	register: async (data: {
		username: string;
		email: string;
		password: string;
		fullName: string;
		title?: string;
		department?: string;
		institution?: string;
	}) =>
		(
			await json<AuthResponse>("/auth/register", jsonBody("POST", data), {
				authRequired: false,
			})
		).user,

	logout: () =>
		json<void>("/auth/logout", { method: "POST" }, { authRequired: false }),

	updateProfile: (data: Partial<Omit<User, "id" | "username" | "createdAt">>) =>
		json<User>("/auth/me", jsonBody("PUT", data)),

	changePassword: async (currentPassword: string, newPassword: string) =>
		(
			await json<AuthResponse>(
				"/auth/password",
				jsonBody("POST", { currentPassword, newPassword }),
			)
		).user,

	startTwoFactorSetup: (password: string) =>
		json<TwoFactorSetup>("/auth/2fa/setup", jsonBody("POST", { password })),

	enableTwoFactor: async (code: string) =>
		(await json<AuthResponse>("/auth/2fa/enable", jsonBody("POST", { code })))
			.user,

	disableTwoFactor: async (password: string, code: string) =>
		(
			await json<AuthResponse>(
				"/auth/2fa/disable",
				jsonBody("POST", { password, code }),
			)
		).user,
};

export const scansApi = {
	/** Every scan of the current user, fetched in bounded pages. */
	async list(): Promise<MRIScan[]> {
		const scans: MRIScan[] = [];
		for (let offset = 0; ; offset += PAGE_SIZE) {
			const response = await send(`/scans?limit=${PAGE_SIZE}&offset=${offset}`);
			const page = (await response.json()) as MRIScan[];
			scans.push(...page);
			const total = Number(response.headers.get("X-Total-Count") ?? 0);
			if (page.length < PAGE_SIZE || scans.length >= total) return scans;
		}
	},

	get: (id: string) => json<MRIScan>(`/scans/${encodeURIComponent(id)}`),

	update: (id: string, data: { patientName?: string; patientId?: string }) =>
		json<MRIScan>(`/scans/${encodeURIComponent(id)}`, jsonBody("PUT", data)),

	remove: (id: string) =>
		json<{ message: string }>(`/scans/${encodeURIComponent(id)}`, {
			method: "DELETE",
		}),

	retry: (id: string) =>
		json<MRIScan>(`/scans/${encodeURIComponent(id)}/retry`, { method: "POST" }),

	upload(files: File[], patientName: string, patientId: string) {
		const form = new FormData();
		files.forEach((file) => form.append("files", file));
		form.append("patientName", patientName);
		if (patientId) form.append("patientId", patientId);
		return json<MRIScan>("/process-mri", { method: "POST", body: form });
	},

	async sliceImage(id: string, index: number, modality: string) {
		const response = await send(
			`/scans/${encodeURIComponent(id)}/slice/${index}?modality=${encodeURIComponent(modality.toLowerCase())}`,
		);
		return {
			image: await response.blob(),
			totalSlices: Number(response.headers.get("X-Total-Slices") ?? 0),
		};
	},

	async segmentationImage(id: string, index: number) {
		return (
			await send(`/scans/${encodeURIComponent(id)}/segmentation/${index}`)
		).blob();
	},

	async volume(id: string, key: string) {
		return (
			await send(
				`/scans/${encodeURIComponent(id)}/download/${encodeURIComponent(key.toLowerCase())}`,
			)
		).arrayBuffer();
	},
};

export const reportsApi = {
	pdf: (id: string) =>
		download(`/scans/${encodeURIComponent(id)}/report.pdf`, `${id}-report.pdf`),
	csv: (id: string) =>
		download(`/scans/${encodeURIComponent(id)}/report.csv`, `${id}-report.csv`),
	exportAll: () => download("/scans/export.csv", "neuroscan-reports.csv"),
};
