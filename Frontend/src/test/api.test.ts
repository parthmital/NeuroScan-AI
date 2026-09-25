import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, authApi, scansApi, setUnauthorizedHandler } from "@/lib/api";

const jsonResponse = (
	body: unknown,
	status = 200,
	headers: Record<string, string> = {},
) =>
	new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json", ...headers },
	});

describe("api client", () => {
	const fetchMock = vi.fn();

	beforeEach(() => {
		fetchMock.mockReset();
		vi.stubGlobal("fetch", fetchMock);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		setUnauthorizedHandler(() => {});
	});

	it("sends the session cookie and CSRF header, never a bearer token", async () => {
		fetchMock.mockResolvedValue(jsonResponse({ id: "scan-1" }));
		await scansApi.get("scan-1");
		const [url, init] = fetchMock.mock.calls[0];
		expect(url).toBe("/api/scans/scan-1");
		expect(init.credentials).toBe("include");
		expect(init.headers["X-Requested-With"]).toBe("NeuroScan");
		expect(init.headers.Authorization).toBeUndefined();
	});

	it("encodes path segments", async () => {
		fetchMock.mockResolvedValue(jsonResponse({}));
		await scansApi.get("../../auth/me");
		expect(fetchMock.mock.calls[0][0]).toBe("/api/scans/..%2F..%2Fauth%2Fme");
	});

	it("reports the server's message and notifies on an ended session", async () => {
		const onUnauthorized = vi.fn();
		setUnauthorizedHandler(onUnauthorized);
		fetchMock.mockResolvedValue(
			jsonResponse({ detail: "Not authenticated" }, 401),
		);
		const error = await scansApi.list().catch((e) => e);
		expect(error).toBeInstanceOf(ApiError);
		expect(error.message).toBe("Not authenticated");
		expect(error.status).toBe(401);
		expect(onUnauthorized).toHaveBeenCalledOnce();
	});

	it("does not treat a failed login as an ended session", async () => {
		const onUnauthorized = vi.fn();
		setUnauthorizedHandler(onUnauthorized);
		fetchMock.mockResolvedValue(
			jsonResponse({ detail: "Incorrect username or password" }, 401),
		);
		await expect(authApi.login("a", "b")).rejects.toThrow(
			"Incorrect username or password",
		);
		expect(onUnauthorized).not.toHaveBeenCalled();
	});

	it("turns a two-factor challenge into an otp-required result", async () => {
		fetchMock.mockResolvedValue(
			jsonResponse(
				{ detail: "Two-factor code required", otpRequired: true },
				401,
			),
		);
		await expect(authApi.login("a", "b")).resolves.toEqual({
			status: "otp-required",
		});
	});

	it("formats validation errors field by field", async () => {
		fetchMock.mockResolvedValue(
			jsonResponse(
				{
					detail: [{ loc: ["body", "password"], msg: "Password too short" }],
				},
				422,
			),
		);
		await expect(
			authApi.register({
				username: "u",
				email: "e",
				password: "p",
				fullName: "f",
			}),
		).rejects.toThrow("password: Password too short");
	});

	it("pages through all scans", async () => {
		const page = (count: number, start: number) =>
			Array.from({ length: count }, (_, i) => ({ id: `scan-${start + i}` }));
		fetchMock
			.mockResolvedValueOnce(
				jsonResponse(page(500, 0), 200, { "X-Total-Count": "501" }),
			)
			.mockResolvedValueOnce(
				jsonResponse(page(1, 500), 200, { "X-Total-Count": "501" }),
			);
		const scans = await scansApi.list();
		expect(scans).toHaveLength(501);
		expect(fetchMock.mock.calls[1][0]).toBe("/api/scans?limit=500&offset=500");
	});
});
