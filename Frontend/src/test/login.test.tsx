import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "@/components/auth/AuthContext";
import Login from "@/pages/Login";

const user = {
	id: 1,
	username: "drrivera",
	email: "a@x.org",
	fullName: "Alex Rivera",
	title: null,
	department: null,
	institution: null,
	createdAt: "2026-09-26T00:00:00Z",
	totpEnabled: true,
};

const respond = (body: unknown, status = 200) =>
	Promise.resolve(
		new Response(JSON.stringify(body), {
			status,
			headers: { "Content-Type": "application/json" },
		}),
	);

const renderLogin = () =>
	render(
		<QueryClientProvider client={new QueryClient()}>
			<AuthProvider>
				<MemoryRouter initialEntries={["/login"]}>
					<Routes>
						<Route path="/login" element={<Login />} />
						<Route path="/" element={<p>Dashboard home</p>} />
					</Routes>
				</MemoryRouter>
			</AuthProvider>
		</QueryClientProvider>,
	);

describe("Login", () => {
	const fetchMock = vi.fn();

	beforeEach(() => {
		localStorage.setItem("token", "stale-bearer-token");
		vi.stubGlobal("fetch", fetchMock);
		fetchMock.mockImplementation((url: string, init: RequestInit) => {
			if (url === "/api/auth/me") return respond({ detail: "no" }, 401);
			const body = JSON.parse(String(init.body));
			if (!body.otp) {
				return respond(
					{ detail: "Two-factor code required", otpRequired: true },
					401,
				);
			}
			return body.otp === "123456"
				? respond({ user })
				: respond({ detail: "Invalid two-factor code" }, 401);
		});
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		localStorage.clear();
	});

	it("asks for the authenticator code, then signs in", async () => {
		renderLogin();
		await waitFor(() => expect(localStorage.getItem("token")).toBeNull());

		fireEvent.change(screen.getByLabelText("Username"), {
			target: { value: "drrivera" },
		});
		fireEvent.change(screen.getByLabelText("Password"), {
			target: { value: "correct-horse-battery" },
		});
		fireEvent.click(screen.getByRole("button", { name: /sign in/i }));

		const otp = await screen.findByLabelText("Authentication code");
		fireEvent.change(otp, { target: { value: "000000" } });
		fireEvent.click(screen.getByRole("button", { name: /verify/i }));
		expect(await screen.findByRole("alert")).toHaveTextContent(
			"Invalid two-factor code",
		);

		fireEvent.change(otp, { target: { value: "123456" } });
		fireEvent.click(screen.getByRole("button", { name: /verify/i }));
		expect(await screen.findByText("Dashboard home")).toBeInTheDocument();
	});

	it("validates before calling the server", async () => {
		renderLogin();
		fireEvent.click(screen.getByRole("button", { name: /sign in/i }));
		expect(await screen.findByRole("alert")).toHaveTextContent(
			"Enter your username and password.",
		);
		const loginCalls = fetchMock.mock.calls.filter(
			([url]) => url === "/api/auth/login",
		);
		expect(loginCalls).toHaveLength(0);
	});
});
