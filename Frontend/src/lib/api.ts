const API_BASE_URL = "http://localhost:8000/api";

const getAuthHeader = () => {
	const token = localStorage.getItem("token");
	return token ? { Authorization: `Bearer ${token}` } : {};
};

const handleResponse = async (response: Response) => {
	if (response.status === 401) {
		localStorage.removeItem("token");
		localStorage.removeItem("user");
		window.location.href = "/login";
		throw new Error("Unauthorized");
	}
	if (!response.ok) {
		const error = await response
			.json()
			.catch(() => ({ detail: response.statusText }));
		throw new Error(error.detail || "API request failed");
	}
	return response.json();
};

export const fetchScans = async () => {
	const response = await fetch(`${API_BASE_URL}/scans`, {
		headers: { ...getAuthHeader() },
	});
	return handleResponse(response);
};

export const fetchScan = async (id: string) => {
	const response = await fetch(`${API_BASE_URL}/scans/${id}`, {
		headers: { ...getAuthHeader() },
	});
	return handleResponse(response);
};

export const processMRI = async (
	files: File[],
	patientName?: string,
	patientId?: string,
) => {
	const formData = new FormData();
	files.forEach((file) => formData.append("files", file));
	if (patientName) formData.append("patientName", patientName);
	if (patientId) formData.append("patientId", patientId);

	const response = await fetch(`${API_BASE_URL}/process-mri`, {
		method: "POST",
		headers: { ...getAuthHeader() },
		body: formData,
	});
	return handleResponse(response);
};
