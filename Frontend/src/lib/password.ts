/** Mirrors the server's password policy so users get feedback before submitting. */
export const PASSWORD_MIN_LENGTH = 12;
const PASSWORD_MAX_BYTES = 72;

export const passwordProblem = (password: string): string | null => {
	if (password.length < PASSWORD_MIN_LENGTH) {
		return `Password must be at least ${PASSWORD_MIN_LENGTH} characters.`;
	}
	if (new TextEncoder().encode(password).length > PASSWORD_MAX_BYTES) {
		return `Password must be at most ${PASSWORD_MAX_BYTES} bytes.`;
	}
	return null;
};
