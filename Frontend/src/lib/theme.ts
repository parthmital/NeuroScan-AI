/** The initial theme is applied by public/theme-init.js before React loads. */
export const isDarkTheme = () =>
	document.documentElement.classList.contains("dark");

export const setTheme = (dark: boolean) => {
	document.documentElement.classList.toggle("dark", dark);
	try {
		localStorage.setItem("theme", dark ? "dark" : "light");
	} catch {
		// Storage may be unavailable; the theme still applies for this session.
	}
};
