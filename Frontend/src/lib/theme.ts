/** The initial theme is applied by public/theme-init.js before React loads. */
export const isDarkTheme = () =>
	document.documentElement.classList.contains("dark");

export const setTheme = (dark: boolean) => {
	const apply = () => document.documentElement.classList.toggle("dark", dark);
	// Cross-fades the whole page where supported instead of flipping every colour at once.
	const reduceMotion = window.matchMedia?.(
		"(prefers-reduced-motion: reduce)",
	).matches;
	if (document.startViewTransition && !reduceMotion) {
		document.startViewTransition(apply);
	} else {
		apply();
	}
	try {
		localStorage.setItem("theme", dark ? "dark" : "light");
	} catch {
		// Storage may be unavailable; the theme still applies for this session.
	}
};
