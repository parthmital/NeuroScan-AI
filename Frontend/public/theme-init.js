// Applies the saved (or system) colour theme before first paint. Loaded as a blocking
// external script so the page needs no inline script and works under a strict CSP.
(function () {
	try {
		var theme = localStorage.getItem("theme");
		var dark =
			theme === "dark" ||
			(!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
		document.documentElement.classList.toggle("dark", dark);
	} catch (error) {
		// Storage can be unavailable; the light theme is used.
	}
})();
