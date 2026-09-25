import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

// Module boundaries (see ARCHITECTURE.md): lower layers never import higher ones.
const forbid = (...groups) => ({
	"no-restricted-imports": [
		"error",
		{
			patterns: groups.map((group) => ({
				group: [`@/${group}/*`, `@/${group}`],
				message: `This layer must not depend on ${group}.`,
			})),
		},
	],
});

export default tseslint.config(
	{ ignores: ["dist", "coverage"] },
	{
		extends: [js.configs.recommended, ...tseslint.configs.recommended],
		files: ["**/*.{ts,tsx}"],
		languageOptions: {
			ecmaVersion: 2022,
			globals: globals.browser,
		},
		plugins: {
			"react-hooks": reactHooks,
			"react-refresh": reactRefresh,
		},
		rules: {
			...reactHooks.configs.recommended.rules,
			"react-refresh/only-export-components": [
				"warn",
				{ allowConstantExport: true },
			],
			"@typescript-eslint/no-unused-vars": [
				"error",
				{ argsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
			],
			"no-console": ["error", { allow: ["warn", "error"] }],
		},
	},
	{
		files: ["src/lib/**"],
		rules: forbid("components", "hooks", "pages"),
	},
	{
		files: ["src/hooks/**"],
		rules: forbid("components", "pages"),
	},
	{
		files: ["src/components/ui/**"],
		rules: forbid(
			"hooks",
			"pages",
			"components/auth",
			"components/scan",
			"components/settings",
			"components/dashboard",
			"components/layout",
			"components/upload",
		),
	},
	{
		files: ["src/components/**"],
		rules: forbid("pages"),
	},
);
