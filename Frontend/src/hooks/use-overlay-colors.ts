import { useCallback, useState } from "react";
import { isHexColour } from "@/lib/color";
import type { OverlayColors, TumourRegion } from "@/lib/types";

const STORAGE_KEY = "overlayColors";

export const DEFAULT_OVERLAY_COLORS: OverlayColors = {
	wt: "#eab308",
	tc: "#ef4444",
	et: "#3b82f6",
};

/** Reads saved colours, ignoring anything malformed or tampered with. */
const readStoredColors = (): OverlayColors => {
	try {
		const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
		if (saved && typeof saved === "object") {
			return {
				wt: isHexColour(saved.wt) ? saved.wt : DEFAULT_OVERLAY_COLORS.wt,
				tc: isHexColour(saved.tc) ? saved.tc : DEFAULT_OVERLAY_COLORS.tc,
				et: isHexColour(saved.et) ? saved.et : DEFAULT_OVERLAY_COLORS.et,
			};
		}
	} catch {
		// Unreadable storage falls back to the defaults.
	}
	return DEFAULT_OVERLAY_COLORS;
};

export const useOverlayColors = () => {
	const [colors, setColors] = useState<OverlayColors>(readStoredColors);

	const setColor = useCallback((region: TumourRegion, color: string) => {
		if (!isHexColour(color)) return;
		setColors((previous) => {
			const next = { ...previous, [region]: color };
			try {
				localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
			} catch {
				// Storage may be unavailable (private mode); the choice still applies this session.
			}
			return next;
		});
	}, []);

	return { colors, setColor };
};
