const HEX_COLOUR = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i;

export const isHexColour = (value: unknown): value is string =>
	typeof value === "string" && HEX_COLOUR.test(value);

/** "#rrggbb" to 0-255 channels; unparseable input falls back to white. */
export const parseHex = (hex: string) => {
	const match = HEX_COLOUR.exec(hex);
	if (!match) return { r: 255, g: 255, b: 255 };
	return {
		r: parseInt(match[1], 16),
		g: parseInt(match[2], 16),
		b: parseInt(match[3], 16),
	};
};
