import { useRef, useEffect, useState, useCallback } from "react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { scansApi } from "@/lib/api";
import { parseHex } from "@/lib/color";
import type { OverlayColors, Overlays } from "@/lib/types";

interface SliceViewerProps {
	scanId?: string;
	modalities?: string[];
	hasSegmentation: boolean;
	overlays: Overlays;
	overlayColors: OverlayColors;
}

const DEFAULT_TOTAL_SLICES = 155;
const CANVAS_SIZE = 480;

export const SliceViewer = ({
	scanId,
	modalities,
	hasSegmentation,
	overlays,
	overlayColors,
}: SliceViewerProps) => {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const [slice, setSlice] = useState(78);
	const [totalSlices, setTotalSlices] = useState(DEFAULT_TOTAL_SLICES);
	const [isLoading, setIsLoading] = useState(false);
	const [selectedModality, setSelectedModality] = useState(
		modalities?.[0] || "FLAIR",
	);
	// Debounced state for fetching to prevent glitching
	const [debouncedSlice, setDebouncedSlice] = useState(slice);
	const [debouncedOverlays, setDebouncedOverlays] = useState(overlays);
	const [debouncedOverlayColors, setDebouncedOverlayColors] =
		useState(overlayColors);

	useEffect(() => {
		const timer = setTimeout(() => {
			setDebouncedSlice(slice);
			setDebouncedOverlays(overlays);
			setDebouncedOverlayColors(overlayColors);
		}, 300); // 300ms delay

		return () => clearTimeout(timer);
	}, [slice, overlays, overlayColors]);

	const fetchSlice = useCallback(async () => {
		if (!scanId || !canvasRef.current) return;

		const canvas = canvasRef.current;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		setIsLoading(true);
		try {
			const { image, totalSlices: total } = await scansApi.sliceImage(
				scanId,
				debouncedSlice,
				selectedModality,
			);
			if (total > 0) {
				setTotalSlices(total);
				setSlice((current) => Math.min(current, total - 1));
			}

			const scanBitmap = await createImageBitmap(image);

			// Clear canvas
			ctx.fillStyle = "#0a0a0a";
			ctx.fillRect(0, 0, canvas.width, canvas.height);

			// Draw Scan Image centered and scaled
			const scale = Math.min(
				canvas.width / scanBitmap.width,
				canvas.height / scanBitmap.height,
			);
			const x = (canvas.width - scanBitmap.width * scale) / 2;
			const y = (canvas.height - scanBitmap.height * scale) / 2;
			ctx.drawImage(
				scanBitmap,
				x,
				y,
				scanBitmap.width * scale,
				scanBitmap.height * scale,
			);

			// Fetch and Draw Segmentation Overlay
			try {
				if (hasSegmentation) {
					const segBlob = await scansApi.segmentationImage(
						scanId,
						debouncedSlice,
					);
					const segBitmap = await createImageBitmap(segBlob);

					// Create offscreen canvas for pixel manipulation
					const offCanvas = document.createElement("canvas");
					offCanvas.width = segBitmap.width;
					offCanvas.height = segBitmap.height;
					const offCtx = offCanvas.getContext("2d");
					if (offCtx) {
						offCtx.drawImage(segBitmap, 0, 0);
						const imageData = offCtx.getImageData(
							0,
							0,
							segBitmap.width,
							segBitmap.height,
						);
						const data = imageData.data;

						const wtColor = parseHex(debouncedOverlayColors.wt);
						const tcColor = parseHex(debouncedOverlayColors.tc);
						const etColor = parseHex(debouncedOverlayColors.et);

						for (let i = 0; i < data.length; i += 4) {
							// Mask channels: R=WT, G=TC, B=ET
							const wtProb = data[i];
							const tcProb = data[i + 1];
							const etProb = data[i + 2];
							// data[i+3] is dummy alpha from PNG load (255)

							// Determine dominant class or blend
							let r = 0,
								g = 0,
								b = 0,
								a = 0;

							// Simple additive blending
							if (debouncedOverlays.wt && wtProb > 10) {
								r += wtColor.r;
								g += wtColor.g;
								b += wtColor.b;
								a = Math.max(a, wtProb);
							}
							if (debouncedOverlays.tc && tcProb > 10) {
								r += tcColor.r;
								g += tcColor.g;
								b += tcColor.b;
								a = Math.max(a, tcProb);
							}
							if (debouncedOverlays.et && etProb > 10) {
								r += etColor.r;
								g += etColor.g;
								b += etColor.b;
								a = Math.max(a, etProb);
							}

							if (a > 0) {
								data[i] = r;
								data[i + 1] = g;
								data[i + 2] = b;
								// Apply opacity
								data[i + 3] = Math.min(255, a * debouncedOverlays.opacity);
							} else {
								data[i + 3] = 0;
							}
						}

						offCtx.putImageData(imageData, 0, 0);

						// Draw Overlay (scaled to match scan)
						ctx.drawImage(
							offCanvas,
							x,
							y,
							scanBitmap.width * scale,
							scanBitmap.height * scale,
						);
					}
				}
			} catch (err) {
				console.warn("Segmentation fetch failed or not available", err);
			}

			// Orientation markers
			ctx.fillStyle = "rgba(0, 200, 180, 0.5)";
			ctx.font = "13px monospace";
			ctx.fillText("A", canvas.width / 2 - 4, 18);
			ctx.fillText("P", canvas.width / 2 - 4, canvas.height - 8);
			ctx.fillText("R", 8, canvas.height / 2 + 4);
			ctx.fillText("L", canvas.width - 16, canvas.height / 2 + 4);

			setIsLoading(false);
		} catch (error) {
			console.error("Error fetching slice:", error);
			setIsLoading(false);
		}
	}, [
		scanId,
		hasSegmentation,
		debouncedSlice,
		selectedModality,
		debouncedOverlays,
		debouncedOverlayColors,
	]);

	useEffect(() => {
		if (scanId) {
			// Fetching from the API is external sync; its loading flag is set before the first await.
			// eslint-disable-next-line react-hooks/set-state-in-effect
			fetchSlice();
		}
	}, [fetchSlice, scanId]);

	// Resize canvas to fit container
	useEffect(() => {
		const container = containerRef.current;
		const canvas = canvasRef.current;
		if (!container || !canvas) return;

		const observer = new ResizeObserver((entries) => {
			const { width } = entries[0].contentRect;
			const size = Math.round(width);
			canvas.width = size;
			canvas.height = size;
			if (scanId) fetchSlice();
		});

		observer.observe(container);
		return () => observer.disconnect();
	}, [fetchSlice, scanId]);

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-3 px-1 sm:gap-4">
				<span className="w-14 shrink-0 font-mono text-xs text-muted-foreground">
					Slice {slice + 1}
				</span>
				<Slider
					aria-label="Slice"
					value={[slice]}
					onValueChange={([v]) => setSlice(v)}
					min={0}
					max={totalSlices - 1}
					step={1}
					className="flex-1"
				/>
				<span className="w-8 shrink-0 text-right font-mono text-xs text-muted-foreground">
					{totalSlices}
				</span>
			</div>
			<div
				ref={containerRef}
				className="relative flex aspect-square items-center justify-center overflow-hidden rounded-xl border border-border bg-black"
			>
				{isLoading && (
					<div className="absolute inset-0 z-10 flex items-center justify-center bg-black/40 backdrop-blur-xs">
						<div className="flex flex-col items-center gap-2">
							<div className="h-8 w-8 animate-spin rounded-full border-2 border-medical border-t-transparent" />
							<p className="font-mono text-[10px] uppercase tracking-widest text-medical">
								Loading Slice
							</p>
						</div>
					</div>
				)}
				<canvas
					ref={canvasRef}
					width={CANVAS_SIZE}
					height={CANVAS_SIZE}
					className="h-full w-full object-contain"
				/>

				{/* HUD Overlay */}
				<div className="pointer-events-none absolute left-4 top-4">
					<p className="font-mono text-[10px] text-medical/80">
						Slice: {slice + 1}/{totalSlices}
					</p>
					<p className="font-mono text-[10px] text-medical/80">
						Modality: {selectedModality}
					</p>
				</div>
			</div>
			{/* Modality Selector */}
			{modalities && modalities.length > 0 && (
				<div className="flex justify-center gap-2 pt-1">
					{modalities.map((m) => (
						<button
							key={m}
							onClick={() => setSelectedModality(m)}
							aria-pressed={selectedModality === m}
							className={cn(
								"rounded-md border px-3 py-1.5 text-xs font-semibold transition-all",
								selectedModality === m
									? "border-medical bg-medical text-medical-foreground shadow-xs"
									: "border-border bg-card text-muted-foreground hover:border-medical/50 hover:bg-muted/50",
							)}
						>
							{m}
						</button>
					))}
				</div>
			)}
		</div>
	);
};
