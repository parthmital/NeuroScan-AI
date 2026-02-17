import { useRef, useEffect, useState, useCallback } from "react";
import { Slider } from "@/components/ui/slider";
import { cn, hexToRgba } from "@/lib/utils";

interface Overlays {
	wt: boolean;
	tc: boolean;
	et: boolean;
	opacity: number;
}

interface SliceViewerProps {
	scanId?: string;
	modalities?: string[];
	overlays: Overlays;
	overlayColors?: {
		wt: string;
		tc: string;
		et: string;
	};
}

const DEFAULT_TOTAL_SLICES = 155;
const CANVAS_SIZE = 480;

export const SliceViewer = ({
	scanId,
	modalities,
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
	// Parse hex color to RGB object
	const getRgb = (hex: string) => {
		const bigint = parseInt(hex.slice(1), 16);
		const r = (bigint >> 16) & 255;
		const g = (bigint >> 8) & 255;
		const b = bigint & 255;
		return { r, g, b };
	};

	const fetchSlice = useCallback(async () => {
		if (!scanId || !canvasRef.current) return;

		const canvas = canvasRef.current;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		setIsLoading(true);
		try {
			const token = localStorage.getItem("token");
			const headers = { Authorization: `Bearer ${token}` };

			// Fetch Scan Slice
			const scanRes = await fetch(
				`http://localhost:8000/api/scans/${scanId}/slice/${debouncedSlice}?modality=${selectedModality.toLowerCase()}`,
				{ headers },
			);

			if (!scanRes.ok) throw new Error("Failed to fetch slice");

			// Update total slices if header is present
			const total = scanRes.headers.get("X-Total-Slices");
			if (total) setTotalSlices(parseInt(total));

			const scanBlob = await scanRes.blob();
			const scanBitmap = await createImageBitmap(scanBlob);

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
				const segRes = await fetch(
					`http://localhost:8000/api/scans/${scanId}/segmentation/${debouncedSlice}`,
					{ headers },
				);

				if (segRes.ok) {
					const segBlob = await segRes.blob();
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

						const wtColor = debouncedOverlayColors
							? getRgb(debouncedOverlayColors.wt)
							: { r: 234, g: 179, b: 8 };
						const tcColor = debouncedOverlayColors
							? getRgb(debouncedOverlayColors.tc)
							: { r: 239, g: 68, b: 68 };
						const etColor = debouncedOverlayColors
							? getRgb(debouncedOverlayColors.et)
							: { r: 59, g: 130, b: 246 };

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

			// HUD overlay (Now moved to DOM)
			// ctx.fillStyle = "rgba(0, 200, 180, 0.7)";
			// ctx.font = "11px monospace";
			// ctx.fillText(`Slice: ${debouncedSlice + 1}/${totalSlices}`, 12, 20);
			// ctx.fillText(`Modality: ${selectedModality}`, 12, 36);

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
		debouncedSlice,
		selectedModality,
		totalSlices,
		debouncedOverlays,
		debouncedOverlayColors,
	]);

	useEffect(() => {
		if (scanId) {
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
			<div className="flex items-center gap-3 sm:gap-4 px-1">
				<span className="text-xs font-mono text-muted-foreground w-14 shrink-0">
					Slice {slice + 1}
				</span>
				<Slider
					value={[slice]}
					onValueChange={([v]) => setSlice(v)}
					min={0}
					max={totalSlices - 1}
					step={1}
					className="flex-1"
				/>
				<span className="text-xs font-mono text-muted-foreground w-8 text-right shrink-0">
					{totalSlices}
				</span>
			</div>
			<div
				ref={containerRef}
				className="relative bg-black rounded-xl overflow-hidden border border-border aspect-square flex items-center justify-center"
			>
				{isLoading && (
					<div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-10">
						<div className="flex flex-col items-center gap-2">
							<div className="w-8 h-8 border-2 border-medical border-t-transparent rounded-full animate-spin" />
							<p className="text-[10px] text-medical font-mono tracking-widest uppercase">
								Loading Slice
							</p>
						</div>
					</div>
				)}
				<canvas
					ref={canvasRef}
					width={CANVAS_SIZE}
					height={CANVAS_SIZE}
					className="w-full h-full object-contain"
				/>

				{/* HUD Overlay */}
				<div className="absolute top-4 left-4 pointer-events-none">
					<p className="text-[10px] font-mono text-medical/80">
						Slice: {slice + 1}/{totalSlices}
					</p>
					<p className="text-[10px] font-mono text-medical/80">
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
							className={cn(
								"px-3 py-1.5 text-xs font-semibold rounded-md border transition-all",
								selectedModality === m
									? "bg-medical text-medical-foreground border-medical shadow-sm"
									: "bg-card text-muted-foreground border-border hover:border-medical/50 hover:bg-muted/50",
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
