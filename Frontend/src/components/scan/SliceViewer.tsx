import { useCallback, useEffect, useRef, useState } from "react";
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
// Slices kept in memory per viewer; enough for every slice of one or two BraTS modalities.
const CACHE_LIMIT = 320;
// Neighbouring slices fetched after the current one, so stepping through feels instant.
const PREFETCH_RADIUS = 2;

const scanKey = (scanId: string, modality: string, index: number) =>
	`${scanId}:${modality}:${index}`;
const maskKey = (scanId: string, index: number) => `${scanId}:${index}`;

/** Adds an entry, dropping the oldest ones beyond the cache limit. */
const remember = <T,>(map: ReadonlyMap<string, T>, key: string, value: T) => {
	const next = new Map(map);
	next.set(key, value);
	for (const oldest of next.keys()) {
		if (next.size <= CACHE_LIMIT) break;
		next.delete(oldest);
	}
	return next;
};

const toImageData = (bitmap: ImageBitmap): ImageData | null => {
	const canvas = document.createElement("canvas");
	canvas.width = bitmap.width;
	canvas.height = bitmap.height;
	const ctx = canvas.getContext("2d");
	if (!ctx) return null;
	ctx.drawImage(bitmap, 0, 0);
	return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
};

/** Colours the mask (R = WT, G = TC, B = ET) with the chosen colours and opacity. */
const colouriseMask = (
	mask: ImageData,
	overlays: Overlays,
	colors: OverlayColors,
): HTMLCanvasElement | null => {
	const layer = document.createElement("canvas");
	layer.width = mask.width;
	layer.height = mask.height;
	const ctx = layer.getContext("2d");
	if (!ctx) return null;
	const output = ctx.createImageData(mask.width, mask.height);
	const source = mask.data;
	const data = output.data;
	const regions = [
		[overlays.wt, parseHex(colors.wt)],
		[overlays.tc, parseHex(colors.tc)],
		[overlays.et, parseHex(colors.et)],
	] as const;
	for (let i = 0; i < source.length; i += 4) {
		let r = 0;
		let g = 0;
		let b = 0;
		let a = 0;
		regions.forEach(([visible, colour], channel) => {
			const value = source[i + channel];
			if (visible && value > 10) {
				r += colour.r;
				g += colour.g;
				b += colour.b;
				a = Math.max(a, value);
			}
		});
		if (a > 0) {
			data[i] = r;
			data[i + 1] = g;
			data[i + 2] = b;
			data[i + 3] = Math.min(255, a * overlays.opacity);
		}
	}
	ctx.putImageData(output, 0, 0);
	return layer;
};

interface Frame {
	scan: ImageBitmap;
	mask: ImageData | null;
}

const drawFrame = (
	canvas: HTMLCanvasElement,
	frame: Frame,
	overlays: Overlays,
	colors: OverlayColors,
) => {
	const ctx = canvas.getContext("2d");
	if (!ctx) return;
	ctx.fillStyle = "#0a0a0a";
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	const { scan, mask } = frame;
	const scale = Math.min(
		canvas.width / scan.width,
		canvas.height / scan.height,
	);
	const width = scan.width * scale;
	const height = scan.height * scale;
	const x = (canvas.width - width) / 2;
	const y = (canvas.height - height) / 2;
	ctx.imageSmoothingQuality = "high";
	ctx.drawImage(scan, x, y, width, height);

	const layer = mask && colouriseMask(mask, overlays, colors);
	if (layer) ctx.drawImage(layer, x, y, width, height);

	// Orientation markers
	ctx.fillStyle = "rgba(0, 200, 180, 0.5)";
	ctx.font = "13px monospace";
	ctx.fillText("A", canvas.width / 2 - 4, 18);
	ctx.fillText("P", canvas.width / 2 - 4, canvas.height - 8);
	ctx.fillText("R", 8, canvas.height / 2 + 4);
	ctx.fillText("L", canvas.width - 16, canvas.height / 2 + 4);
};

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
	const [selectedModality, setSelectedModality] = useState(
		modalities?.[0] || "FLAIR",
	);
	const [canvasSize, setCanvasSize] = useState(CANVAS_SIZE);
	const [scans, setScans] = useState<ReadonlyMap<string, ImageBitmap>>(
		new Map(),
	);
	// null: the mask for that slice could not be loaded, so the slice is shown without it.
	const [masks, setMasks] = useState<ReadonlyMap<string, ImageData | null>>(
		new Map(),
	);
	const [failedKey, setFailedKey] = useState<string | null>(null);

	// Mirrors of the caches and the wanted slice for the async loader, which outlives renders.
	const cache = useRef({ scans, masks });
	const wanted = useRef({ modality: selectedModality, index: slice });
	const totalRef = useRef(totalSlices);
	const loading = useRef(new Map<string, Promise<void>>());
	const pumping = useRef(false);
	const lastFrame = useRef<Frame | null>(null);

	const isCached = useCallback(
		(modality: string, index: number) =>
			!!scanId &&
			cache.current.scans.has(scanKey(scanId, modality, index)) &&
			(!hasSegmentation || cache.current.masks.has(maskKey(scanId, index))),
		[scanId, hasSegmentation],
	);

	/** Fetches one slice (and its mask) once, however often it is asked for. */
	const load = useCallback(
		(modality: string, index: number): Promise<void> => {
			if (!scanId || isCached(modality, index)) return Promise.resolve();
			const key = scanKey(scanId, modality, index);
			const pending = loading.current.get(key);
			if (pending) return pending;

			const request = (async () => {
				const mKey = maskKey(scanId, index);
				const [{ image, totalSlices: total }, mask] = await Promise.all([
					scansApi.sliceImage(scanId, index, modality),
					hasSegmentation && !cache.current.masks.has(mKey)
						? scansApi
								.segmentationImage(scanId, index)
								.then(createImageBitmap)
								.then(toImageData)
								.catch(() => null)
						: undefined,
				]);
				const bitmap = await createImageBitmap(image);
				const next = {
					scans: remember(cache.current.scans, key, bitmap),
					masks:
						mask === undefined
							? cache.current.masks
							: remember(cache.current.masks, mKey, mask),
				};
				cache.current = next;
				setScans(next.scans);
				setMasks(next.masks);
				if (total > 0 && total !== totalRef.current) {
					totalRef.current = total;
					setTotalSlices(total);
					setSlice((current) => Math.min(current, total - 1));
				}
			})().finally(() => loading.current.delete(key));
			loading.current.set(key, request);
			return request;
		},
		[scanId, hasSegmentation, isCached],
	);

	/**
	 * Loads the wanted slice, one request at a time. While the slider moves, only the newest
	 * position is fetched when a request finishes, so dragging never queues a backlog.
	 */
	const pump = useCallback(async () => {
		if (pumping.current) return;
		pumping.current = true;
		try {
			for (;;) {
				const { modality, index } = wanted.current;
				if (isCached(modality, index)) break;
				try {
					await load(modality, index);
				} catch {
					if (scanId) setFailedKey(scanKey(scanId, modality, index));
					return;
				}
			}
		} finally {
			pumping.current = false;
		}
		const target = wanted.current;
		for (let offset = 1; offset <= PREFETCH_RADIUS; offset++) {
			for (const neighbour of [target.index + offset, target.index - offset]) {
				// Stop once the user has moved on; the new position gets its own prefetch.
				if (wanted.current !== target) return;
				if (neighbour >= 0 && neighbour < totalRef.current) {
					await load(target.modality, neighbour).catch(() => undefined);
				}
			}
		}
	}, [isCached, load, scanId]);

	useEffect(() => {
		wanted.current = { modality: selectedModality, index: slice };
		void pump();
	}, [slice, selectedModality, pump]);

	const currentScan = scanId
		? scans.get(scanKey(scanId, selectedModality, slice))
		: undefined;
	const currentMask = scanId ? masks.get(maskKey(scanId, slice)) : undefined;
	const failed =
		!currentScan &&
		!!scanId &&
		failedKey === scanKey(scanId, selectedModality, slice);
	const waiting = !currentScan && !failed;

	// Draws from memory: overlay, colour, and opacity changes never refetch.
	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		if (canvas.width !== canvasSize) {
			canvas.width = canvasSize;
			canvas.height = canvasSize;
		}
		if (currentScan) {
			lastFrame.current = { scan: currentScan, mask: currentMask ?? null };
		}
		// Until the new slice arrives the previous one stays on screen.
		if (lastFrame.current) {
			drawFrame(canvas, lastFrame.current, overlays, overlayColors);
		}
	}, [currentScan, currentMask, overlays, overlayColors, canvasSize]);

	useEffect(() => {
		const container = containerRef.current;
		if (!container) return;
		const observer = new ResizeObserver(([entry]) => {
			setCanvasSize(Math.max(1, Math.round(entry.contentRect.width)));
		});
		observer.observe(container);
		return () => observer.disconnect();
	}, []);

	const firstLoad = waiting && scans.size === 0;

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
				aria-busy={waiting}
				className="relative flex aspect-square items-center justify-center overflow-hidden rounded-xl border border-border bg-black"
			>
				{firstLoad && (
					<div className="absolute inset-0 z-10 flex items-center justify-center">
						<div className="flex flex-col items-center gap-2">
							<div className="h-8 w-8 animate-spin rounded-full border-2 border-medical border-t-transparent" />
							<p className="font-mono text-[10px] uppercase tracking-widest text-medical">
								Loading slice
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
				{/* Shown only when a slice takes noticeably long, so quick steps never flicker. */}
				<div
					role="status"
					className={cn(
						"pointer-events-none absolute right-4 top-4 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-widest transition-opacity duration-200",
						failed ? "text-destructive" : "text-medical",
						(waiting && !firstLoad) || failed
							? "opacity-100 delay-200"
							: "opacity-0",
					)}
				>
					{failed ? (
						"Slice unavailable"
					) : (
						<>
							<span className="h-2.5 w-2.5 animate-spin rounded-full border border-medical border-t-transparent" />
							Loading
						</>
					)}
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
