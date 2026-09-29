import { useRef, useEffect, useState, useCallback } from "react";
import { RotateCcw, Sun, Contrast } from "lucide-react";
import { Segmented } from "@/components/ui/segmented";
import { Slider } from "@/components/ui/slider";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { scansApi } from "@/lib/api";
import { parseHex } from "@/lib/color";
import type { OverlayColors, Overlays } from "@/lib/types";

interface VolumeViewer3DProps {
	scanId: string;
	modalities: string[];
	overlays: Overlays;
	overlayColors: OverlayColors;
	hasSegmentation: boolean;
}

/** vtk.js colour transfer functions take channels in 0-1. */
const unitRgb = (hex: string) => {
	const { r, g, b } = parseHex(hex);
	return { r: r / 255, g: g / 255, b: b / 255 };
};

export const VolumeViewer3D = ({
	scanId,
	modalities,
	overlays,
	overlayColors,
	hasSegmentation,
}: VolumeViewer3DProps) => {
	const containerRef = useRef<HTMLDivElement>(null);
	const cleanupRef = useRef<(() => void) | null>(null);

	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [selectedModality, setSelectedModality] = useState(
		modalities?.[0] || "FLAIR",
	);
	const [brightness, setBrightness] = useState(0);
	const [contrast, setContrast] = useState(1);

	// Latest display settings, read when a volume finishes loading. Keeping them out of the
	// loading effect's dependencies stops every slider move from re-downloading the volume.
	const displayRef = useRef({ overlays, overlayColors, brightness, contrast });
	useEffect(() => {
		displayRef.current = { overlays, overlayColors, brightness, contrast };
	});

	// Exposed reset camera handle
	const resetCameraRef = useRef<(() => void) | null>(null);

	// Expose segmentation transfer functions and render window
	const segCtfunRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
	const segOfunRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
	const mainCtfunRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
	const renderWindowRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any

	const resetCamera = useCallback(() => {
		resetCameraRef.current?.();
	}, []);

	const updateSegmentationOverlays = useCallback(
		(currentOverlays: Overlays, currentColors: OverlayColors) => {
			if (
				!segCtfunRef.current ||
				!segOfunRef.current ||
				!renderWindowRef.current
			)
				return;

			const ctfun = segCtfunRef.current;
			const ofun = segOfunRef.current;

			const wtColor = unitRgb(currentColors.wt);
			const tcColor = unitRgb(currentColors.tc);
			const etColor = unitRgb(currentColors.et);

			const activeOpacity = currentOverlays.opacity ?? 0.6;

			ctfun.removeAllPoints();
			ctfun.addRGBPoint(0, 0, 0, 0); // background
			ctfun.addRGBPoint(1, wtColor.r, wtColor.g, wtColor.b);
			ctfun.addRGBPoint(2, tcColor.r, tcColor.g, tcColor.b);
			ctfun.addRGBPoint(3, etColor.r, etColor.g, etColor.b);

			ofun.removeAllPoints();
			ofun.addPoint(0, 0); // background transparent
			ofun.addPoint(1, currentOverlays.wt ? activeOpacity : 0);
			ofun.addPoint(2, currentOverlays.tc ? activeOpacity : 0);
			ofun.addPoint(3, currentOverlays.et ? activeOpacity : 0);

			renderWindowRef.current.render();
		},
		[],
	);

	const updateColorMapping = useCallback((b: number, c: number) => {
		if (!mainCtfunRef.current || !renderWindowRef.current) return;
		const ctfun = mainCtfunRef.current;

		const lower = 127.5 - 127.5 / c - b;
		const upper = 127.5 + 127.5 / c - b;

		ctfun.removeAllPoints();
		ctfun.addRGBPoint(lower, 0, 0, 0);
		ctfun.addRGBPoint(upper, 1, 1, 1);

		renderWindowRef.current.render();
	}, []);

	useEffect(() => {
		if (!containerRef.current || !scanId) return;

		// Cleanup any previous vtk instance
		cleanupRef.current?.();
		cleanupRef.current = null;
		setIsLoading(true);
		setError(null);

		let mounted = true;
		const container = containerRef.current;

		const run = async () => {
			try {
				// -------------------------------------------------------
				// Dynamic imports keep vtk.js out of Vite's static analysis
				// -------------------------------------------------------
				const [
					{ default: vtkGenericRenderWindow },
					{ default: vtkVolume },
					{ default: vtkVolumeMapper },
					{ default: vtkColorTransferFunction },
					{ default: vtkPiecewiseFunction },
					{ default: vtkImageData },
					{ default: vtkDataArray },
					{ default: vtkInteractorStyleTrackballCamera },
					nifti,
				] = await Promise.all([
					import("@kitware/vtk.js/Rendering/Misc/GenericRenderWindow"),
					import("@kitware/vtk.js/Rendering/Core/Volume"),
					import("@kitware/vtk.js/Rendering/Core/VolumeMapper"),
					import("@kitware/vtk.js/Rendering/Core/ColorTransferFunction"),
					import("@kitware/vtk.js/Common/DataModel/PiecewiseFunction"),
					import("@kitware/vtk.js/Common/DataModel/ImageData"),
					import("@kitware/vtk.js/Common/Core/DataArray"),
					import("@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera"),
					import("nifti-reader-js"),
					// Needed to register OpenGL representations for VTK classes
					import("@kitware/vtk.js/Rendering/Profiles/Volume"),
					import("@kitware/vtk.js/Rendering/Profiles/Geometry"),
				]);

				if (!mounted) return;

				// Initialise vtk render window
				const grw = vtkGenericRenderWindow.newInstance();
				grw.setContainer(container);
				grw.resize();

				const renderer = grw.getRenderer();
				const renderWindow = grw.getRenderWindow();
				renderer.setBackground(0.05, 0.05, 0.05);

				// Enable trackball interaction
				const interactor = renderWindow.getInteractor();
				const style = vtkInteractorStyleTrackballCamera.newInstance();
				interactor.setInteractorStyle(style);

				// Volume pipeline
				const volume = vtkVolume.newInstance();
				const mapper = vtkVolumeMapper.newInstance();
				mapper.setSampleDistance(0.7);
				mapper.setAutoAdjustSampleDistances(true);
				volume.setMapper(mapper);

				const ctfun = vtkColorTransferFunction.newInstance();
				const ofun = vtkPiecewiseFunction.newInstance();
				mainCtfunRef.current = ctfun;

				const volProp = volume.getProperty() as any; // eslint-disable-line @typescript-eslint/no-explicit-any
				volProp.setRGBTransferFunction(0, ctfun);
				volProp.setScalarOpacity(0, ofun);
				volProp.setInterpolationTypeToLinear();
				volProp.setShade(true);
				volProp.setAmbient(0.2);
				volProp.setDiffuse(0.7);
				volProp.setSpecular(0.3);

				renderer.addVolume(volume);

				// Segmentation pipeline (if any)
				const segVolume = vtkVolume.newInstance();
				const segMapper = vtkVolumeMapper.newInstance();
				segMapper.setSampleDistance(0.7);
				segMapper.setAutoAdjustSampleDistances(true);
				segVolume.setMapper(segMapper);

				const segCtfun = vtkColorTransferFunction.newInstance();
				const segOfun = vtkPiecewiseFunction.newInstance();
				segCtfunRef.current = segCtfun;
				segOfunRef.current = segOfun;

				const segProp = segVolume.getProperty() as any; // eslint-disable-line @typescript-eslint/no-explicit-any
				segProp.setRGBTransferFunction(0, segCtfun);
				segProp.setScalarOpacity(0, segOfun);
				segProp.setInterpolationTypeToNearest(); // Categorical classes
				segProp.setShade(false); // Do not shade the mask to keep colors bright

				renderer.addVolume(segVolume);

				// Expose render window
				renderWindowRef.current = renderWindow;

				// Expose reset camera
				resetCameraRef.current = () => {
					try {
						renderer.resetCamera();
						renderer.resetCameraClippingRange();

						// Adjust the active camera slightly if needed to ensure volume represents properly
						const camera = renderer.getActiveCamera();
						if (camera) {
							camera.azimuth(0);
							camera.elevation(0);
							camera.roll(0);
						}

						// Re-render and restore interactors
						renderWindow.render();
					} catch (_) {
						void 0;
					}
				};

				// Handle resize
				const resizeObserver = new ResizeObserver(() => {
					try {
						grw.resize();
					} catch (_) {
						/* ignore */
					}
				});
				resizeObserver.observe(container);

				// Cleanup function
				cleanupRef.current = () => {
					resizeObserver.disconnect();
					resetCameraRef.current = null;
					segCtfunRef.current = null;
					segOfunRef.current = null;
					mainCtfunRef.current = null;
					renderWindowRef.current = null;
					try {
						volume.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						mapper.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						ctfun.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						ofun.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						segVolume.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						segMapper.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						segCtfun.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						segOfun.delete();
					} catch (_) {
						/* ignore */
					}
					try {
						grw.delete();
					} catch (_) {
						/* ignore */
					}
				};

				// -------------------------------------------------------
				// Fetch and load the NIfTI volume(s)
				// -------------------------------------------------------
				const buffers = await Promise.all([
					scansApi.volume(scanId, selectedModality),
					...(hasSegmentation ? [scansApi.volume(scanId, "segmentation")] : []),
				]);
				if (!mounted) return;

				// Helper to load nifti buffer into imageData
				const loadNifti = (bufferObj: ArrayBuffer, doNormalize: boolean) => {
					let dataBuffer = bufferObj;
					if (nifti.isCompressed(dataBuffer)) {
						dataBuffer = nifti.decompress(dataBuffer);
					}
					if (!nifti.isNIFTI(dataBuffer)) {
						throw new Error("Server did not return a valid NIfTI file.");
					}

					const header = nifti.readHeader(dataBuffer);
					const image = nifti.readImage(header, dataBuffer);

					const dims = [header.dims[1], header.dims[2], header.dims[3]];
					const spacing = [
						header.pixDims[1] || 1,
						header.pixDims[2] || 1,
						header.pixDims[3] || 1,
					];

					const imageDataObj = vtkImageData.newInstance();
					imageDataObj.setDimensions(dims);
					imageDataObj.setSpacing(spacing);

					// Convert to Float32
					let typedArray: Float32Array;
					switch (header.datatypeCode) {
						case nifti.NIFTI1.TYPE_UINT8:
							typedArray = new Float32Array(new Uint8Array(image));
							break;
						case nifti.NIFTI1.TYPE_INT16:
							typedArray = new Float32Array(new Int16Array(image));
							break;
						case nifti.NIFTI1.TYPE_FLOAT32:
							typedArray = new Float32Array(image);
							break;
						default:
							typedArray = new Float32Array(image);
					}

					let finalArray = typedArray;

					if (doNormalize) {
						// Normalize to 0–255
						let min = Infinity,
							max = -Infinity;
						for (let i = 0; i < typedArray.length; i++) {
							if (typedArray[i] < min) min = typedArray[i];
							if (typedArray[i] > max) max = typedArray[i];
						}
						const range = max - min || 1;
						finalArray = new Float32Array(typedArray.length);
						for (let i = 0; i < typedArray.length; i++) {
							finalArray[i] = ((typedArray[i] - min) / range) * 255;
						}
					}

					const dataArrayObj = vtkDataArray.newInstance({
						name: "Scalars",
						values: finalArray,
						numberOfComponents: 1,
					});
					imageDataObj.getPointData().setScalars(dataArrayObj);
					return imageDataObj;
				};

				// Structural Volume
				const structuralImage = loadNifti(buffers[0], true);
				mapper.setInputData(structuralImage);

				// Segmentation Volume
				if (hasSegmentation && buffers.length > 1) {
					const segImage = loadNifti(buffers[1], false); // Don't normalize masks
					segMapper.setInputData(segImage);
				}

				const display = displayRef.current;
				updateSegmentationOverlays(display.overlays, display.overlayColors);
				// Brain-tissue pure grayscale transfer
				updateColorMapping(display.brightness, display.contrast);

				// Opacity mapping
				ofun.removeAllPoints();
				ofun.addPoint(0, 0.0); // Background noise completely transparent
				ofun.addPoint(5, 0.0); // Low-intensity noise transparent
				ofun.addPoint(40, 0.2); // Outer structures slightly visible
				ofun.addPoint(120, 0.5); // Mid-tones
				ofun.addPoint(255, 0.8); // High intensity (white matter/lesions) very visible

				renderer.resetCamera();
				renderer.resetCameraClippingRange();
				renderWindow.render();

				if (mounted) setIsLoading(false);
			} catch (err) {
				if (mounted) {
					setError(
						err instanceof Error
							? err.message
							: "Unknown error loading 3D volume.",
					);
					setIsLoading(false);
				}
			}
		};

		run();

		return () => {
			mounted = false;
			cleanupRef.current?.();
			cleanupRef.current = null;
		};
	}, [
		scanId,
		selectedModality,
		hasSegmentation,
		updateColorMapping,
		updateSegmentationOverlays,
	]);

	// Handle brightness/contrast reactivity
	useEffect(() => {
		updateColorMapping(brightness, contrast);
	}, [brightness, contrast, updateColorMapping]);

	// Handle overlay reactivity dynamically
	useEffect(() => {
		updateSegmentationOverlays(overlays, overlayColors);
	}, [overlays, overlayColors, updateSegmentationOverlays]);

	return (
		<div className="space-y-3">
			{/* Canvas container — same aspect-square + border style as SliceViewer */}
			<div
				ref={containerRef}
				className="relative flex aspect-square items-center justify-center overflow-hidden rounded-lg border border-border bg-[#0a0a0a]"
			>
				{/* Loading overlay */}
				{isLoading && (
					<div className="absolute inset-0 z-10 flex items-center justify-center bg-black/40 backdrop-blur-xs">
						<div className="flex flex-col items-center gap-2">
							<div className="h-6 w-6 animate-spin rounded-full border-2 border-white/60 border-t-transparent" />
							<p className="font-mono text-[10px] uppercase tracking-[0.12em] text-white/60">
								Loading 3D Volume
							</p>
						</div>
					</div>
				)}

				{/* Error overlay */}
				{error && !isLoading && (
					<div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/80 px-6 text-center">
						<p className="font-mono text-xs text-red-400">
							Failed to load 3D volume
						</p>
						<p className="font-mono text-[10px] text-white/50">{error}</p>
					</div>
				)}

				{/* HUD overlay — only when loaded */}
				{!isLoading && !error && (
					<>
						<div className="pointer-events-none absolute left-3 top-3 z-10 space-y-0.5 font-mono text-[10.5px] leading-4 text-white/60">
							<p>3D Volume Render</p>
							<p>Modality: {selectedModality}</p>
						</div>

						{/* Orientation hints */}
						<span className="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2 font-mono text-xs text-white/45">
							S
						</span>
						<span className="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2 font-mono text-xs text-white/45">
							I
						</span>

						{/* Reset camera */}
						<Tooltip>
							<TooltipTrigger asChild>
								<button
									onClick={resetCamera}
									aria-label="Reset camera"
									className="absolute bottom-3 right-3 z-10 flex items-center gap-1.5 rounded-md border border-white/15 bg-black/50 px-2 py-1 font-mono text-[10.5px] text-white/70 backdrop-blur-xs transition-colors hover:border-white/30 hover:text-white focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
								>
									<RotateCcw className="h-3 w-3" />
									Reset
								</button>
							</TooltipTrigger>
							<TooltipContent>Reset camera</TooltipContent>
						</Tooltip>
					</>
				)}
			</div>

			{/* Modality and Contrast Controls */}
			<div className="flex flex-col gap-4 pt-1 sm:flex-row sm:items-center sm:justify-between">
				{modalities && modalities.length > 0 && (
					<Segmented
						aria-label="Modality"
						options={modalities.map((m) => ({ value: m, label: m }))}
						value={selectedModality}
						onChange={setSelectedModality}
					/>
				)}

				<div className="flex flex-1 items-center gap-6 sm:max-w-[320px]">
					<div className="flex flex-1 items-center gap-2">
						<Sun className="h-3.5 w-3.5 text-muted-foreground" />
						<Slider
							aria-label="Brightness"
							value={[brightness]}
							onValueChange={([v]) => setBrightness(v)}
							min={-100}
							max={100}
							step={1}
							className="flex-1"
						/>
					</div>
					<div className="flex flex-1 items-center gap-2">
						<Contrast className="h-3.5 w-3.5 text-muted-foreground" />
						<Slider
							aria-label="Contrast"
							value={[contrast]}
							onValueChange={([v]) => setContrast(v)}
							min={0.1}
							max={2.5}
							step={0.1}
							className="flex-1"
						/>
					</div>
				</div>
			</div>
		</div>
	);
};
