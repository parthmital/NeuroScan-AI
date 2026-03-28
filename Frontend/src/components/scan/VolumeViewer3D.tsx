import { useRef, useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { RotateCcw } from "lucide-react";

interface Overlays {
	wt: boolean;
	tc: boolean;
	et: boolean;
	opacity: number;
}

interface VolumeViewer3DProps {
	scanId: string;
	modalities: string[];
	overlays?: Overlays;
	overlayColors?: {
		wt: string;
		tc: string;
		et: string;
	};
	hasSegmentation?: boolean;
}

function hexToRgb(hex: string) {
	const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result
		? {
				r: parseInt(result[1], 16) / 255,
				g: parseInt(result[2], 16) / 255,
				b: parseInt(result[3], 16) / 255,
			}
		: { r: 1, g: 1, b: 1 };
}

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

	// Exposed reset camera handle
	const resetCameraRef = useRef<(() => void) | null>(null);

	// Expose segmentation transfer functions and render window
	const segCtfunRef = useRef<any>(null);
	const segOfunRef = useRef<any>(null);
	const renderWindowRef = useRef<any>(null);

	const resetCamera = useCallback(() => {
		resetCameraRef.current?.();
	}, []);

	const updateSegmentationOverlays = useCallback(
		(currentOverlays: Overlays, currentColors: any) => {
			if (
				!segCtfunRef.current ||
				!segOfunRef.current ||
				!renderWindowRef.current
			)
				return;

			const ctfun = segCtfunRef.current;
			const ofun = segOfunRef.current;

			const wtColor = hexToRgb(currentColors.wt || "#eab308");
			const tcColor = hexToRgb(currentColors.tc || "#ef4444");
			const etColor = hexToRgb(currentColors.et || "#3b82f6");

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

				const volProp: any = volume.getProperty();
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

				// Apply initial segmentation overlay state so they are visible immediately
				if (overlays && overlayColors) {
					updateSegmentationOverlays(overlays, overlayColors);
				}

				const segProp: any = segVolume.getProperty();
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
					} catch (_) {}
				};

				// Handle resize
				const resizeObserver = new ResizeObserver(() => {
					try {
						grw.resize();
					} catch (_) {}
				});
				resizeObserver.observe(container);

				// Cleanup function
				cleanupRef.current = () => {
					resizeObserver.disconnect();
					resetCameraRef.current = null;
					segCtfunRef.current = null;
					segOfunRef.current = null;
					renderWindowRef.current = null;
					try {
						volume.delete();
					} catch (_) {}
					try {
						mapper.delete();
					} catch (_) {}
					try {
						ctfun.delete();
					} catch (_) {}
					try {
						ofun.delete();
					} catch (_) {}
					try {
						segVolume.delete();
					} catch (_) {}
					try {
						segMapper.delete();
					} catch (_) {}
					try {
						segCtfun.delete();
					} catch (_) {}
					try {
						segOfun.delete();
					} catch (_) {}
					try {
						grw.delete();
					} catch (_) {}
				};

				// -------------------------------------------------------
				// Fetch and load the NIfTI volume(s)
				// -------------------------------------------------------
				const token = localStorage.getItem("token");
				if (!token) throw new Error("User not authenticated.");

				const fetchPromises = [
					fetch(
						`http://localhost:8000/api/scans/${scanId}/download/${selectedModality.toLowerCase()}`,
						{ headers: { Authorization: `Bearer ${token}` } },
					),
				];

				if (hasSegmentation) {
					fetchPromises.push(
						fetch(
							`http://localhost:8000/api/scans/${scanId}/download/segmentation`,
							{ headers: { Authorization: `Bearer ${token}` } },
						),
					);
				}

				const responses = await Promise.all(fetchPromises);

				for (const res of responses) {
					if (!res.ok) {
						const text = await res.text().catch(() => "");
						throw new Error(
							`Failed to download MRI volume (${res.status})${text ? ": " + text : ""}`,
						);
					}
				}

				const buffers = await Promise.all(
					responses.map((res) => res.arrayBuffer()),
				);

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

					// Apply initial segmentation overlay state
					if (overlays && overlayColors) {
						updateSegmentationOverlays(overlays, overlayColors);
					}
				}

				// Brain-tissue pure grayscale transfer (matching 2D SliceViewer)
				ctfun.removeAllPoints();
				// Strict linear mapping from scalar value (0-255) to RGB (0-1)
				ctfun.addRGBPoint(0, 0, 0, 0); // Black
				ctfun.addRGBPoint(255, 1, 1, 1); // White

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
			} catch (err: any) {
				console.error("[VolumeViewer3D]", err);
				if (mounted) {
					setError(err?.message ?? "Unknown error loading 3D volume.");
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
	}, [scanId, selectedModality, hasSegmentation]);

	// Handle overlay reactivity dynamically
	useEffect(() => {
		if (overlays && overlayColors) {
			updateSegmentationOverlays(overlays, overlayColors);
		}
	}, [overlays, overlayColors, updateSegmentationOverlays]);

	return (
		<div className="space-y-3">
			{/* Canvas container — same aspect-square + border style as SliceViewer */}
			<div
				ref={containerRef}
				className="relative flex aspect-square items-center justify-center overflow-hidden rounded-xl border border-border bg-[#0d0d0d]"
			>
				{/* Loading overlay */}
				{isLoading && (
					<div className="absolute inset-0 z-10 flex items-center justify-center bg-black/40 backdrop-blur-sm">
						<div className="flex flex-col items-center gap-2">
							<div className="h-8 w-8 animate-spin rounded-full border-2 border-medical border-t-transparent" />
							<p className="font-mono text-[10px] uppercase tracking-widest text-medical">
								Loading 3D Volume
							</p>
						</div>
					</div>
				)}

				{/* Error overlay */}
				{error && !isLoading && (
					<div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/80 px-6 text-center">
						<p className="font-mono text-xs text-destructive">
							Failed to load 3D volume
						</p>
						<p className="font-mono text-[10px] text-muted-foreground">
							{error}
						</p>
					</div>
				)}

				{/* HUD overlay — only when loaded */}
				{!isLoading && !error && (
					<>
						<div className="pointer-events-none absolute left-4 top-4 z-10">
							<p className="font-mono text-[10px] text-medical/80">
								3D Volume Render
							</p>
							<p className="font-mono text-[10px] text-medical/80">
								Modality: {selectedModality}
							</p>
						</div>

						{/* Orientation hints */}
						<span className="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2 font-mono text-[13px] text-medical/50">
							S
						</span>
						<span className="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2 font-mono text-[13px] text-medical/50">
							I
						</span>

						{/* Reset camera */}
						<button
							onClick={resetCamera}
							title="Reset camera"
							className="absolute bottom-4 right-4 z-10 flex items-center gap-1 rounded-md border border-border bg-card/80 px-2 py-1 text-[10px] font-medium text-muted-foreground backdrop-blur-sm transition-colors hover:border-medical/50 hover:text-medical"
						>
							<RotateCcw className="h-3 w-3" />
							Reset
						</button>
					</>
				)}
			</div>

			{/* Modality selector — identical pill style to SliceViewer */}
			{modalities && modalities.length > 0 && (
				<div className="flex justify-center gap-2 pt-1">
					{modalities.map((m) => (
						<button
							key={m}
							onClick={() => setSelectedModality(m)}
							className={cn(
								"rounded-md border px-3 py-1.5 text-xs font-semibold transition-all",
								selectedModality === m
									? "border-medical bg-medical text-medical-foreground shadow-sm"
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
