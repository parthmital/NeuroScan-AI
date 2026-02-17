import { useEffect, useRef, useState } from "react";
import * as cornerstone from "@cornerstonejs/core";
import * as cornerstoneTools from "@cornerstonejs/tools";
import {
	cornerstoneNiftiImageLoader,
	init as initNiftiLoader,
	createNiftiImageIdsAndCacheMetadata,
} from "@cornerstonejs/nifti-volume-loader";

let isCornerstoneInitialized = false;

const initializeCornerstone = async () => {
	if (isCornerstoneInitialized) return;

	await cornerstone.init();
	await cornerstoneTools.init();

	// Initialize and register the NIFTI image loader
	initNiftiLoader();
	cornerstone.imageLoader.registerImageLoader(
		"nifti",
		cornerstoneNiftiImageLoader,
	);

	// add tools
	cornerstoneTools.addTool(cornerstoneTools.WindowLevelTool);
	cornerstoneTools.addTool(cornerstoneTools.ZoomTool);
	cornerstoneTools.addTool(cornerstoneTools.PanTool);
	cornerstoneTools.addTool(cornerstoneTools.TrackballRotateTool);

	isCornerstoneInitialized = true;
};

interface QuadViewerProps {
	scanId: string;
	modality?: string;
	overlays?: {
		wt: boolean;
		tc: boolean;
		et: boolean;
		opacity: number;
	};
	overlayColors?: {
		wt: string;
		tc: string;
		et: string;
	};
}

// Helper to parse hex to 0-255 array
const getRgbArray = (hex: string): [number, number, number, number] => {
	const bigint = parseInt(hex.slice(1), 16);
	const r = (bigint >> 16) & 255;
	const g = (bigint >> 8) & 255;
	const b = bigint & 255;
	return [r, g, b, 255]; // Alpha will be overridden by opacity
};

export const QuadViewer = ({
	scanId,
	modality = "flair",
	overlays,
	overlayColors,
}: QuadViewerProps) => {
	const axialRef = useRef<HTMLDivElement>(null);
	const sagittalRef = useRef<HTMLDivElement>(null);
	const coronalRef = useRef<HTMLDivElement>(null);
	const volume3DRef = useRef<HTMLDivElement>(null);

	const renderingEngineId = useRef("quadViewEngine");
	const [isReady, setIsReady] = useState(false);
	const [isLoading, setIsLoading] = useState(false);
	const [segLoaded, setSegLoaded] = useState(false);

	// Effect to update segmentation Visibility/Opacity/Colors when controls change
	useEffect(() => {
		// Only run if segmentation is successfully loaded and we have overlays props
		if (!segLoaded || !overlays || !overlayColors) return;

		const updateAppearance = () => {
			const segmentationId = `SEG-${scanId}`;
			const toolGroupIds = [
				`QuadToolGroup-${scanId}-2D`,
				`QuadToolGroup-${scanId}-3D`,
			];

			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const colorConfig = (cornerstoneTools.segmentation as any).config.color;
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const visibilityConfig = (cornerstoneTools.segmentation as any).config
				.visibility;

			// Define colors
			const wtColor = getRgbArray(overlayColors.wt);
			const tcColor = getRgbArray(overlayColors.tc);
			const etColor = getRgbArray(overlayColors.et);

			// Update opacity for all
			const opacity = Math.round(overlays.opacity * 255);

			// Base colors from props
			const baseWtColor = getRgbArray(overlayColors.wt);
			const baseTcColor = getRgbArray(overlayColors.tc);
			const baseEtColor = getRgbArray(overlayColors.et);

			// Logic to mimic SliceViewer hierarchy:
			// Label 1 = Edema (Visible if WT is ON). Color = WT
			const l1_visible = overlays.wt;
			// Clone to avoid mutation issues
			const l1_color = [...baseWtColor];
			l1_color[3] = l1_visible ? opacity : 0;

			// Label 2 = Core (Visible if TC is ON, OR if WT is ON)
			// Color: If TC on -> TC color. Else -> WT color.
			const l2_visible = overlays.tc || overlays.wt;
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const l2_color: any = overlays.tc ? [...baseTcColor] : [...baseWtColor];
			l2_color[3] = l2_visible ? opacity : 0;

			// Label 3 = Enhancing (Visible if ET is ON, OR TC is ON, OR WT is ON)
			// Color: ET > TC > WT
			const l3_visible = overlays.et || overlays.tc || overlays.wt;
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const l3_color: any = overlays.et
				? [...baseEtColor]
				: overlays.tc
					? [...baseTcColor]
					: [...baseWtColor];
			l3_color[3] = l3_visible ? opacity : 0;

			const segments = [
				{ index: 1, visible: l1_visible, color: l1_color },
				{ index: 2, visible: l2_visible, color: l2_color },
				{ index: 3, visible: l3_visible, color: l3_color },
			];

			const renderingEngine = cornerstone.getRenderingEngine(
				renderingEngineId.current,
			);

			toolGroupIds.forEach((tgId) => {
				const toolGroup = cornerstoneTools.ToolGroupManager.getToolGroup(tgId);
				if (!toolGroup) return;

				// Update global segmentation representation config usage
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
				const segConfig = (cornerstoneTools.segmentation as any).config;
				let segRepConfig = null;

				if (segConfig.getSegmentationRepresentationSpecificConfig) {
					segRepConfig = segConfig.getSegmentationRepresentationSpecificConfig(
						tgId,
						{ segmentationId },
					);
				}

				// If no specific config exists yet, create one based on defaults
				if (!segRepConfig) {
					segRepConfig = {
						renderFill: true,
						renderOutline: true,
						renderInactiveSegmentations: true,
						representations: {
							Labelmap: {
								renderFill: true,
								renderOutline: true,
								outlineWidthActive: 2,
								outlineWidthInactive: 2,
								fillAlpha: overlays.opacity,
								outlineOpacity: overlays.opacity,
							},
						},
					};
				}

				if (segRepConfig) {
					// Update the properties directly on the object as well for safety
					segRepConfig.renderFill = true;
					segRepConfig.renderOutline = true;
					segRepConfig.outlineWidthActive = 2;
					segRepConfig.outlineWidthInactive = 2;
					segRepConfig.fillAlpha = overlays.opacity;
					segRepConfig.outlineOpacity = overlays.opacity; // Sync outline opacity

					if (segConfig.setSegmentationRepresentationSpecificConfig) {
						segConfig.setSegmentationRepresentationSpecificConfig(
							tgId,
							{ segmentationId },
							segRepConfig,
						);
					}
				}

				const viewportIds = toolGroup.getViewportIds();
				viewportIds.forEach((vpId) => {
					// Set visibility per segment
					segments.forEach((seg) => {
						try {
							// Set visibility
							visibilityConfig.setSegmentIndexVisibility(
								vpId,
								segmentationId,
								seg.index,
								seg.visible,
							);
							// Set Color (including opacity)
							colorConfig.setSegmentIndexColor(
								vpId,
								segmentationId,
								seg.index,
								seg.color,
							);
						} catch (e) {
							// ignore
						}
					});
				});
			});

			// Specific fix for 3D Volume Viewport
			const viewport3D = renderingEngine?.getViewport("VOLUME_3D");
			if (viewport3D) {
				viewport3D.render();
			}

			if (renderingEngine) {
				renderingEngine.render();
				// Delayed render to ensure 3D volume updates catch up
				setTimeout(() => {
					const vp3D = renderingEngine.getViewport("VOLUME_3D");
					if (vp3D) vp3D.render();
					renderingEngine.render();
				}, 50);
			}
		};

		updateAppearance();
	}, [overlays, overlayColors, segLoaded, scanId]);

	useEffect(() => {
		initializeCornerstone().then(() => setIsReady(true));

		const engineId = renderingEngineId.current;
		return () => {
			const engine = cornerstone.getRenderingEngine(engineId);
			if (engine) engine.destroy();
		};
	}, []);

	useEffect(() => {
		if (
			!isReady ||
			!axialRef.current ||
			!sagittalRef.current ||
			!coronalRef.current ||
			!volume3DRef.current ||
			!scanId
		)
			return;

		const engineId = renderingEngineId.current;
		const toolGroupId = `QuadToolGroup-${scanId}`;

		const loadVolumes = async () => {
			setIsLoading(true);
			setSegLoaded(false);
			try {
				const renderingEngine = new cornerstone.RenderingEngine(engineId);

				// Define viewports
				const viewportInputs = [
					{
						viewportId: "AXIAL",
						type: cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
						element: axialRef.current!,
						defaultOptions: {
							orientation: cornerstone.Enums.OrientationAxis.AXIAL,
							background: [0, 0, 0] as [number, number, number],
						},
					},
					{
						viewportId: "SAGITTAL",
						type: cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
						element: sagittalRef.current!,
						defaultOptions: {
							orientation: cornerstone.Enums.OrientationAxis.SAGITTAL,
							background: [0, 0, 0] as [number, number, number],
						},
					},
					{
						viewportId: "CORONAL",
						type: cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
						element: coronalRef.current!,
						defaultOptions: {
							orientation: cornerstone.Enums.OrientationAxis.CORONAL,
							background: [0, 0, 0] as [number, number, number],
						},
					},
					{
						viewportId: "VOLUME_3D",
						type: cornerstone.Enums.ViewportType.VOLUME_3D,
						element: volume3DRef.current!,
						defaultOptions: {
							background: [0, 0, 0] as [number, number, number],
						},
					},
				];

				// Enable all viewports
				viewportInputs.forEach((input) => {
					renderingEngine.enableElement(input);
				});

				// Setup tools - separate groups for 2D and 3D
				const toolGroup2DId = `${toolGroupId}-2D`;
				const toolGroup3DId = `${toolGroupId}-3D`;

				// 2D Tool Group
				let toolGroup2D =
					cornerstoneTools.ToolGroupManager.getToolGroup(toolGroup2DId);
				if (!toolGroup2D) {
					toolGroup2D =
						cornerstoneTools.ToolGroupManager.createToolGroup(toolGroup2DId)!;

					toolGroup2D.addTool(cornerstoneTools.WindowLevelTool.toolName);
					toolGroup2D.addTool(cornerstoneTools.ZoomTool.toolName);
					toolGroup2D.addTool(cornerstoneTools.PanTool.toolName);

					// Set active tools for 2D
					toolGroup2D.setToolActive(cornerstoneTools.WindowLevelTool.toolName, {
						bindings: [
							{ mouseButton: cornerstoneTools.Enums.MouseBindings.Primary },
						],
					});
					toolGroup2D.setToolActive(cornerstoneTools.ZoomTool.toolName, {
						bindings: [
							{ mouseButton: cornerstoneTools.Enums.MouseBindings.Secondary },
						],
					});
					toolGroup2D.setToolActive(cornerstoneTools.PanTool.toolName, {
						bindings: [
							{ mouseButton: cornerstoneTools.Enums.MouseBindings.Auxiliary },
						],
					});
				}

				// 3D Tool Group
				let toolGroup3D =
					cornerstoneTools.ToolGroupManager.getToolGroup(toolGroup3DId);
				if (!toolGroup3D) {
					toolGroup3D =
						cornerstoneTools.ToolGroupManager.createToolGroup(toolGroup3DId)!;

					toolGroup3D.addTool(cornerstoneTools.TrackballRotateTool.toolName);
					toolGroup3D.addTool(cornerstoneTools.ZoomTool.toolName);
					toolGroup3D.addTool(cornerstoneTools.PanTool.toolName);

					// Set active tools for 3D
					toolGroup3D.setToolActive(
						cornerstoneTools.TrackballRotateTool.toolName,
						{
							bindings: [
								{ mouseButton: cornerstoneTools.Enums.MouseBindings.Primary },
							],
						},
					);
					toolGroup3D.setToolActive(cornerstoneTools.ZoomTool.toolName, {
						bindings: [
							{ mouseButton: cornerstoneTools.Enums.MouseBindings.Secondary },
						],
					});
					toolGroup3D.setToolActive(cornerstoneTools.PanTool.toolName, {
						bindings: [
							{ mouseButton: cornerstoneTools.Enums.MouseBindings.Auxiliary },
						],
					});
				}

				// Add 2D viewports to 2D tool group
				["AXIAL", "SAGITTAL", "CORONAL"].forEach((viewportId) => {
					toolGroup2D!.addViewport(viewportId, engineId);
				});

				// Add 3D viewport to 3D tool group
				toolGroup3D.addViewport("VOLUME_3D", engineId);

				// Load NIFTI volume
				const token = localStorage.getItem("token");
				const imageUrl = `http://localhost:8000/api/scans/${scanId}/download/${modality}?token=${token}`;
				const volumeId = `nifti:${imageUrl}`;

				const imageIds = await createNiftiImageIdsAndCacheMetadata({
					url: imageUrl,
				});

				const volume = await cornerstone.volumeLoader.createAndCacheVolume(
					volumeId,
					{
						imageIds,
					},
				);

				await volume.load();

				// Set volume for all viewports
				await cornerstone.setVolumesForViewports(
					renderingEngine,
					[{ volumeId }],
					["AXIAL", "SAGITTAL", "CORONAL", "VOLUME_3D"],
				);

				// Configure 3D volume rendering
				const viewport3D = renderingEngine.getViewport(
					"VOLUME_3D",
				) as cornerstone.Types.IVolumeViewport;
				if (viewport3D && viewport3D.setProperties) {
					// Get the volume actor
					const volumeActor = viewport3D.getDefaultActor();
					if (volumeActor && volumeActor.actor) {
						const volumeActorEntry = volumeActor.actor;

						// Set up volume rendering properties
						// eslint-disable-next-line @typescript-eslint/no-explicit-any
						const property = volumeActorEntry.getProperty() as any;

						// Enable shading for better 3D appearance
						property.setShade(true);
						property.setAmbient(0.2);
						property.setDiffuse(0.7);
						property.setSpecular(0.3);
						property.setSpecularPower(8.0);

						// Set up opacity and color transfer functions for MRI
						const rgbTransferFunction = property.getRGBTransferFunction(0);
						const opacityTransferFunction = property.getScalarOpacity(0);

						// Clear existing points
						rgbTransferFunction.removeAllPoints();
						opacityTransferFunction.removeAllPoints();

						// MRI Brain preset - adjust these values based on your data
						// These are typical values for T1/T2/FLAIR MRI
						rgbTransferFunction.addRGBPoint(0, 0.0, 0.0, 0.0);
						rgbTransferFunction.addRGBPoint(100, 0.3, 0.3, 0.3);
						rgbTransferFunction.addRGBPoint(200, 0.6, 0.6, 0.6);
						rgbTransferFunction.addRGBPoint(500, 1.0, 1.0, 1.0);

						// Opacity transfer function - make lower values more transparent
						opacityTransferFunction.addPoint(0, 0.0);
						opacityTransferFunction.addPoint(50, 0.0);
						opacityTransferFunction.addPoint(100, 0.1);
						opacityTransferFunction.addPoint(200, 0.3);
						opacityTransferFunction.addPoint(500, 0.8);
						opacityTransferFunction.addPoint(1000, 1.0);

						// Set interpolation type
						property.setInterpolationTypeToLinear();
					}
				}

				renderingEngine.render();

				// Load and display segmentation if available
				try {
					const segUrl = `http://localhost:8000/api/scans/${scanId}/download/segmentation?token=${token}`;
					const segVolumeId = `nifti:seg_${scanId}`;
					const segmentationId = `SEG-${scanId}`;

					console.log("Loading segmentation from:", segUrl);

					const segImageIds = await createNiftiImageIdsAndCacheMetadata({
						url: segUrl,
					});

					console.log("Segmentation image IDs created:", segImageIds.length);

					const segVolume = await cornerstone.volumeLoader.createAndCacheVolume(
						segVolumeId,
						{
							imageIds: segImageIds,
						},
					);

					await segVolume.load();
					console.log("Segmentation volume loaded");

					// FORCE FRAME OF REFERENCE MATCH
					// This is critical if the processed NIfTI lost its original FrameOfReferenceUID
					// or if the loader generated a new one.
					const mainVolume = cornerstone.cache.getVolume(volumeId);
					if (mainVolume && segVolume) {
						const mainFoR = mainVolume.metadata.FrameOfReferenceUID;
						const segFoR = segVolume.metadata.FrameOfReferenceUID;

						console.log(`Main Volume FoR: ${mainFoR}`);
						console.log(`Segmentation FoR: ${segFoR}`);

						if (mainFoR && segFoR && mainFoR !== segFoR) {
							console.warn(
								"Frame of Reference mismatch! Forcing segmentation to match main volume.",
							);
							segVolume.metadata.FrameOfReferenceUID = mainFoR;
						}
					}

					// Add segmentation to Cornerstone
					await cornerstoneTools.segmentation.addSegmentations([
						{
							segmentationId: segmentationId,
							representation: {
								type: cornerstoneTools.Enums.SegmentationRepresentations
									.Labelmap,
								data: {
									volumeId: segVolumeId,
								},
							},
						},
					]);

					console.log("Segmentation added to state with ID:", segmentationId);

					// Add segmentation to 2D viewports
					const viewports2D = toolGroup2D?.getViewportIds() || [];
					for (const viewportId of viewports2D) {
						await cornerstoneTools.segmentation.addSegmentationRepresentations(
							viewportId,
							[
								{
									segmentationId: segmentationId,
									type: cornerstoneTools.Enums.SegmentationRepresentations
										.Labelmap,
								},
							],
						);
					}

					// Set active for 2D ToolGroup
					cornerstoneTools.segmentation.activeSegmentation.setActiveSegmentation(
						toolGroup2DId,
						segmentationId,
					);

					// Add segmentation to 3D viewport
					const viewports3D = toolGroup3D?.getViewportIds() || [];
					for (const viewportId of viewports3D) {
						await cornerstoneTools.segmentation.addSegmentationRepresentations(
							viewportId,
							[
								{
									segmentationId: segmentationId,
									type: cornerstoneTools.Enums.SegmentationRepresentations
										.Labelmap,
								},
							],
						);
					}

					// Set active for 3D ToolGroup
					cornerstoneTools.segmentation.activeSegmentation.setActiveSegmentation(
						toolGroup3DId,
						segmentationId,
					);

					// Force initial 3D config to ensure visibility
					try {
						const initial3DConfig = {
							renderFill: true,
							renderOutline: true,
							fillAlpha: 0.7,
							outlineOpacity: 1.0,
							representations: {
								Labelmap: {
									renderFill: true,
									renderOutline: true,
									fillAlpha: 0.7,
									outlineOpacity: 1.0,
								},
							},
						};
						// eslint-disable-next-line @typescript-eslint/no-explicit-any
						const segConfig = (cornerstoneTools.segmentation as any).config;
						if (segConfig.setSegmentationRepresentationSpecificConfig) {
							segConfig.setSegmentationRepresentationSpecificConfig(
								toolGroup3DId,
								{ segmentationId },
								initial3DConfig,
							);
						}
					} catch (e) {
						console.warn("Could not set initial 3D config", e);
					}

					// Mark segmentation as loaded, which triggers the effect to apply colors/visibility
					setSegLoaded(true);

					console.log("Segmentation loading flow completed");
				} catch (segErr) {
					console.error("Segmentation failed to load:", segErr);
					console.warn("Continuing without segmentation overlay");
				}

				setIsLoading(false);
			} catch (e) {
				console.error("Failed to load volumes", e);
				setIsLoading(false);
			}
		};

		loadVolumes();

		return () => {
			try {
				cornerstoneTools.ToolGroupManager.destroyToolGroup(`${toolGroupId}-2D`);
				cornerstoneTools.ToolGroupManager.destroyToolGroup(`${toolGroupId}-3D`);
			} catch (e) {
				console.warn("Cleanup error", e);
			}
		};
	}, [isReady, scanId, modality]); // Removed overlays from here to prevent full reload

	return (
		<div className="w-full h-full grid grid-cols-2 grid-rows-2 gap-1 bg-black p-1 rounded-lg relative min-h-[600px]">
			{/* Axial View */}
			<div className="relative bg-black rounded overflow-hidden border border-border/50">
				<div
					ref={axialRef}
					className="w-full h-full absolute inset-0"
					style={{ zIndex: 1 }}
				/>
				<div
					className="absolute top-2 left-2 text-xs font-bold text-red-500 bg-black/80 px-2 py-1 rounded backdrop-blur-sm border border-red-500/30 pointer-events-none"
					style={{ zIndex: 999 }}
				>
					AXIAL
				</div>
			</div>

			{/* Sagittal View */}
			<div className="relative bg-black rounded overflow-hidden border border-border/50">
				<div
					ref={sagittalRef}
					className="w-full h-full absolute inset-0"
					style={{ zIndex: 1 }}
				/>
				<div
					className="absolute top-2 left-2 text-xs font-bold text-green-500 bg-black/80 px-2 py-1 rounded backdrop-blur-sm border border-green-500/30 pointer-events-none"
					style={{ zIndex: 999 }}
				>
					SAGITTAL
				</div>
			</div>

			{/* Coronal View */}
			<div className="relative bg-black rounded overflow-hidden border border-border/50">
				<div
					ref={coronalRef}
					className="w-full h-full absolute inset-0"
					style={{ zIndex: 1 }}
				/>
				<div
					className="absolute top-2 left-2 text-xs font-bold text-blue-500 bg-black/80 px-2 py-1 rounded backdrop-blur-sm border border-blue-500/30 pointer-events-none"
					style={{ zIndex: 999 }}
				>
					CORONAL
				</div>
			</div>

			{/* 3D Volume */}
			<div className="relative bg-black rounded overflow-hidden border border-border/50">
				<div
					ref={volume3DRef}
					className="w-full h-full absolute inset-0"
					style={{ zIndex: 1 }}
				/>
				<div
					className="absolute top-2 left-2 text-xs font-bold text-purple-500 bg-black/80 px-2 py-1 rounded backdrop-blur-sm border border-purple-500/30 pointer-events-none"
					style={{ zIndex: 999 }}
				>
					VOLUME 3D
				</div>
			</div>

			{/* Loading Overlay */}
			{(isLoading || !isReady) && (
				<div
					className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm rounded-lg"
					style={{ zIndex: 9999 }}
				>
					<div className="flex flex-col items-center gap-3">
						<div className="w-10 h-10 border-2 border-medical border-t-transparent rounded-full animate-spin" />
						<p className="text-sm text-medical font-mono tracking-wider">
							{!isReady ? "Initializing 3D Engine..." : "Loading Volume..."}
						</p>
					</div>
				</div>
			)}
		</div>
	);
};
