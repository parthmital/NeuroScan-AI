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

	isCornerstoneInitialized = true;
};

interface VolumeViewerProps {
	scanId: string;
	modality?: string;
}

export const VolumeViewer = ({
	scanId,
	modality = "flair",
}: VolumeViewerProps) => {
	const elementRef = useRef<HTMLDivElement>(null);
	const renderingEngineId = useRef("myRenderingEngine");
	const viewportId = "CT_VOLUME_VIEWPORT"; // constant string
	const [isReady, setIsReady] = useState(false);

	useEffect(() => {
		initializeCornerstone().then(() => setIsReady(true));

		const engineId = renderingEngineId.current;
		return () => {
			// Cleanup if needed on unmount (often handled by engine.destroy)
			const engine = cornerstone.getRenderingEngine(engineId);
			if (engine) engine.destroy();

			// Destroy toolgroup if we made one (we generated ID based on scanId below? No, used constant in previous step, need to fix)
		};
	}, []);

	useEffect(() => {
		if (!isReady || !elementRef.current || !scanId) return;

		const engineId = renderingEngineId.current;
		// precise toolgroup id
		const toolGroupId = `VolumeToolGroup-${scanId}-${Math.random().toString(36).substr(2, 9)}`;

		const loadVolume = async () => {
			const renderingEngine = new cornerstone.RenderingEngine(engineId);

			const viewportInput = {
				viewportId,
				type: cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
				element: elementRef.current!,
				defaultOptions: {
					orientation: cornerstone.Enums.OrientationAxis.AXIAL,
					background: [0, 0, 0] as [number, number, number],
				},
			};

			renderingEngine.enableElement(viewportInput);

			// Construct URLs
			const token = localStorage.getItem("token");
			const baseUrl = `http://localhost:8000/api/scans/${scanId}/download`;
			const imageUrl = `${baseUrl}/${modality}?token=${token}`;

			try {
				// Initialize ToolGroup
				const toolGroup =
					cornerstoneTools.ToolGroupManager.createToolGroup(toolGroupId)!;

				// Add tools to ToolGroup
				toolGroup.addTool(cornerstoneTools.WindowLevelTool.toolName);
				toolGroup.addTool(cornerstoneTools.ZoomTool.toolName);
				toolGroup.addTool(cornerstoneTools.PanTool.toolName);

				// Set Active/Passive tools
				toolGroup.setToolActive(cornerstoneTools.WindowLevelTool.toolName, {
					bindings: [
						{
							mouseButton: cornerstoneTools.Enums.MouseBindings.Primary, // Left Click
						},
					],
				});
				toolGroup.setToolActive(cornerstoneTools.ZoomTool.toolName, {
					bindings: [
						{
							mouseButton: cornerstoneTools.Enums.MouseBindings.Secondary, // Right Click
						},
					],
				});
				toolGroup.setToolActive(cornerstoneTools.PanTool.toolName, {
					bindings: [
						{
							mouseButton: cornerstoneTools.Enums.MouseBindings.Auxiliary, // Middle Click
						},
					],
				});

				// Add Viewport to ToolGroup
				toolGroup.addViewport(viewportId, engineId);

				// Create volume from NIFTI file
				const volumeId = `nifti:${imageUrl}`;
				const imageIds = await createNiftiImageIdsAndCacheMetadata({
					url: imageUrl,
				});

				// Create and cache the volume
				const volume = await cornerstone.volumeLoader.createAndCacheVolume(
					volumeId,
					{
						imageIds,
					},
				);

				// Set volume
				await volume.load();

				// Add volumes to viewport
				await cornerstone.setVolumesForViewports(
					renderingEngine,
					[{ volumeId }],
					[viewportId],
				);

				renderingEngine.render();
			} catch (e) {
				console.error("Failed to load volume", e);
			}
		};

		loadVolume();

		return () => {
			// Cleanup
			try {
				cornerstoneTools.ToolGroupManager.destroyToolGroup(toolGroupId);
				const engine = cornerstone.getRenderingEngine(engineId);
				if (engine) engine.destroy();
				cornerstone.cache.purgeCache();
			} catch (e) {
				console.warn("Cleanup error", e);
			}
		};
	}, [isReady, scanId, modality]);

	return (
		<div className="w-full h-full min-h-[500px] bg-black relative rounded-lg overflow-hidden">
			<div ref={elementRef} className="w-full h-full" />
			{!isReady && (
				<div className="absolute inset-0 flex items-center justify-center text-white bg-black/50">
					Initializing 3D Engine...
				</div>
			)}
		</div>
	);
};
