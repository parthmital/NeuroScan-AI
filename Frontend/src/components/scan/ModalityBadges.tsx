export const ModalityBadges = ({ modalities }: { modalities: string[] }) => (
	<div className="flex gap-1">
		{modalities.map((modality) => (
			<span
				key={modality}
				className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground"
			>
				{modality}
			</span>
		))}
	</div>
);
