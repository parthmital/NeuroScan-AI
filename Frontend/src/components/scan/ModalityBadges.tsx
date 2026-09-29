export const ModalityBadges = ({ modalities }: { modalities: string[] }) => (
	<div className="flex flex-wrap gap-1">
		{modalities.map((modality) => (
			<span
				key={modality}
				className="rounded-[4px] border border-border bg-muted/60 px-1.5 py-px font-mono text-[10.5px] font-medium tracking-[0.02em] text-muted-foreground"
			>
				{modality}
			</span>
		))}
	</div>
);
