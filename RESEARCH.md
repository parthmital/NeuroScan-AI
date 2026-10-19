# Research Notes

## References for the backbone benchmark paper (2026-10-17)

**Brief.** Find peer-reviewed sources for every architectural, methodological and background claim in `Materials/Research Papers/Standardised_Comparison_of_Convolutional_and_Transformer_Backbones_for_Brain_Tumour_Image_Classification/main.tex`, and check whether its two preprint citations have peer-reviewed versions.

**Scope.** Crossref works API (by title and by DOI), publisher proceedings pages (NeurIPS, PMLR, JMLR), ML Anthology pages for ICLR papers (OpenReview blocks automated requests), and a web search for the Vangala et al. audit. Searched 2026-10-17.

**Findings.**

- 34 references were added, each checked against Crossref or the publisher's page for authors, venue, volume, pages and DOI or URL. ICLR, NeurIPS and AAAI entries have no page numbers in their official records, so BibTeX warns about empty pages for 8 entries; no pages were invented.
- Simonyan and Zisserman (VGG) was published at ICLR 2015 ([ML Anthology](https://mlanthology.org/iclr/2015/simonyan2015iclr-very/), [Oxford VGG page](https://www.robots.ox.ac.uk/~vgg/research/very_deep/)); the arXiv entry was replaced.
- Vangala et al., "Scores that hold, benchmarks that leak" (arXiv 2610.00421), was found only as an arXiv preprint and in secondary summaries; no peer-reviewed version exists as of the search date. _Superseded later the same day:_ the citation was removed. Its test-set duplicate percentages were dropped; the leakage context now cites Fateh et al., "BRISC: Annotated dataset for brain tumor segmentation and classification", Scientific Data 13, 361 (2026), [doi:10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y), which curates the Cheng, SARTAJ (Bhuvaji) and Br35H aggregate, removes exact and near-duplicates before splitting, and notes that the source release has no patient, sequence or slice information (verified in the article text). The NCC definition cites Zitova and Flusser, Image and Vision Computing 21(11), 977-1000 (2003), [doi:10.1016/S0262-8856(03)00137-9](https://doi.org/10.1016/S0262-8856%2803%2900137-9).
- Gneiting and Raftery, JASA 102(477), 359-378 (2007), [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437), supports the proper-scoring-rule statement in the classification paper's training paragraph.
- In the segmentation paper, TransUNet now cites its peer-reviewed version, Chen et al., Medical Image Analysis 97, 103280 (2024), [doi:10.1016/j.media.2024.103280](https://doi.org/10.1016/j.media.2024.103280). Bakas et al. (2018, arXiv 1811.02629) has no peer-reviewed version and stays because the BraTS data-use terms ask for it.
- Holm (1979) has no DOI; its JSTOR stable URL is <https://www.jstor.org/stable/4615733>.
- Official NeurIPS BibTeX records for Vaswani et al. (2017) and Paszke et al. (2019) leave `pages` empty, and ICLR has no page numbers, so no pages were added for these entries.

**Recommendation.** Keep the paper free of preprints; recheck Bakas et al. (2018) for a peer-reviewed version before submission of the segmentation paper.

## Tectonic BibTeX rerun warning (2026-10-17)

**Brief.** Every paper build with Tectonic 0.15.0 printed "internal consistency problem when checking if main.bbl changed" and stopped after six TeX passes.

**Findings.** The warning reproduces with a minimal natbib document, so it is not caused by the papers. It is Tectonic issue [#1150](https://github.com/tectonic-typesetting/tectonic/issues/1150) (duplicate of #1146), fixed by pull request #1155, which makes BibTeX close its log files on success. Tectonic 0.17.0 (released 2026-07-27; Windows MSVC zip SHA-256 `f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`, matching the release digest) builds the same documents with two TeX passes and no warning.

**Recommendation.** Use Tectonic 0.17.0 or later for `npm run papers`.

## xr cross-references on Overleaf (2026-10-17)

Overleaf's guide [Cross referencing with the xr package in Overleaf](https://www.overleaf.com/learn/how-to/Cross_referencing_with_the_xr_package_in_Overleaf) explains that the external document must be compiled first so its `.aux` exists, and provides a `latexmkrc` (by John Collins) that rebuilds external documents automatically.

**Sources.** [Crossref API](https://api.crossref.org/works); [NeurIPS proceedings](https://proceedings.neurips.cc/); [PMLR](https://proceedings.mlr.press/); [JMLR](https://jmlr.org/papers/v12/pedregosa11a.html); [ML Anthology](https://mlanthology.org/).
