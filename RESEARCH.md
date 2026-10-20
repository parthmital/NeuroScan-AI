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

## Target journals for the classification benchmark paper (2026-10-20)

**Brief.** Pick three journals for `Materials/Research Papers/Standardised_Comparison_of_Convolutional_and_Transformer_Backbones_for_Brain_Tumour_Image_Classification/main.tex`: a nine-backbone 2D MRI benchmark whose main contribution is the evaluation protocol (perceptual-hash deduplication, group-aware splits, calibration, Holm-corrected McNemar tests) and the head-masked shortcut audit.

**Scope.** Web searches for journal scope and metrics; Elsevier and Springer journal pages refused automated requests (HTTP 403 and a login redirect), so metrics come from third-party aggregators, which disagree. Searched 2026-10-20.

**Findings.**

- _Engineering Applications of Artificial Intelligence_ (Elsevier, IFAC journal): scope asks for novel AI aspects in a real-world engineering application, validated on public datasets for replicability. Aggregators list a 2025 impact factor between 8.0 and 9.0, Q1 ([Journal Metrics](https://www.journalmetrics.org/journal/engineering-applications-of-artificial-intelligence), [BioxBio](https://www.bioxbio.com/journal/ENG-APPL-ARTIF-INTEL)). The manuscript is already in its `cas-sc` template and its abstract states the contribution to AI and the engineering application.
- _Computers in Biology and Medicine_ (Elsevier): aggregators list a 2025 impact factor between 6.3 and 8.35, Q1, with roughly 4 months to acceptance ([Journal Metrics](https://www.journalmetrics.org/journal/computers-in-biology-and-medicine), [SciJournal](https://www.scijournal.org/impact-factor-of-comput-biol-med.shtml)). Uses the same Elsevier CAS template.
- _Journal of Imaging Informatics in Medicine_ (Springer, official SIIM journal, renamed from _Journal of Digital Imaging_): scope covers machine learning and AI in medical imaging informatics ([SIIM](https://siim.org/resources/publications/journal-of-imaging-informatics)). Reported impact factors range from 3.1 to 5.3 ([SciJournal](https://www.scijournal.org/journal-of-imaging-informatics-in-medicine), [WoS-journal.info](https://wos-journal.info/journalid/25296)). Needs reformatting to the Springer template.

**Recommendation.** Submit to _Engineering Applications of Artificial Intelligence_ first, then _Computers in Biology and Medicine_ (Elsevier transfer, no reformatting), then _Journal of Imaging Informatics in Medicine_. Confirm impact factors in Clarivate JCR before deciding on metrics alone.

## Literature table of the classification paper (2026-10-20)

**Brief.** Before reusing the classification paper's Table 1 (existing studies) in the Review 3 deck, check each row's model, dataset, task and accuracy against the cited article.

**Scope.** Abstracts from Europe PMC and the Semantic Scholar API by DOI, the publisher page of Seetha and Raja (2018), and a web search for Rehman et al. (2020). Publisher pages of Elsevier, Springer and IEEE refused automated requests. Searched 2026-10-20.

**Findings.**

- Noreen et al. (2020), IEEE Access, [doi:10.1109/ACCESS.2020.2978629](https://doi.org/10.1109/ACCESS.2020.2978629): Inception-v3 99.34% and DenseNet201 99.51% on the three-class figshare dataset. Table 1 lists it as DenseNet121, binary detection, 99.51%.
- Rehman et al. (2020), Circuits, Systems, and Signal Processing, [doi:10.1007/s00034-019-01246-3](https://doi.org/10.1007/s00034-019-01246-3): AlexNet, GoogLeNet and VGG16 on the three-class figshare dataset; fine-tuned VGG16 reached 98.69%. Table 1 lists it as InceptionV3, binary detection, 99.34%.
- Khan et al. (2020), Diagnostics, [doi:10.3390/diagnostics10080565](https://doi.org/10.3390/diagnostics10080565): VGG16 and VGG19 features with ELM-based selection on BraTS 2015, 2017 and 2018, accuracy 97.8%, 96.9% and 92.5%. Table 1 lists it as VGG16 on 3,064 images, binary detection, 98.69%.
- Talo et al. (2019), Cognitive Systems Research, [doi:10.1016/j.cogsys.2018.12.007](https://doi.org/10.1016/j.cogsys.2018.12.007): ResNet34, normal vs abnormal, 613 MR images, 100% five-fold accuracy. Table 1 lists it as ResNet50 on 253 images, 95.23%.
- Deepak and Ameer (2019), Computers in Biology and Medicine, [doi:10.1016/j.compbiomed.2019.103345](https://doi.org/10.1016/j.compbiomed.2019.103345): GoogLeNet features, patient-level five-fold cross-validation, mean accuracy 98%. Table 1 lists it as InceptionV3, 97.50%.
- Ghassemi et al. (2020), Biomedical Signal Processing and Control, [doi:10.1016/j.bspc.2019.101678](https://doi.org/10.1016/j.bspc.2019.101678): a six-layer CNN pretrained as a GAN discriminator, three-class figshare dataset, five-fold cross-validation; the abstract gives no accuracy figure. Table 1 lists it as EfficientNet, binary detection, 98.87%.
- Seetha and Raja (2018), [Biomedical and Pharmacology Journal 11(3)](https://biomedpharmajournal.org/vol11no3/brain-tumor-classification-using-convolutional-neural-networks/): ImageNet-pretrained CNN with the last layer trained, tumour vs non-tumour (Radiopaedia, BRATS 2015); 97.5% is the training accuracy.
- Cheng et al. (2015), PLoS ONE, [doi:10.1371/journal.pone.0140381](https://doi.org/10.1371/journal.pone.0140381): accuracy up to 91.28% (bag-of-words with ring partition); the paper rounds it to 90%. Zacharaki et al. (2009), [doi:10.1002/mrm.22147](https://doi.org/10.1002/mrm.22147): 85% for metastasis vs glioma and 88% for glioma grading, leave-one-out. Tummala et al. (2022) 98.7% and Asiri et al. (2024) 97% match Table 1, but Asiri et al. classify four classes including no tumour.

**Recommendation.** The deck (`Materials/Presentation.pptx`, slide 5) uses the verified figures above. Correct Table 1, the matching sentences of the literature review and Table 11 (comparison with existing literature) in the classification paper before submission: the 3,064-image figshare studies are subtype classification, not binary detection.
