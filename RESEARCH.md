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

_Applied 2026-10-21:_ the paper's literature review, Table 1 and Table 11, and the project report's Section 2.1 and Table 2.1 (`Materials/Report Template.md`), now use these figures. A second check the same day (Semantic Scholar and Europe PMC abstracts by DOI, the Seetha and Raja publisher page) confirmed every finding above. Ghassemi et al. (2020) gives no accuracy in its abstract and a web search found none attributable to it, so the paper cites it for its method only and drops it from both tables. Khan et al. (2020) stays in the text only, because its BraTS dataset sizes were not verified.

## Introduction statistics and cost ratio of the classification paper (2026-10-21)

**Brief.** Check the CBTRUS figures in the paper's Introduction, and the claim that model cost spans two orders of magnitude.

**Findings.**

- Ostrom et al. (2023), CBTRUS 2016-2020 report, [PMC10550277](https://pmc.ncbi.nlm.nih.gov/articles/PMC10550277/): "Counts and rates from the 453,623 brain and other CNS tumors", Figure 14 caption "Annual Average Cases=90,725", and the abstract "Gliomas accounted for 26.3% of all tumors". The paper's figures match.
- From the notebooks' `model_profiles.csv`, GFLOPs range from 0.431 (MobileNetV3-Large) to 30.932 (VGG16), a factor of 72, and parameters from 4.01 M to 134.27 M, a factor of 33. Neither is two orders of magnitude (100), so the paper, report and deck now give the factor instead.

## Segmentation notebook redesign: hybrid ensemble and external validation (2026-10-22)

**Brief.** The segmentation results (best mean Dice 0.8199, UNet) were weaker than GCSER-UNet (Mukherjee et al., [arXiv:2605.30510](https://arxiv.org/abs/2605.30510), submitted 2026-05-28), which reports BraTS 2020 Dice of 0.95 (WT), 0.92 (TC) and 0.90 (ET), and the method had no novel contribution. Constraints: one Kaggle session on 2x T4 (about 12 hours), fully rerunnable, no BraTS 2021+ pretrained weights (BraTS 2021 contains the BraTS 2020 cases).

**Findings.**

- GCSER-UNet is a 2D model. From the PDF (Sections III-B1, IV-B, IV-D2): it keeps "slices containing at least one positive pixel" per region, splits the processed data 75:15:10 without stating a patient-level split, and does not state how Dice is aggregated. Its numbers are therefore slice-wise on tumour-containing slices and not comparable with per-patient 3D BraTS Dice. The notebook now reports both protocols for every model.
- The previous pipeline resampled each 240x240x155 volume to 128^3 and trained 50 epochs with flips and intensity jitter only. Native-resolution patches, nnU-Net-style augmentation and sliding-window inference are the standard remedies (nnU-Net, Isensee et al., Nature Methods 18, 203-211 (2021), [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)).
- The earlier run showed complementary behaviour: transformers had higher sensitivity (Swin UNETR WT 0.9619 against UNet 0.9059) and CNNs higher Dice (`segmentation/metrics/test_results.csv`), which motivates region-wise fusion of a CNN and a transformer.
- External data on Kaggle (Kaggle dataset API, checked 2026-10-22): `aiocta/brats2023-ssa-training-dataset` holds the official `ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2` folder with 60 cases of five NIfTI files each (`-t1c`, `-t1n`, `-t2f`, `-t2w`, `-seg`; ET is label 3). Other BraTS-Africa uploads exist (for example `aondonamoses/brats-africa`, `zenjapainai/brats-africa`) with re-organised folders. BraTS 2021 copies (`dschettler8845/brats-2021-task1`) overlap BraTS 2020 and were rejected as an external set.

**Decision.** Adapt MONAI 1.6.1 (Apache-2.0, released 2026-09-27 on PyPI; requires torch >= 2.8, which Kaggle's 2.11 satisfies) for the networks (DynUNet as nnU-Net, SegResNetDS, SwinUNETR-v2) and sliding-window inference, installed with `--no-deps` so Kaggle's PyTorch is untouched; einops (MIT) is installed if missing because SwinUNETR imports it. The fusion, the quality-control analysis and Seg-Grad-CAM (Vinogradova et al., AAAI 2020) are written in the notebook. BraTS-Africa is the external test set; the internal validation and test patients are fixed to those of the earlier benchmark.

**Alternatives considered.** Keeping the hand-written networks (more code to maintain, no pretrained-free advantage); MedNeXt from MONAI as a third CNN (dropped to fit the session); BraTS 2021 new cases as a second external set (overlap with BraTS 2020 must be removed by image matching, and about 880 cases would not fit the session with TTA); `monai.visualize.GradCAM` (built for classification targets, so the segmentation target is implemented directly).

**Not verified.** Run time, memory and all results of the new notebook; they exist only after a Kaggle run.
