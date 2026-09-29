# Literature review reading guide

Eight distinct papers, two for each topic in **Tasks for the first two weeks** in [ProjectInstructions.md](../ProjectInstructions.md). Selected and checked on 19 September 2026 for the ICE/IC journey-disruption project.

The selection combines a broad review, practical railway studies, and evaluation guidance. Papers often span several topics; their folder indicates the main reason to read them. The project-specific suggestions below are reading recommendations, not claims that the papers used your exact prediction setting.

## 1. Prediction targets

### Spanninger, Trivella, Büchel & Corman (2022)
**A review of train delay prediction approaches**  
Journal of Rail Transport Planning & Management, 22, 100312.  
[Read PDF](01_prediction_targets/01_Spanninger_2022_Review_of_Train_Delay_Prediction.pdf) · [Publication](https://doi.org/10.1016/j.jrtpm.2022.100312) · 17 pages

**Why this paper:** Gives the vocabulary for distinguishing arrival/departure delay, prediction horizons, and deterministic versus probabilistic outputs. It also compares applicability under ordinary delays and disruptions, making it a useful starting point for all four topics.

**Read for:** A table of target, prediction unit, information available at prediction time, and forecast horizon. Use these distinctions to define your final-arrival target and before-departure/during-travel settings.

### Jiang, Persson & Åkesson (2019)
**Punctuality prediction: combined probability approach and random forest modelling with railway delay statistics in Sweden**  
2019 IEEE Intelligent Transportation Systems Conference, pp. 2797–2802.  
[Read PDF](01_prediction_targets/02_Jiang_2019_Punctuality_Prediction.pdf) · [Publication](https://doi.org/10.1109/ITSC.2019.8916892) · 6 pages

**Why this paper:** Provides a concrete example of predicting destination punctuality as a probability, using a combination of logistic regression and random forests. Its 2017 training / 2018 testing setup is also relevant to temporal evaluation.

**Read for:** How continuous delay is converted into a binary outcome and how an earlier delay informs the final-stop outcome. Its roughly six-minute punctuality threshold is different from your proposed 60-minute severe-delay threshold; it does not establish a cancellation model.

## 2. Common inputs

### Oneto et al. (2016)
**Advanced Analytics for Train Delay Prediction Systems by Including Exogenous Weather Data**  
2016 IEEE International Conference on Data Science and Advanced Analytics, pp. 458–467.  
[Read PDF](02_common_inputs/01_Oneto_2016_Exogenous_Weather_Data.pdf) · [Publication](https://doi.org/10.1109/DSAA.2016.57) · 10 pages

**Why this paper:** Closely matches your railway-only versus railway-plus-weather research question. It combines historical train movements with weather information in an Italian railway case study and compares several regression methods.

**Read for:** Weather variables, spatial matching to railway checkpoints, forecast versus observed weather, and the design of the weather comparison. For your experiment, retain the same journeys and temporal split, and allow only weather available at the prediction cutoff. Its reported benefit is not a guaranteed benefit for German ICE/IC journeys.

### Huang et al. (2020)
**Modeling train operation as sequences: A study of delay prediction with operation and weather data**  
Transportation Research Part E: Logistics and Transportation Review, 141, 102022.  
[Read PDF](02_common_inputs/02_Huang_2020_Operation_and_Weather_Sequences.pdf) · [Publication](https://doi.org/10.1016/j.tre.2020.102022) · 23 pages

**Why this paper:** Complements Oneto with explicit infrastructure, timetable, weather, and train-interaction inputs. Its sequence model and feature sensitivity analysis help explain why neighbouring trains and earlier stops can matter.

**Read for:** Section 2.2's input definitions and the feature/component analysis in Section 5. Separate variables you can reconstruct from Bahn-Vorhersage from infrastructure and operational variables that may be unavailable. The Chinese high-speed-rail setting and real-time inputs limit direct transfer to before-departure prediction.

## 3. Common models

### Li et al. (2021; first published online 2020)
**Near-term train delay prediction in the Dutch railways network**  
International Journal of Rail Transportation, 9(6), 520–539.  
[Read PDF](03_common_models/01_Li_2021_Near_Term_Dutch_Railway_Delays.pdf) · [Publication](https://doi.org/10.1080/23248378.2020.1843194) · 21 PDF pages, including publisher cover

**Why this paper:** A practical comparison centred on random forests, with neural-network, XGBoost, gradient-boosting and statistical benchmarks. It is useful for designing the standard-ML stage of your project and understanding feature importance.

**Read for:** How current delay and operating conditions predict delay about 20 minutes ahead. Compare that task with final-arrival prediction. Critically assess the held-out-Mondays test design and random validation sampling; use a chronological journey-level split for your own deployment-style evaluation.

### Arthaud, Lecoeur & Pierre (2024)
**Transformers à Grande Vitesse: Massively parallel real-time predictions of train delay propagation**  
Journal of Rail Transport Planning & Management, 29, 100418.  
[Read PDF](03_common_models/02_Arthaud_2024_Transformers_a_Grande_Vitesse.pdf) · [Journal publication](https://doi.org/10.1016/j.jrtpm.2023.100418) · [Author manuscript](https://arxiv.org/abs/2105.08526) · 10 pages

**Why this paper:** Provides an advanced-model contrast to the tabular baselines: a transformer approach to network-scale delay propagation on the French railway network. It helps explain the motivation for modelling interactions between trains.

**Read for:** Model inputs, embeddings, treatment of network interactions, and comparison against operational predictions. Treat this as an optional extension after the core pipeline works; network-wide real-time information is a substantially richer setting than one or two corridors.

**Version:** The supplied PDF is arXiv v2, dated 21 December 2023, associated with the 2024 journal publication. It is an author manuscript, not the publisher-formatted journal PDF.

## 4. Evaluation methods

### Tiong Kah Yong, Ma & Palmqvist (2025)
**AP-GRIP evaluation framework for data-driven train delay prediction models: systematic literature review**  
European Transport Research Review, 17, article 13.  
[Read PDF](04_evaluation_methods/01_Tiong_2025_AP_GRIP_Evaluation_Framework.pdf) · [Publication](https://doi.org/10.1186/s12544-024-00704-7) · 21 pages

**Why this paper:** Directly addresses how to evaluate train-delay models beyond a single accuracy score. The framework covers accuracy, precision, generalisability, robustness, interpretability, and practicality across several analysis dimensions.

**Read for:** Sections 3–5. Use them to structure a comparison of overall performance, routes, time periods, train groups, and disruption conditions. Do not assume that every occurrence of “precision” in this framework means positive predictive value for your severe-disruption classifier.

### Spanninger, Wiedemann & Corman (2024)
**Quantifying the dynamic predictability of train delay with uncertainty-aware neural networks**  
Transportation Research Part C: Emerging Technologies, 162, 104563.  
[Read PDF](04_evaluation_methods/02_Spanninger_2024_Dynamic_Predictability_and_Uncertainty.pdf) · [Publication](https://doi.org/10.1016/j.trc.2024.104563) · [Authors' code](https://github.com/mie-lab/train_delay) · 28 pages

**Why this paper:** Connects uncertainty estimates with how predictability changes over the journey. It introduces a dynamic prediction-horizon framework and evaluates predictive distributions and intervals, rather than only point errors.

**Read for:** Prediction-interval coverage, likelihood-based evaluation, and performance as a function of time remaining. These are useful for the optional during-journey risk updates. Delay-distribution uncertainty should not be treated as an already-validated cancellation probability.

## Suggested reading order

1. **Spanninger 2022:** establish the terminology and map of the field.
2. **Jiang 2019:** define probability targets and thresholds.
3. **Oneto 2016:** plan the weather experiment.
4. **Li 2021:** establish the practical model comparison.
5. **AP-GRIP 2025:** decide how to judge model usefulness.
6. **Huang 2020:** deepen the input and interaction analysis.
7. **Spanninger 2024:** add uncertainty and prediction-horizon evaluation.
8. **Arthaud 2024:** explore the optional advanced-model extension.

For each paper, record: **target and unit; prediction cutoff/horizon; input groups; model and baseline; train/validation/test split; metrics; treatment of severe delays and cancellations; transfer limits for this project.**

## Project-specific gaps to retain in your review

These papers give a foundation for delay and punctuality prediction, but they do not collectively validate the exact target **severe final delay or cancellation on a German ICE/IC journey**. Cancellation labels, incomplete journeys, and the before-departure information boundary still need explicit treatment.

For your own comparison, report delay MAE/RMSE on completed journeys separately from severe-disruption classification results. Consider precision/recall, PR-AUC, and probability calibration/Brier score for the binary target. These are recommendations for your project, not a claim that all eight papers use these measures. Do not remove severe delays simply to improve average error.

## Files and provenance

- There are exactly **eight distinct full-text PDFs**, with two in each topic folder.
- Spanninger (2022) and the Arthaud author manuscript were freshly downloaded from the University of Twente repository and arXiv, respectively.
- The other six were copied from the existing `AlexMasterThesis/related-work/files` collection. Their originals were left unchanged. Publisher/repository downloads for AP-GRIP and Spanninger (2024) returned access/rate errors, so the existing full texts were retained.
- [sources.json](sources.json) records each source, acquisition method, version, DOI, page count and SHA-256 checksum. Publication metadata and relevant full-text content were checked; every PDF page was successfully parsed and all eight first pages were rendered for inspection.
- [references.bib](references.bib) contains the eight bibliography entries with updated local file paths. The AP-GRIP PDF displays its first author's full name as “Tiong Kah Yong”; the existing bibliography uses the author form “Tiong, Kah Yong”.

This is a focused starter reading list, not an exhaustive systematic literature search.
