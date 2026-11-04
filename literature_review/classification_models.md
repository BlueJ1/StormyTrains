# Classification models beyond the baselines: literature survey

Written 31 October 2026 for the `severity` task of `frankfurt_munich/modeling/` (`severity_class`: under 20 min, 20–59 min, 60+ min, cancelled; multi-class log loss for choosing models, macro-F1 reported as well, over the 5 week folds). It is the companion of [regression_models.md](regression_models.md) and, like it, a focused survey, not a systematic one. Claims marked *(abstract)* were checked only against the abstract or a summary page; the others against the full text in this folder, the paper's own web page or tables, or the documentation named.

## Starting point

- **Baselines (19 October):** class shares give log loss 0.968 (24 h) to 0.897 (0 min); logistic regression 0.927 to 0.778. Macro-F1 0.22 (24 h) to 0.36 (0 min). Weather changes log loss by at most 0.0015.
- **The classes are imbalanced:** in the CV set (17,507 journeys) 62.8% are under 20 min, 25.5% 20–59 min, 6.2% 60+ min and 5.5% cancelled.
- **The classes are not fully ordered:** the three delay classes are, but a cancellation is not "more delayed" than 60+ min, so ordinal models do not fit directly.
- **Probabilities are the output.** Models are chosen by log loss, which judges the whole predicted distribution over the four classes, not only the most probable class.

## What train-delay studies use for classes

| Study | Setting | Target | Models compared | Finding relevant to us |
|---|---|---|---|---|
| Jiang et al. 2019 (in this folder) | Sweden, passenger and freight trains, 2017 train / 2018 test | Binary: arrival at the destination more than 5 min 59 s late, as a probability | Linear regression, Weibull, logistic regression, random forest, and random forest probabilities fed into a logistic regression | Random forest alone did not beat logistic regression (RMSE of the probability 0.06 vs 0.05); the combination was best |
| Bao et al. 2026 (preprint) | Netherlands, trajectories | Binary: significantly delayed | Logistic regression, random forest, XGBoost, LightGBM; naive persistence | Random forest best when months are mixed (balanced accuracy 0.72); LightGBM ≈ XGBoost on later months (0.60). Weather helped within months but generally hurt on later months. No log loss or calibration reported |
| Brakenhoff et al. 2026 | Netherlands, trajectories | Binary: cancelled | Dynamic graph neural network vs baselines including XGBoost | The GNN and XGBoost both clearly beat the other baselines; F1 at most 0.4. Operational features more useful than weather *(abstract)* |
| Laifa et al. 2022 | Tunisia, 12,350 journeys | 13 delay classes of 5 min, as the first stage of a regression | LightGBM classifier only (benchmarks compared on the regression) | A delay-class classifier as a building block; no classification metric reported |

The regression studies in [regression_models.md](regression_models.md) (Li et al. 2021, Oneto et al. 2016, Nair et al. 2019 and others) use the same model families for minutes of delay.

**Summary.**

- **Supported:** the model families. Random forest, XGBoost and LightGBM are the standard step after linear and logistic models for train delays, and tree ensembles also match or beat neural networks on mid-sized tabular classification (Grinsztajn et al. 2022).
- **Not supported by a direct study:**
  - no study found compares random forest and boosting on several delay classes;
  - none treats cancellation as one class among delay classes;
  - none of the train-delay studies found scores probabilities by log loss. They report accuracy, F1, balanced accuracy or AUC, so the literature supports the models' rankings, not the quality of their probabilities.
- **Weather:** as for regression, mixed. Bao et al. and Brakenhoff et al. find weather less useful than operational features, especially on later periods.

## Probabilities and calibration

Because log loss scores the probabilities, how well each model's probabilities are calibrated matters as much as its ranking.

- **Random forests avoid 0 and 1.** Averaging many trees pulls probabilities that should be near 0 or 1 towards the middle (Niculescu-Mizil & Caruana 2005, as summarised in the scikit-learn calibration guide, which shows peaks near 0.2 and 0.9 and a sigmoid-shaped calibration curve). Exact zeros, which would make log loss very large, are therefore unlikely; too cautious probabilities are the expected problem. Jiang et al.'s result fits this.
- **Boosted trees:** the same paper found boosted trees poorly calibrated, but with AdaBoost-style boosting. XGBoost and LightGBM with the multi-class objective minimise log loss directly, so they may be better calibrated; this is an assumption that we did not find tested for this setting and should check with calibration curves.
- **Logistic regression** is likely to be well calibrated by itself, since its loss is log loss with the canonical link (scikit-learn calibration guide). This is part of why it is a strong baseline here.
- **SVMs** do not give probabilities. scikit-learn's `SVC(probability=True)` fits Platt scaling by an extra 5-fold cross-validation inside each fit, described as expensive for large data, and the probabilities may disagree with `predict` (scikit-learn SVM guide).
- **Fixing calibration:** Platt scaling or isotonic regression (`CalibratedClassifierCV`) after the model. Fitting the calibration needs data not used to fit the model, which inside our 5 folds means a further split of each training fold.

## Suggested models

These follow stage 2 of the roadmap in `ProjectInstructions.md` ("Random Forest; XGBoost or LightGBM") and the regression models already tuned for the `delay` task.

1. **LightGBM, multi-class.** Minimises log loss directly; native categorical features and missing `current_delay`, as in the delay model. Tied for best on delay. LightGBM appears as a classifier in Laifa et al. and Bao et al.
2. **XGBoost, multi-class.** Very close to LightGBM; best at 24 h and 6 h with weather on delay; a strong baseline in Bao et al. and Brakenhoff et al. Worth keeping as a cross-check.
3. **Random forest classifier.** The most common model in the train-delay literature and best in one of Bao et al.'s settings. Its probabilities are likely too cautious; whether that costs log loss is something to measure, with calibration as the fix if it does.

**Setup, as for delay:** one model per prediction time, railway-only and with weather on the same rows and folds, the same flat random search of 50 configurations chosen by mean log loss, macro-F1 reported as well. Multi-class boosters fit one tree per class per round, so they take roughly four times as long as the delay boosters.

**Class weights:** not recommended for log loss. Weighting the rare classes up shifts every predicted probability towards them, which log loss on the true class shares penalises. Macro-F1, which uses only the most probable class, would rise. This is our reasoning, not a finding of the studies above.

## Not recommended now

- **SVM (SVC):** no native probabilities; Platt scaling costs 5 extra fits per fit; the weakest model on delay.
- **Ordinal logistic regression:** cancellation breaks the ordering of the classes.
- **Neural networks:** stage 3 of the roadmap; Grinsztajn et al. suggest little gain on data like ours.

## Possible extension: two-stage model

One classifier for P(cancelled), and for completed journeys the delay classes from a delay model: either a 3-class classifier or a predicted delay distribution (quantile regression forest or NGBoost, see [regression_models.md](regression_models.md)), from which P(delay ≥ 20) and P(delay ≥ 60) follow. The four class probabilities are then P(cancelled) and (1 − P(cancelled)) times each delay-class probability. It links the classification and the probabilistic delay models, and Laifa et al. and Jiang et al. show that combined models can beat single ones, but it is more work and no study found uses this exact split.

## References

- Bao, J. L., Alsahag, A. M. M., Mohammadi Ziabari, S. S. (2026). Predicting delayed train trajectories on the Dutch railway network: explainable AI evaluation of topological, operational and weather features with tree based ensemble methods. arXiv:2609.34692.
- Brakenhoff, B., Alsahag, A. M. M., Mohammadi Ziabari, S. S. (2026). Dynamic GNNs for predicting train cancellations on the Dutch railway network: a multi-season study of environmental and operational factors. *Digital Technologies Research and Applications*, 5(1). https://doi.org/10.54963/dtra.v5i1.1709
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35*, 507–520.
- Jiang, S., Persson, C., Åkesson, J. (2019). Punctuality prediction: combined probability approach and random forest modelling with railway delay statistics in Sweden. *IEEE ITSC*, 2797–2802. https://doi.org/10.1109/ITSC.2019.8916892
- Laifa, H., Khcherif, R., Ben Ghezala, H. (2022). Predicting trains delays using a two-level machine learning approach. *ICAART 2022*, vol. 3, 737–744.
- Niculescu-Mizil, A., Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632. https://doi.org/10.1145/1102351.1102430
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 61–74.
- scikit-learn documentation: [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) and [Support vector machines, scores and probabilities](https://scikit-learn.org/stable/modules/svm.html), checked 31 October 2026.
