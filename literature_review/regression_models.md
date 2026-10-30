# Regression models beyond the baselines: literature survey

Written 26 October 2026 for the `delay` task of `frankfurt_munich/modeling/` (final arrival delay in minutes, completed journeys, RMSE over the 5 week folds). It complements the reading guide in [README.md](README.md) and is a focused survey, not a systematic one. Claims marked *(abstract)* were checked only against the abstract or a summary; the others against the full text in this folder or the paper's tables.

## Starting point

- **Baselines (19 October):** training mean RMSE 25.4 min; linear regression 24.6 (24 h, 6 h) to 22.0 (0 min). Weather changes RMSE by less than 0.03 min, possibly because a linear model can use each weather value only as a straight line.
- **The target is skewed:** in the CV set (16,551 completed journeys) the mean is 18.4 min, the median 10, the 99th percentile 117 and the maximum 279; 12.9% arrive early (down to −57). Skewness 2.6. This matters for both model choice and the probabilistic models: RMSE is dominated by the large delays, and a symmetric (normal) error distribution fits badly.
- **Size:** about 16.5k training rows per prediction time with roughly 10 railway features (+14 weather). Small and tabular, which favours tree ensembles over neural networks (Grinsztajn et al. 2022).

## What train-delay studies use

| Study | Setting | Models compared | Finding relevant to us |
|---|---|---|---|
| Nair et al. 2019 | Deutsche Bahn, whole network, real-time | Random forest (with weather, traffic, work zones) + kernel regression + mesoscopic simulation, as an ensemble | Only DB study found. RF is the statistical workhorse; weather is one input among many, its separate effect is not isolated *(abstract)* |
| Li et al. 2021 (in this folder) | Netherlands, delay 20 min ahead | Random forest vs ANN, XGBoost, GBDT, statistical models | Random forest best; current delay dominates |
| Oneto et al. 2016 (in this folder); 2018 | Italy (RFI), regression per checkpoint | Extreme learning machines, kernel regularised least squares, random forest | Adding weather improved accuracy over railway-only |
| Wang & Zhang 2019 | China, Beijing–Guangzhou, 3 months | Gradient-boosted regression trees with weather | Under severe weather the type of weather drives delay; otherwise historical delay does |
| Marković et al. 2015 | Serbia, arrival delay | Support vector regression vs ANN | SVR better *(abstract)* |
| Kecman & Goverde 2015 | Netherlands, event times | Robust linear regression, regression trees, random forests | Running and dwell times; early example of trees beside linear models *(abstract)* |
| Al Ghamdi et al. 2024 | UK | Heterogeneous ensembles (stacking) vs RF, XGBoost | Stacking several model types beat single RF/XGBoost *(abstract)* |
| Spanninger et al. 2024 (in this folder) | Switzerland, final arrival delay, dynamic horizon | NN with Gaussian output + test-time dropout; **NGBoost** (normal, log-normal); Markov chains; Bayesian network; median and current-delay benchmarks | NN best (MAE 46 s), NGBoost close (+4–6% MAE/RMSE). **Adding weather did not improve** the NN. Proper evaluation of predicted distributions (LoR, conformal intervals) |
| Stratil-Sauer & Breyer 2025 (preprint) | Sweden, journey reliability | Bayesian regression with a **mixture of two log-normals**, mean and spread depending on covariates | The two-component mixture fits delays much better than one log-normal |
| Corman & Kecman 2018 | Sweden, real-time | Bayesian network | Delay as a distribution updated along the journey |
| Borin et al. 2026 (preprint) | Finland, Oulu | XGBoost with three weather encodings | Weather categories (e.g. blizzard) beat raw values; no railway-only baseline, so it does not answer our question |

**Summary.** For tabular railway data the literature's standard step after linear models is a tree ensemble: random forest or gradient boosting (XGBoost, LightGBM, GBDT). Neither wins consistently; random forest often does as well as boosting on small data with few features. Evidence on weather is mixed: studies that report a gain (Oneto, Wang & Zhang) do not compare with strong railway-only models on the same split, while Spanninger et al. found no gain. A small or zero gain, as in our linear models, is therefore plausible and, per `ProjectInstructions.md`, a useful finding.

## Suggested point-prediction models

These follow stage 2 of the roadmap in `ProjectInstructions.md` ("Random Forest; XGBoost or LightGBM").

1. **Random forest** (`sklearn.ensemble.RandomForestRegressor`). The most common model in the train-delay literature, robust with few hyperparameters. Needs `current_delay` imputed (scikit-learn's forests accept NaN since 1.4, so it can stay missing as the STATUS decision for tree models says).
2. **Gradient-boosted trees.** Either `sklearn.ensemble.HistGradientBoostingRegressor` (already installed; LightGBM's algorithm; native missing values and categorical features) or LightGBM itself (new dependency, a few more options). Both handle the weather features non-linearly, which is the point of the weather comparison.

Both fit the existing pipeline unchanged: a factory in `models.py` and `run_cv("x", "delay", make_x)` with and without `weather=True`. For a fair weather comparison, the two feature sets should get the same tuning procedure (or none), so a gain is not just extra tuning.

Not recommended now: SVR and kernel methods (slow at 16k rows, few advantages over trees), neural networks (stage 3, and Grinsztajn et al. suggest little gain on data like ours), stacking (only after single models are understood).

## Suggested probabilistic models

A predicted distribution of the delay, rather than one number, matches the passenger question ("how likely is a serious delay?"): P(delay ≥ 20) and P(delay ≥ 60) come straight from it. Two candidates that complement each other:

### 1. NGBoost (Duan et al. 2020): parametric

Gradient boosting that predicts the parameters of a chosen distribution (for example mean and standard deviation of a normal), trained with a proper scoring rule. It is the probabilistic tree model used for train delays by Spanninger et al. 2024, where it was within about 5% of their best model.

- Python package `ngboost`, scikit-learn API; trees as base learners, so it also uses weather non-linearly.
- Distributions: normal, log-normal, Laplace, Student-t and others. The log-normal needs a positive target, so delay would be shifted (for example by +60 min); a normal fits our skewed delays badly. Choosing the distribution is a decision for us.
- Weakness: one parametric family per model; slower than LightGBM.
- Related: **LightGBMLSS / XGBoostLSS** (März 2019; März & Kneib 2022) do the same on LightGBM/XGBoost and support mixture distributions, which would allow a two-component model like Stratil-Sauer & Breyer's or our own `arrival_delays/` two-exponential fit.

### 2. Quantile regression forest (Meinshausen 2006): non-parametric

A random forest that keeps all training targets in each leaf, so for a new journey it returns the whole conditional distribution (any quantile, any exceedance probability), with no assumption about its shape.

- Python package `quantile-forest` (scikit-learn compatible); the same trees as the random forest above, so it costs little extra once the forest exists.
- Handles skew, early arrivals and the long tail as they are in the data.
- Weakness: cannot predict beyond the delays seen in training; no density, so no log-likelihood (use CRPS or pinball loss instead).

### Evaluating them

The agreed protocol scores the delay task by RMSE; the mean (or for QRF the forest's mean) of the predicted distribution can be scored the same way, so the probabilistic models stay comparable with the others. Additional scores would be needed for the distribution itself. Common choices (Gneiting & Raftery 2007; Spanninger et al. 2024):

- **CRPS** (continuous ranked probability score): proper, in minutes, works for both models;
- **pinball (quantile) loss** at chosen quantiles;
- **interval coverage and width** (does the 80% interval contain 80% of delays?) and a PIT histogram for calibration;
- the derived **P(delay ≥ 20) and P(delay ≥ 60)**, scored by log loss or Brier score. These cover completed journeys only; combining them with a cancellation probability would give the four severity classes.

## References

- Al Ghamdi, M., Parr, G., Wang, W. (2024). Heterogeneous machine learning ensembles for predicting train delays. *IEEE Transactions on Intelligent Transportation Systems*, 25, 5138–5153. https://doi.org/10.1109/TITS.2023.3337858
- Borin, V. P., de Souza Sant'Ana, J. M., Mahmood, N. H. (2026). Predicting train delays in Finland using machine learning and weather data. arXiv:2609.11277.
- Corman, F., Kecman, P. (2018). Stochastic prediction of train delays in real-time using Bayesian networks. *Transportation Research Part C*, 95, 599–615. https://doi.org/10.1016/j.trc.2018.08.003
- Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A., Schuler, A. (2020). NGBoost: Natural gradient boosting for probabilistic prediction. *Proceedings of ICML*, PMLR 119.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35*, 507–520.
- Kecman, P., Goverde, R. M. P. (2015). Online data-driven adaptive prediction of train event times. *IEEE Transactions on Intelligent Transportation Systems*, 16, 465–474. https://doi.org/10.1109/TITS.2014.2347136
- Li, Z., Wen, C., Hu, R., Xu, C., Huang, P., Jiang, X. (2021). Near-term train delay prediction in the Dutch railways network. *International Journal of Rail Transportation*, 9(6), 520–539.
- Marković, N., Milinković, S., Tikhonov, K. S., Schonfeld, P. (2015). Analyzing passenger train arrival delays with support vector regression. *Transportation Research Part C*, 56, 251–262. https://doi.org/10.1016/j.trc.2015.04.004
- März, A. (2019). XGBoostLSS: An extension of XGBoost to probabilistic forecasting. arXiv:1907.03178.
- März, A., Kneib, T. (2022). Distributional gradient boosting machines. arXiv:2204.00778.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- Nair, R., Hoang, T. L., Laumanns, M., Chen, B., Cogill, R., Szabó, J., Walter, T. (2019). An ensemble prediction model for train delays. *Transportation Research Part C*, 104, 196–209. https://doi.org/10.1016/j.trc.2019.04.026
- Oneto, L. et al. (2016). Advanced analytics for train delay prediction systems by including exogenous weather data. *IEEE DSAA*, 458–467. Oneto, L. et al. (2018). Train delay prediction systems: a big data analytics perspective. *Big Data Research*, 11, 54–64. https://doi.org/10.1016/j.bdr.2017.05.002
- Spanninger, T., Wiedemann, N., Corman, F. (2024). Quantifying the dynamic predictability of train delay with uncertainty-aware neural networks. *Transportation Research Part C*, 162, 104563.
- Stratil-Sauer, N., Breyer, N. (2025). Probabilistic modeling of delays for train journeys with transfers. arXiv:2504.17479.
- Wang, P., Zhang, Q. (2019). Train delay analysis and prediction based on big data fusion. *Transportation Safety and Environment*, 1(1), 79–88. https://doi.org/10.1093/tse/tdy001
