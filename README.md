# Bike-sharing demand

This project predicts hourly bike rentals, then classifies whether an hour is unusually busy. All five phases are in [bike_sharing_demand.ipynb](bike_sharing_demand.ipynb).

## Files and how to run

| Location | Purpose |
|---|---|
| `bike_sharing_demand.ipynb` | Main code, explanations, plots and results |
| `project_report.pdf` | Six-page report to submit |
| `sample_submission.csv` | Final count predictions to submit |
| `data/` | The three original supplied CSVs; do not edit them |
| `artifacts/phase_one.json` … `phase_five.json` | Saved models, settings and phase handoffs |
| `artifacts/bonus_asymmetric.json` | Bonus model, optimizer settings and cost comparison |
| `docs/requirements_checklist.md` | PDF requirement audit and remaining team responsibilities |
| `requirements.txt` | Required Python packages |
| `.vscode/` | Optional project interpreter setting |
| `.venv/` | Local Python environment; do not submit or commit it |

Create the environment once, from the project folder in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` exists, skip its creation. In VS Code, select `.venv\Scripts\python.exe` as both the Python interpreter and notebook kernel. Restart the kernel and run **all cells from the top**. Each phase needs the earlier phase's variables and saved result.

Command-line execution:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 bike_sharing_demand.ipynb
```

This regenerates the five phase artifacts, bonus artifact, report and root submission CSV. The original template in `data/sample_submission.csv` stays unchanged. Save any notebook edits before running this command; reopen the tab afterward if VS Code is showing an older version.

## Short glossary

| Term | Simple meaning |
|---|---|
| Feature / weight | An input / its learned coefficient |
| Intercept | The model's starting level before input effects |
| Training / validation | Rows used to learn / held-out rows used to compare settings |
| Scaling | Put inputs on similar numerical ranges using training statistics |
| RMSE | Typical error size in rentals/hour; lower is better |
| R² | How much target variation the predictions explain; higher is better |
| Learning rate | Size of an optimizer's update |
| Degree | Highest power added to a polynomial model |
| Bias / variance | Missing useful patterns / fitting patterns that do not hold on new data |
| Regularization | A penalty that discourages large coefficients |
| Ablation | Remove an input group, refit and compare the error |
| KKT check | A numerical test that the penalized solution is close to its optimum |
| Precision / recall | How many alerts are correct / how many true high-demand hours are found |
| F1 / ROC-AUC | Balance precision and recall / measure probability ranking |

## Phase One

**Goal:** create the required split and fit a linear model with our own gradient descent.

**Code walkthrough**

1. Import libraries. Sort IDs `16003585`, `16003110`, `14004179` as strings, join with underscores, and apply SHA-256 modulo 100000. The sorted string is `14004179_16003110_16003585`; the seed is **29578**. The brief's example seed is also checked.
2. Read the CSVs and check IDs, missing values, duplicates, schema and timeline gaps. A missing hour is not assumed to have zero demand.
3. Make exactly one `train_test_split(test_size=0.20, random_state=seed)`: **8,708 training / 2,178 validation rows**. Keep those IDs for every later phase.
4. Encode hour, season, weather, weekday and month as one-hot columns. Keep binary calendar and continuous weather inputs. Extract day-of-month from `dteday`; year/month are already available. Drop the record ID. Fit means/scales on training only, then add the intercept.
5. Predict with `X @ weights` and minimize half-MSE. The gradient is `X.T @ (X @ weights - y) / m`. Our function supports a chosen learning rate and initial weights.
6. Compare stable learning rates, plot loss, check the gradient numerically and compare training RMSE with a separate least-squares optimum. The reference calculation does not supply the reported weights.
7. Report training/validation R² and RMSE, compare with the training-mean baseline, inspect errors and save `artifacts/phase_one.json`.

**Why these choices?** Category codes are labels, not numerical distances. Scaling improves update stability. Weather codes 3/4 are pooled because code 4 has no training examples. A nearly repeated elapsed-date trend was removed after it caused a false stopping signal. We retain overlapping inputs for Phase Four to assess rather than claiming they are useless without evidence.

**Settings and result:** rates `0.1/L`, `0.5/L`, `1/L` plus excessive `2.2/L`; `L` is the largest training-Hessian eigenvalue. Stable rates are below `2/L`. Stop below relative loss change `1e-10` for 20 steps, cap 12,000. Selected rate **0.02763121**, **9,938 updates**. Validation **R² 0.69795 / RMSE 99.46491**.

**Output to Phase Two:** actual weights, their feature order, training settings and fitted preprocessing. **Be able to explain:** the seed, gradient update, why scaling is training-only, and how we checked convergence.

### Optional bonus (+5%): underprediction costs more

The brief says missing demand costs three times as much as overpredicting by the same amount. We use **weighted squared error**: for prediction error `e = prediction - actual`, use weight `a = 3` when `e < 0`, otherwise `a = 1`.

`Loss = sum(a * e²) / (2m)` and `gradient = X.T @ (a * e) / m`.

The derivative of each half-squared error is `3e` below zero and `e` above zero; both give zero at zero. Recalculate `a` after each update because an error can switch sides.

**What the code does:** start from Phase One's MSE weights and run our own full-batch gradient descent on the same linear features, training rows and training-only scales. Use rate `1/(3L)` because the new gradient's Lipschitz bound is `3L`. Stop at maximum absolute training gradient `1e-5`, with a 50,000-step cap. Finite differences check all coefficients. The model converged in **4,338 updates**. Neither validation stopping nor a library optimizer is used.

| Same validation rows, raw predictions | Original MSE model | Bonus 3:1 model |
|---|---:|---:|
| Asymmetric half-MSE (lower is better) | 10750.60 | 8595.38 |
| Mean predicted rentals/hour | 194.19 | 236.41 |
| Underprediction frequency | 44.67% | 30.76% |
| Missed rentals/hour, averaged over all rows | 37.28 | 21.30 |
| RMSE (rentals/hour) | 99.46 | 108.77 |
| R² | 0.69795 | 0.63881 |

**What it means:** mean predictions rise **42.22 rentals/hour** and the specified validation cost falls **20.05%**. Ordinary RMSE rises: the model now gives more importance to missed demand. The hourly plot shows where predictions move. This is an assumed squared-error cost, not measured money or a claim about every future date.

**Why keep the main model chain?** The required phases compare ordinary R²/RMSE and must pass actual weights forward. The bonus is a separate cost experiment, so Phase Two still uses the original Phase One MSE weights. Its coefficients, settings and measurements are saved in `artifacts/bonus_asymmetric.json`. The report includes the gradient derivation and comparison on page one. **Be able to explain:** why the gradient changes, why predictions rise on average, and why lower operator cost can come with higher RMSE.

## Phase Two

**Goal:** add curved weather effects while continuing Phase One's model.

**Code walkthrough**

1. Load Phase One's artifact and verify seed, IDs, feature order and hashes.
2. Keep its original columns. Add individual powers of temperature, humidity and wind speed: degree 2 adds squares; degree 3 also adds cubes. Do not add cross-column interactions.
3. Scale the added terms on training only. Copy the actual Phase One weights and give new terms zero weights. Assertions check that starting predictions match Phase One.
4. Train with the same gradient-descent function. Try the old rate plus `0.5/L` and `1/L` for each expanded matrix, skipping unstable rates. Check each fit against its training optimum.
5. Compare training/validation R² and RMSE, choose the lowest validation RMSE and save `artifacts/phase_two.json`.

**Why these choices?** Weather effects may curve. Expanding one-hot indicators would repeat them. `atemp` stays linear because it overlaps strongly with temperature (training correlation about 0.986). The limited basis keeps the first comparison understandable; it is not proof that interactions lack value.

**Settings and result:** tolerance `1e-12`, patience 20, cap 100,000. Degree **3**, rate **0.23337059**, **2,848 updates**. Validation **R² 0.70375 / RMSE 98.50538**. Improvement over Phase One is modest.

**Output to Phase Three:** selected degree, expanded terms and actual fitted values. **Be able to explain:** what a squared/cubic term changes and why copying weights plus zeros preserves the starting model.

## Phase Three

**Goal:** check underfitting/overfitting and whether random validation represents later periods.

**Code walkthrough**

1. Read Phase Two's selected degree **3** and keep its weather-power choice.
2. Compare degrees **1–5** on the unchanged seeded split using our gradient descent. Check optimization before interpreting model error.
3. Make **one** chronological split at the date containing the 80% row boundary. Keep complete dates together; fit all its preprocessing only on earlier rows. Do not use random-split weights that have seen later dates.
4. Compare training/validation R² and RMSE for both splits. Inspect the target mean/spread and mean predictions over later dates.
5. Save an evidence-based diagnosis and target degree in `artifacts/phase_three.json`.

| Degree | Random train RMSE | Random validation RMSE | Later-date train RMSE | Later-date validation RMSE |
|---|---:|---:|---:|---:|
| 1 | 100.2631 | 99.4655 | 91.8688 | 131.0797 |
| 2 | 99.8886 | 99.1821 | 91.4592 | 131.0799 |
| 3 | 99.1686 | 98.5054 | 90.7208 | 130.3631 |
| 4 | 98.9808 | 98.3458 | 90.3616 | 131.0821 |
| 5 | 98.9299 | 98.3803 | 90.3173 | 131.0440 |

**Why these choices?** Degrees 1–5 surround the previous degree rather than starting an unrelated model. The date-only cutoff avoids choosing a period because it gives a good score. It is **2012-08-05**, giving **8,703 earlier / 2,183 later rows**. Use rate `1/L`, tolerance `1e-12`, patience 20 and cap 500,000; all ten fits converge and pass the optimum check.

**Diagnosis:** degree 3 is reasonably fitted within this additive family on random validation; there is no large positive train/validation error gap. Degree 5 shows a small worsening after degree 4. The later-date weakness is much larger: degree-3 R² falls from **0.7038 to 0.6409**, while RMSE rises **31.86**. Different periods have different target means (**174.45 vs 259.85**), so the gap is not proof that overfitting alone caused it.

**Output to Phase Four:** target **degree 4**, because it wins required random validation, and the time-risk diagnosis. Later dates instead favor degree 3; we report that disagreement. Trust the chronological estimate more for later-period prediction, but one cutoff is not proof of all future performance. **Be able to explain:** why the two estimates differ and why a penalty may not solve a changed demand pattern.

## Phase Four

**Goal:** compare all three required penalties and select inputs for Phase Five.

**Code walkthrough**

1. Read Phase Three's degree **4** and diagnosis. Keep its 62 scaled predictors and exact split.
2. Implement a common loss: `half-MSE + lambda * ((1-rho)*sum(w²)/2 + rho*sum(abs(w)))`. Ridge uses `rho=0`; Lasso uses `rho=1`; Elastic Net mixes both. Never penalize the intercept.
3. Our accelerated proximal-gradient function takes a gradient step, then uses soft thresholding to make small L1 weights exactly zero. A KKT residual checks the penalized optimum; a separate Ridge solution checks the implementation.
4. Search lambda `0.0001, 0.001, 0.01, 0.1, 1, 10, 100`; Elastic Net also tests ratios `0.2, 0.5, 0.8`. These **35 fits** use identical rows, scales and checks. Select lowest validation RMSE per method.
5. Remove each original input's derived terms, refit at fixed winning settings, and measure the validation change. For the excluded ID, test adding it instead. Save every verdict with its numbers.
6. Save all winning models and the selected Lasso's exact nonzero mask in `artifacts/phase_four.json`.

| Method | Lambda | L1 ratio | Validation R² | Validation RMSE | Nonzero terms |
|---|---:|---:|---:|---:|---:|
| Unpenalized degree 4 | 0 | — | 0.704710 | 98.34580 | 62 |
| Lasso | 0.01 | 1.0 | 0.704721 | 98.34383 | 57 |
| Ridge | 0.0001 | 0.0 | 0.704707 | 98.34627 | 62 |
| Elastic Net | 0.0001 | 0.8 | 0.704709 | 98.34589 | 62 |

**Why these choices?** Phase Three showed only mild random-split overfitting, so the grid includes near-zero and weak penalties as well as strong ones. Use optimality tolerance `1e-7*(1+max(abs(b)))`, cap 100,000; a flat loss curve alone is not enough. Lasso's RMSE improvement is only **0.00197**: its main benefit is sparsity, not a meaningful accuracy gain. Strong penalties hurt. The time gap remains unproved to be fixed.

**Column-verdict rule:** removal cost above **0.1 rentals/hour** means useful. Otherwise call it redundant if other inputs reconstruct it with validation R² at least **0.90**, or uninformative in this tested model otherwise. These chosen sensitivity limits are not statistical significance tests. For `instant`, the delta is the gain from adding it. `dteday` means incremental day-of-month; year/month remain separately available.

| Input | Verdict | RMSE delta | Reconstruction R² | Lasso terms |
|---|---|---:|---:|---:|
| `instant` | redundant | +0.00912 | 0.99999 | 0 |
| `dteday` | uninformative | +0.08919 | 0.02710 | 1 |
| `season` | redundant | -0.00179 | 1.00000 | 3 |
| `yr` | useful | +9.43777 | 0.04640 | 1 |
| `mnth` | useful | +1.19885 | 0.35750 | 8 |
| `hr` | useful | +45.72496 | 0.01519 | 23 |
| `holiday` | redundant | -0.00521 | 1.00000 | 1 |
| `weekday` | useful | +0.11684 | 0.14247 | 5 |
| `workingday` | redundant | -0.00143 | 1.00000 | 1 |
| `weathersit` | useful | +1.17817 | 0.15457 | 2 |
| `temp` | useful | +1.08211 | 0.61497 | 4 |
| `atemp` | redundant | +0.04207 | 0.97150 | 1 |
| `hum` | useful | +0.39033 | 0.29536 | 3 |
| `windspeed` | useful | +0.11358 | 0.10468 | 4 |
| `cnt` | Target only | — | — | Never an input |

Day-of-month's 0.0892 effect is just below the chosen 0.1 limit; this does not mean dates contain no information. Weekday and wind speed are just above it. These are conditional labels, not universal truths.

**Output to Phase Five:** **57 exact Lasso survivors**, plus the feature order and preprocessing. The five zero terms are `weekday_2`, `mnth_6`, `mnth_7`, `mnth_12`, `hum^2`. Do not remove every redundant group at once: groups can replace each other after refitting. The Lasso mask and group-removal tests answer different questions. **Be able to explain:** how each penalty works, what lambda/rho mean, and why tiny score differences should not be oversold.

## Phase Five

**Goal:** classify unusually high demand using only the 57 selected inputs, then complete the final deliverables.

**Code walkthrough**

1. Read Phase Four's exact ordered Lasso mask. No dropped input is added back.
2. Define high demand as `cnt > training 75th percentile for (hr, workingday)`. Groups need 30 training rows; otherwise use training hour/global fallbacks. Apply fixed training cutoffs to validation; do not fit validation quantiles.
3. Fit logistic regression with our stable sigmoid/cross-entropy and damped Newton updates. The intercept starts at training log-odds and is unpenalized. Count-model weights have different units, so classifier slopes start at zero.
4. Check the gradient numerically. Require loss to decrease and maximum gradient below `1e-7`, with cap 100. Test L2 lambda `0, 0.0001, 0.001, 0.01, 0.1`; select by validation ROC-AUC.
5. Show probability threshold 0.5, then search **0.05–0.95** in 0.01 steps for maximum validation F1. Ties favor recall, then closeness to 0.5. Compare with a majority-class baseline.
6. Save the classifier, cutoffs, threshold, metrics and retrospective in `artifacts/phase_five.json`.
7. For the count CSV, refit the selected **Lasso** on all **10,886 labeled rows** using its chosen settings and existing preprocessing. The brief allows this. Do not report validation performance for this refitted model. Clip impossible negative counts at zero without rounding, and write **574** rows in test order to root `sample_submission.csv`.
8. Generate `project_report.pdf` with six pages from the run's results.

**Why these choices?** A global count cutoff would mostly separate daytime peaks from quiet nights. The contextual 75th percentile flags unusually busy hours among comparable hours; it is not an absolute capacity rule. The 75th percentile and 30-row minimum are practical choices fixed before fitting, not guarantees of best policy. Training positives are **24.63%**, validation **25.57%**.

| Validation model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.7443 | 0 | 0 | 0 | 0.5000 |
| Logistic, threshold 0.50 | 0.8811 | 0.8028 | 0.7092 | 0.7531 | 0.9160 |
| Logistic, threshold 0.44 | 0.8788 | 0.7708 | 0.7487 | 0.7596 | 0.9160 |

Selected classifier lambda is **0**; more L2 shrinkage is not supported by this search. Threshold **0.44** improves F1 and recall, with a small loss of precision/accuracy. It produces **1,497 true negatives, 124 false alerts, 140 misses and 417 true positives**. These are tuned validation scores, not independent future-period results.

The count-regression scores are a different task. The original split's raw Lasso RMSE is **98.34383**; clipping its negative predictions gives **95.92731**. Those scores belong to the training-only model, not the final full-data refit. The source template in `data/` stays unchanged.

**Be able to explain:** the difference between a count cutoff and a probability threshold, why accuracy alone is insufficient, which features enter the classifier, and why final count predictions do not come from classifier probabilities.

## Final checks and honest limits

See [the requirement checklist](docs/requirements_checklist.md) for the final file and code audit. Required submission files are the notebook, six-page PDF and root CSV. No external data or hidden labels are used. There is one seeded split and one chronological split. All five phases have an initial Expectation, actual numeric outputs and a final Outcome.

The optional **+5% asymmetric-cost bonus is implemented and checked** in Phase One. Team registration, correct submission, each member's live explanation, and hidden-test performance cannot be guaranteed by a file audit. Every member must be able to explain **all phases**, even if one person mainly worked on a particular phase. The simplified wording is a study guide; review it and make sure it matches your own understanding. No one can promise a full mark from these files alone.
