# Bike-sharing demand
Machine Learning Project 1: a chained pipeline using the supplied hourly bike-sharing records. The current implementation covers **Phases One through Four**.

Open this folder in VS Code, then select `.venv\Scripts\python.exe` through
**Python: Select Interpreter**. In `bike_sharing_demand.ipynb`, use
**Select Kernel → Python Environments** to choose the same environment.
Restart the notebook kernel and run all cells in order from the project folder. The notebook reads the CSV files from `data/`, trains Phases One and Two, runs Phase Three's complexity and chronological comparisons, fits Phase Four's regularized models, and saves their handoff artifacts.

To set up a new environment from PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Skip environment creation if `.venv` already exists. To execute and save notebook outputs:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 bike_sharing_demand.ipynb
```

## Phase One

### Goal and expectation

Predict `cnt`, the number of rentals in one hour, using a linear model trained with our own full-batch gradient descent. The opening **Expectation** cell records the hypothesis before execution: calendar and weather should explain some demand, categorical hours should capture commuting peaks, and an excessive learning rate may diverge. The closing **Outcome** displays the actual run's results. Later phases will extend this baseline.

### Decisions and reasons

The assignment specifies the seed, split, from-scratch optimizer and evaluation metrics. Feature encoding, scaling, learning rates and stopping settings are engineering choices. The explanations below distinguish those choices from requirements and record the changes prompted by execution.

#### Why check the optimizer?

An optimizer is the procedure that changes coefficients to reduce training error. Here it is our `gradient_descent` function. A decreasing loss curve alone is insufficient: a wrong gradient can still decrease loss, and nearly redundant predictors can produce tiny loss changes before the optimum is reached.

Two independent checks address these risks. Central finite differences check the derivative against small changes in the objective. `np.linalg.lstsq` calculates a reference least-squares solution for the same training matrix, allowing comparison with the optimum our gradient descent should approach. It is a diagnostic, not a replacement optimizer: validation predictions and saved Phase Two weights still come from `gradient_descent`. We compare errors and predictions rather than demand identical coefficients because redundant columns can have different equivalent coefficient vectors.

The final training RMSE difference is approximately **0.00000775 rentals/hour**, below the chosen 0.001 tolerance. This establishes agreement in training objective to that tolerance; it does not establish good future performance or exact agreement for every individual prediction. The largest training prediction difference is approximately **0.75345 rentals/hour**.

#### Change 1: remove the elapsed-days feature

The initial design included elapsed days along with year, month indicators and day-of-month. The reference check failed even though the small-loss-change rule reported convergence: training RMSE was **100.26228**, versus **100.21294** for the reference optimum on that initial matrix, a gap of about **0.04934**. The smallest positive Hessian eigenvalue was approximately **3.59 × 10⁻⁷**. This identified a direction in which gradient descent improves extremely slowly.

Elapsed days is almost reconstructible from year, month and day-of-month. That near redundancy caused poor numerical conditioning. We removed elapsed days, keeping year/month plus the day-of-month extracted from the date. We chose this simpler representation rather than increasing the iteration budget dramatically or replacing gradient descent, because the calendar information remained represented and the assignment requires our optimizer. After the change, the independent optimum check passed.

**Limit:** this was a numerical-stability choice, not proof that a continuous time trend is useless. The initial and final models have different feature matrices, so their optimum scores must not be compared as if only optimization changed. Other representations could be tested later. The failed intermediate figures above document the debugging run; they are not the current model's scores or its saved handoff artifact.

#### Change 2: pool rare weather categories

The initial encoder stopped because weather code 4 occurred in validation but not in training. A separate category-4 effect cannot be estimated without training examples. Silently encoding it as all zeros would give it the clear-weather reference effect, which is a poor semantic fallback.

We mapped codes 3 and 4 to one adverse-weather group, using the brief's descriptions of light versus severe rain/snow. This mapping was chosen after observing category coverage and uses no target values or hidden-test outcomes. It is then applied identically to every partition and stored in the artifact. This is an explicit modeling assumption: severe weather is represented by the effect learned for the pooled group. It sacrifices severity detail and does not prove categories 3 and 4 have equal demand. We did not change the required seed or split to move rare examples into training.

#### Why these other choices?

| Choice | Reason and practical limit |
| --- | --- |
| One-hot hour, month, weekday, season and weather | Codes represent categories. A numeric hour slope would force the same incremental effect throughout the day; separate hour effects can capture peaks. Reference-category encoding removes the obvious full-dummy/intercept dependency, although dependencies across calendar columns remain. Cyclic encoding is an alternative, but one sine/cosine pair would impose a smooth daily shape. |
| Drop `instant` | It is a record index and can act as a proxy for ordering rather than an interpretable demand driver. This is a baseline modeling assumption, not a demonstrated verdict that it has no predictive information; column-level evidence belongs to Phase Four. |
| Training-only standardization | Different feature scales otherwise produce uneven update sizes. Training statistics prevent validation information from influencing fitted transformations. The intercept stays unscaled and rental counts retain their original units. |
| Retain `temp` and `atemp`, and other calendar redundancy | Training correlation between these weather measurements is about **0.986**, but correlation alone does not establish which predictor should be removed. Retaining them leaves that evidence-based comparison to later phases. Exact redundancy can make coefficients nonunique; prediction checks remain meaningful. |
| Full-batch updates from zero | They are deterministic and straightforward to explain for this dataset size. Linear squared-error regression is convex, so random initialization is unnecessary. Accepting an initial vector supports Phase Two's required warm start. |
| Precompute the Hessian and target cross-product | This computes the same full-batch gradient efficiently for a fixed quadratic objective. It is not a normal-equation solve or a library fit. |
| Half-MSE objective | Dividing MSE by two simplifies the gradient without changing the minimizer. Its squared-error penalty also aligns with the required RMSE evaluation. |
| Rates `0.1/L`, `0.5/L`, `1/L`, `2.2/L` | The training Hessian supplies a principled scale: stable rates lie below `2/L`. The last rate deliberately exceeds that bound to demonstrate divergence, rather than assuming every setting converges. This is a small illustrative search, not an exhaustive optimization study. |
| Select rate **0.0276312112** | It had the lowest validation RMSE among converged candidates: **99.464913**, versus **99.465222** and **99.465300**. The differences are less than 0.001 rentals/hour and do not establish a meaningful generalization advantage. We followed the stated selection rule; a speed-based tie rule could instead choose a faster converging rate. |
| Relative-loss tolerance `1e-10`, patience 20, cap 12,000 | The strict tolerance limits premature stopping; patience avoids relying on one small update, and the cap bounds runtime. These are engineering settings, not requirements or proof of convergence by themselves. The separate optimum check is important precisely because this stopping rule initially failed on the ill-conditioned design. |
| Report raw negative predictions | Clipping would change the evaluated predictor. Reporting the **202** negative validation predictions exposes a genuine operational limitation instead of hiding it. Nonnegative postprocessing may be evaluated later, but is not part of this baseline. |
| Training-mean comparison and residual plots | The constant predictor checks whether added complexity improves over a simple fit. Residual and hourly plots expose patterns an aggregate score can conceal, including underprediction of high-demand hours. They do not establish causality. |
| Separate convergence plot panels | A divergent loss reaching roughly 10¹² compressed the stable curves in the initial shared plot. Showing stable excess loss above the reference optimum separately from divergent raw loss makes both behaviors readable; labels state the different quantities. |
| Save weights, feature order, preprocessing and record IDs | Phase Two must continue this team's actual pipeline. Weights only have meaning with their feature order and transformation, and preserving IDs prevents an accidental new split. Input hashes identify the files used. |

The mandated random split and seed **29578** were not discretionary choices. We retained that split throughout debugging. Validation was used for candidate selection, so its reported score is a tuned validation estimate; unseen-period performance remains unresolved until the required chronological analysis in Phase Three.

### 1. Setup and team seed

`hashlib` computes the required SHA-256 hash. The IDs are strings so sorting follows the brief exactly:

```python
ids = sorted(["16003585", "16003110", "14004179"])
concatenated_string = "_".join(ids)
seed = int(hashlib.sha256(concatenated_string.encode()).hexdigest(), 16) % 100000
```

The sorted string is `14004179_16003110_16003585`; the seed is **29578**. An assertion checks the brief's example seed, 41698. The split explicitly receives our team seed. NumPy performs matrix calculations, Pandas loads and prepares tables, Matplotlib draws plots, and IPython displays tables and Markdown. Scikit-learn supplies only `train_test_split`; no model `.fit()` or `.predict()` is used.

### 2. Load and inspect the data

`pd.read_csv` reads `data/train.csv`, `data/test.csv` and `data/sample_submission.csv`. Checks verify the target, feature schema, unique record IDs, train/test separation, submission ordering, missing values and duplicate rows. Unexpected invalid inputs fail visibly rather than silently changing the dataset.

Parsing `dteday` and adding `hr` reconstructs timestamps. Comparison with a complete hourly range counts hours absent from the labeled file. These may include hidden test hours and operational gaps; they do not imply zero rentals, so no artificial rows are added. Submission-template zeros are placeholders, not hidden test labels. No test data is used for fitting or tuning.

### 3. Create the required split

One call to `train_test_split(train_all, test_size=0.20, random_state=seed)` creates **8,708 training rows and 2,178 validation rows**. No additional split is created in Phase One. Validation is used for learning-rate selection and evaluation, so it is not an independent final test estimate. Both partitions' record IDs are saved for later phases.

### 4. Prepare and scale predictors

`encode_features` builds numeric features:

| Input | Representation and reason |
| --- | --- |
| `instant` | Excluded as a record identifier. A model-backed column verdict belongs to Phase Four. |
| `dteday` | Parsed into day-of-month; year and month are already available. Raw strings cannot enter matrix multiplication. An elapsed-days trend is excluded because it nearly duplicates year/month/day and slows optimization. |
| `hr` | Reference-category one-hot encoding gives separate hour effects without imposing a straight slope from midnight to 23:00. |
| `season`, `weathersit`, `weekday`, `mnth` | Reference-category one-hot encoding avoids assuming equal numeric distances between categories. |
| `yr`, `holiday`, `workingday` | Binary numeric indicators. |
| `temp`, `atemp`, `hum`, `windspeed` | Continuous measurements kept from the supplied data. |
| `cnt` | Target only, never a predictor. |

Weather codes 3 and 4 are pooled as adverse weather using the supplied descriptions, with no target-derived mapping. Category 4 is absent from this team's training portion, so a separate coefficient cannot be learned. Other category levels come only from training. The first level is omitted as the reference category. Remaining unknown categories raise an explicit error rather than silently becoming the reference category. The baseline deliberately retains potentially redundant calendar and weather features: Phase Four will evaluate their usefulness using fitted-model evidence.

`feature_mean` and `feature_scale` contain training means and population standard deviations. Each feature becomes `(value - training_mean) / training_std`, and validation uses those same training values. Constant columns are dropped before division. `design_matrix` adds an unscaled column of ones for the intercept. `X_train`/`X_val` contain predictors; `y_train`/`y_val` contain rental counts. Assertions verify finite values, zero training feature means and unit standard deviations. The target remains in rentals/hour.

### 5. Implement gradient descent

Predictions are `X @ weights`, where `@` means matrix multiplication. Minimize half the mean squared error:

```text
J(w) = sum((Xw - y)^2) / (2m)
gradient = X.T @ (Xw - y) / m
w_next = w - learning_rate * gradient
```

`m` is the number of training rows. The half factor simplifies differentiation. `gradient_descent` starts from zero coefficients or accepts an initial vector for the later required warm start. The intercept is part of that vector.

For this quadratic objective, precomputing `X.T @ X / m` and `X.T @ y / m` gives exactly the same full-batch gradient without repeatedly scanning every row. The loss is updated using its exact quadratic change and verified against direct residual loss afterward. This is our optimizer, not a library fit.

`L`, the largest eigenvalue of the training Hessian, sets the stability scale. Rates below `2/L` are stable for the quadratic objective. Compare `0.1/L`, `0.5/L`, `1.0/L`, and an intentionally excessive `2.2/L` to demonstrate divergence. This grounds the search in the actual training matrix.

Stopping requires relative training-loss change below `1e-10` for 20 consecutive updates, with a 12,000-update cap. Nonfinite or explosive loss terminates a run as diverged. The status distinguishes convergence from reaching the cap. Among converged candidates, the lowest validation RMSE selects the final weights. The actual rate and iteration count are printed and saved.

### 6. Check convergence and implementation

The logarithmic loss plot compares every learning rate. Decreasing training loss demonstrates optimization progress, not generalization. Central finite differences perturb each coefficient and check the analytical gradient's signs and factors.

`np.linalg.lstsq` supplies a diagnostic optimum only: it does not train the reported model or supply the Phase Two weights. Gradient descent's training RMSE must agree within 0.001 rentals/hour, and the final saved loss must equal directly computed half-MSE. Compare predictions rather than requiring identical coefficients because redundant columns can admit multiple equivalent weight vectors.

### 7. Evaluate and interpret

`regression_metrics` calculates the required scores directly:

```text
RMSE = sqrt(mean((actual - predicted)^2))
R² = 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
```

RMSE is error in rentals/hour; lower is better. R² equals 1 for perfect predictions and 0 for predicting that partition's target mean; it may be negative. A separate baseline predicts the training target mean even on validation, so it uses no validation labels to fit.

The metric table shows training and validation R²/RMSE. Actual-versus-predicted points show agreement with a perfect-prediction diagonal. Residuals are `actual - predicted`; positive values indicate underprediction. The hourly profile compares validation mean actual and predicted demand. Negative predictions are counted without clipping to preserve the linear model being evaluated. The coefficient table shows every standardized feature's fitted weight.

The mandated random split mixes time periods and may overstate future-period performance; Phase Three will make the required chronological comparison. Residual patterns are descriptive, not evidence of causality. Redundant inputs may make individual coefficients unstable even when predictions converge.

### 8. Save the Phase Two handoff

The notebook writes [artifacts/phase_one.json](artifacts/phase_one.json), containing:

- Actual gradient-descent weights and exact feature order, including the intercept.
- Selected learning rate, iteration count, stopping settings, initialization and objective.
- Date transformation, weather mapping, training-only category levels, means and scales.
- Ordered split record IDs, metrics, loss history and learning-rate comparisons.
- SHA-256 input-file hashes to identify the data used.

`phase_one_weights` and `phase_one_hyperparameters` are also available in memory. Phase Two must preserve this split and the meaning/order of existing transformed features, copy the fitted coefficients into its expanded weight vector and initialize newly added polynomial coefficients to zero. It must consume these actual artifacts instead of fitting an independent Phase One model.

### Outcome and scope

Measured results for team seed **29578**:

| Model | Partition | R² | RMSE (rentals/hour) |
| --- | --- | ---: | ---: |
| Gradient descent | Training | 0.6937 | 100.2631 |
| Gradient descent | Validation | 0.6980 | 99.4649 |
| Training-mean baseline | Validation | -0.0006 | 181.0340 |

The selected learning rate is **0.0276312112**, with convergence after **9,938 updates**. Rates 0.138156 and 0.276312 also converge; 0.607887 diverges. The selected optimizer's training RMSE differs from the least-squares optimum by approximately **0.00000775 rentals/hour**. Validation demand peaks at **17:00**. The model still underpredicts many high-demand hours and produces **202 negative validation predictions**. These operational limitations motivate later modeling work; convergence alone does not solve them.

The first implementation exposed category 4 occurring only in validation and a poorly conditioned elapsed-date feature. Fixed adverse-weather pooling and excluding that redundant trend resolved those issues without changing the required split or using hidden test targets.

The Phase One **Outcome** summary is calculated from the run, with actual seed, partition sizes, convergence settings, training/validation scores and limitations. Saved notebook outputs provide the measured tables and plots. Re-running all cells regenerates the handoff artifacts. Phases Two through Four follow below; the optional asymmetric-cost bonus and Phase Five remain outside this implementation. No final test predictions overwrite the sample submission.

## Phase Two

### Goal and expectation

Extend the Phase One linear model with selected polynomial weather terms and train with the **same from-scratch gradient-descent function**. The Expectation cell is written before execution: curvature may improve demand predictions, training error should decrease, and higher powers may slow convergence. Validation determines the selected polynomial degree; improvement is not assumed.

### 1. Consume and verify Phase One

`phase_one_input` loads `artifacts/phase_one.json`. Assertions verify seed **29578**, exact feature ordering, ordered training/validation IDs and hashes of all input CSVs. No new split is made. `phase_two_base_weights` contains the actual fitted coefficients. The preceding Phase One cells remain the source of the training-only preprocessing and design matrices.

### 2. Construct the polynomial design

`polynomial_design` keeps the complete Phase One matrix, then adds individual powers of `temp`, `hum` and `windspeed`. Degree 2 adds three squared terms; degree 3 adds those squares and three cubes. This is a **selective additive polynomial basis**, without interaction terms, rather than expansion of every possible combination.

Why these features? Temperature, humidity and wind speed can have curved effects, whereas powers of binary indicators merely repeat the indicators. `atemp` stays in the linear baseline but is not expanded because its training correlation with `temp` is about 0.986. Avoiding a second set of temperature powers reduces redundant complexity. This choice is a reasonable starting hypothesis, not proof that excluded powers or interactions lack predictive value.

Powers are built from Phase One's standardized continuous columns. Each added term is centered and scaled using **training-only** means and standard deviations; validation uses the same values. The original columns are unchanged. This keeps the inherited coefficients meaningful and avoids validation leakage. No hidden-test data enters this process.

The initial expanded vector is the saved Phase One vector followed by zeros. Assertions confirm that both training and validation starting predictions match Phase One exactly within floating-point tolerance. New-term scaling cannot disturb those predictions because new coefficients start at zero.

### 3. Train and verify numerical convergence

`phase_two_runs` stores each fit, while `phase_two_search` records its degree, rate, status, iteration count and scores. Every candidate uses `gradient_descent(..., initial_weights=...)`; no library optimizer fits the reported model. Each starts from the original inherited vector, not another candidate's solution.

The Phase One learning rate is carried forward as a candidate. Additional candidates are `0.5/L` and `1/L`, where `L` is the largest eigenvalue of the expanded training Hessian. Only rates below `2/L` are attempted. Expansion changes the gradient's stability scale, so blindly reusing the old rate would be unjustified.

Patience remains 20 updates. Tolerance is tightened to `1e-12` and the iteration budget is raised to 100,000 because polynomial columns can be highly correlated. These are numerical safeguards rather than validation-selected settings. Convergence status alone is insufficient: training RMSE must also agree with a diagnostic `np.linalg.lstsq` optimum within 0.001 rentals/hour. That diagnostic never supplies the saved fitted weights. Direct-loss checks verify that the optimizer's stored loss matches its predictions.

### 4. Select a degree and compare results

Among converged, numerically verified polynomial candidates, select the lowest validation RMSE. Lower degree and fewer iterations break exact ties. `phase_two_comparison` reports each degree's best verified result alongside the actual degree-1 Phase One baseline, using the same training and validation rows throughout.

The convergence plot shows loss above each degree's diagnostic optimum. The degree comparison displays training and validation RMSE. Prediction and residual plots expose errors hidden by aggregate scores. Scores use original rentals/hour units and raw predictions; negative predictions are counted without clipping, consistently with Phase One.

This small degree selection is part of Phase Two. It does not implement Phase Three's broader complexity diagnosis or chronological split. Reusing validation for selection makes the reported result a tuned validation estimate, not an independent final test score. If no polynomial beats degree 1, the notebook says so while still preserving the selected polynomial for the required next-phase analysis.

### 5. Preserve the Phase Three handoff

`artifacts/phase_two.json` records the selected degree, expanded features, exact term order, weights, initial weights, added-term scaling and inherited preprocessing. It also saves optimizer settings, split IDs, data hashes, metrics, search results and convergence history. A hash of the Phase One artifact identifies the actual upstream input.

Phase Three must consume this degree and expansion choice as its starting point, then perform the required complexity and chronological analyses. The Phase Two Outcome cell computes the measured scores and improvement from this execution. Test files and the submission template are not modified.

### Outcome and decisions from the measured results

| Degree | Predictors (excluding intercept) | Training R² | Training RMSE | Validation R² | Validation RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 — Phase One | 53 | 0.69371 | 100.26311 | 0.69795 | 99.46491 |
| 2 | 56 | 0.69599 | 99.88856 | 0.69967 | 99.18202 |
| 3 — selected | 59 | 0.70036 | 99.16864 | 0.70375 | 98.50538 |

RMSE is measured in rentals/hour. **Degree 3** wins this predefined comparison, reducing validation RMSE by **0.95953 rentals/hour**, approximately **0.96%** relative to Phase One. It adds `temp^2`, `hum^2`, `windspeed^2`, `temp^3`, `hum^3` and `windspeed^3`. The improvement supports carrying this basis forward, but is modest: curvature in these three weather variables does not explain all demand variation.

The selected learning rate is **0.2333705880** with **2,848 updates**, tolerance `1e-12` and patience 20. All six degree/rate combinations converged and passed the optimum check. The selected fit's training RMSE differs from its diagnostic optimum by approximately **9 × 10⁻⁹ rentals/hour**. Differences between learning-rate candidates within a degree are less than **0.0001 rentals/hour** on validation: these tiny differences reflect stopping precision and do not establish a meaningful generalization advantage for one rate. The specified minimum-RMSE rule selects the recorded run.

There are **209 negative validation predictions**, up from Phase One's 202. Better aggregate RMSE therefore does not imply improvement in every operational property. Predictions remain unclipped for consistent evaluation. The residual plot still shows substantial error, including underprediction at the high-demand end. No claim about future-period performance or a final bias–variance diagnosis follows from this random-split comparison.

No feature-set changes or expanded search were made after inspecting these scores: the predefined selective powers and degrees 2–3 produced the reported result. Phase Three receives degree 3 and its exact expansion choice; interactions, alternative weather representations and higher complexity remain possible later investigations, not conclusions from Phase Two.

## Phase Three

### Goal, requirements and expectation

Diagnose the Phase Two model's fit empirically, then provide Phase Four with a diagnosis and target degree. The PDF requires consuming Phase Two's degree/expansion choice, varying complexity with training versus held-out performance, and comparing the seeded estimate with **one chronological split**. The notebook opens this phase with an Expectation written before execution and ends with a measured Outcome and artifact. No regularization or classification is performed.

### 1. Verify the Phase Two input and vary complexity

The artifact's degree **3** anchors the predefined range **1–5**. Keep its selective individual powers of `temp`, `hum` and `windspeed`, and all original linear predictors. This changes complexity without confounding it with a new feature family. It does not expand one-hot indicators or introduce interactions. The original seed **29578**, training rows and validation rows remain unchanged.

For each degree, use the same from-scratch `gradient_descent` function. For degree 3 and above, the existing Phase Two coefficients are copied into matching columns and newly added coefficients start at zero; simpler comparisons use the Phase One initialization. A training-Hessian rate `1/L` is selected for stable numerical optimization rather than another validation search. Tolerance is `1e-12`, patience is 20, and the cap is 500,000 updates to allow higher-degree fits to finish. Compare training RMSE with a diagnostic least-squares optimum within 0.001 rentals/hour before interpreting the complexity curve. Library least squares does not supply reported predictions or saved model coefficients.

This check matters: poor optimization can resemble underfitting, while true bias–variance analysis compares adequately optimized models. Iteration counts, stopping status and numerical gaps are displayed rather than assuming all runs converge. Training/validation R² and RMSE are reported for every degree.

### 2. Create exactly one chronological split

Sort the labeled rows by date and hour. Use the date at the 80% row boundary as the cutoff, and hold out that entire date plus all later dates. The rule uses dates only, not targets or achieved scores. Keeping dates intact avoids placing different hours of a boundary day on both sides. The resulting size is approximately 80/20, with no seed, shuffle, or repeated cutoff search.

Chronological category levels, feature means, scales and polynomial-term scaling are fitted afresh using **earlier rows only**. The fixed weather pooling remains the same domain rule. Initialize chronological coefficients from zeros: copying random-split weights would import information from later dates. Separate local preprocessing functions preserve Phase One and Two's original transformations. The hidden test file is never used.

All degrees are compared on this **same single chronological partition**. In particular, degree 3 re-evaluates the Phase Two model family on unseen later periods, satisfying the PDF's time comparison. This does not create multiple chronological splits.

### 3. Read the evidence

The two complexity panels show training and validation RMSE for the seeded and chronological partitions. A daily plot shows actual versus predicted mean demand across observed hours of the latest-period holdout. Missing hours are not invented or assigned zero demand. A table reports each partition's target mean and spread: R² changes with target variation, so RMSE and R² must be interpreted together.

A declining training curve with rising held-out error supports increasing variance. Similar training and random-validation errors plus persistent residual structure support limitations of the additive model rather than strong random-split overfitting. These patterns are evidence, not a mathematical decomposition of irreducible noise, bias and variance from repeated samples. An observational chronological gap can reflect seasonal or temporal distribution changes as well as overfitting; it does not identify a causal mechanism.

For genuinely later-period deployment, the chronological estimate matches the intended direction of prediction better than mixed-time random validation. One cutoff still cannot establish performance across every future season. The seeded validation set remains the assignment's required tuning partition; chronology is a disclosed robustness check, with any conflicting degree ranking reported.

### 4. Select and preserve the Phase Four input

Choose the degree with lowest seeded-validation RMSE; an exact tie favors lower degree. Compare its chronological behavior and state whether the time result changes the practical diagnosis. A winning degree at the boundary of 1–5 is only the best **tested** degree, not a global optimum. The Phase Four artifact records the numeric comparisons, target degree, expanded feature family, reasoning, original seeded IDs, chronological cutoff and IDs, optimizer settings and upstream artifact hash. Phase Four must use this diagnosis to motivate its L1, L2 and Elastic Net search; those models are not implemented in Phase Three.

### Changes to earlier phases

Phase Three is appended to the notebook. Phase One and Two cell sources are preserved; rerunning the notebook refreshes their outputs and regenerates their artifacts from the same code and seed. Readme-wide scope wording is updated to reflect the additional phase. Their feature preparation, optimizer implementation, selection rules and splitting code are not rewritten.

### Measured results and diagnosis

| Degree | Random train RMSE | Random validation RMSE | Chronological train RMSE | Chronological validation RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 100.2631 | 99.4655 | 91.8688 | 131.0797 |
| 2 | 99.8886 | 99.1821 | 91.4592 | 131.0799 |
| 3 — Phase Two anchor | 99.1686 | 98.5054 | 90.7208 | 130.3631 |
| 4 — Phase Four target | 98.9808 | 98.3458 | 90.3616 | 131.0821 |
| 5 | 98.9299 | 98.3803 | 90.3173 | 131.0440 |

All ten fits converged and passed the training-optimum check. The notebook also reports R², iteration counts and numerical gaps for every row. Degree 3 takes only 20 additional random-split updates because it starts from the actual already-converged Phase Two vector; this is expected pipeline continuation, not a new fit from scratch.

**Phase Two diagnosis:** reasonably fitted within the tested additive feature family on the mandated random split, with substantial temporal generalization weakness. Its random training/validation errors, 99.17/98.51, do not show a large positive generalization gap. Residual high-demand errors and limited gains from additional weather powers leave approximation limits unresolved. Similar errors alone do not prove zero bias or identify the noise floor.

Moving from degree 4 to 5 decreases training RMSE by about 0.051 but increases random-validation RMSE by about 0.034. That is evidence of a small higher-degree variance penalty, not catastrophic overfitting. **Degree 4 is the target for Phase Four**, since its random-validation RMSE is the minimum of this predefined experiment. Its improvement over degree 3 is only **0.1596 rentals/hour**, with no statistical-significance claim.

The single chronological cutoff is **2012-08-05**: **8,703** earlier training rows and **2,183** later validation rows, ending on 2012-12-19. At degree 3, validation R² changes from **0.7038** to **0.6409**, while RMSE rises by **31.8577 rentals/hour**, about **32.34%**. Earlier-training target mean is **174.45**, versus **259.85** in the later holdout; the target spread also differs. These measured shifts help contextualize the gap, but do not establish a causal explanation.

For later-period prediction, the chronological estimate is more relevant because its training/evaluation direction matches deployment. It changes the practical diagnosis from a reassuring random-split fit to a model with material time-generalization risk. One cutoff does not quantify all future-period uncertainty, and the gap cannot be attributed entirely to estimator variance.

Chronology favors degree 3 over degree 4 (130.36 versus 131.08 RMSE). We disclose that conflict: degree 4 follows the required seeded tuning protocol, while degree 3 remains a simpler robustness comparator. The target is not a claim that degree 4 is universally best. Phase Four should compare zero/near-zero, weak and moderate penalty strengths across all required L1, L2 and Elastic Net methods. Redundant coefficients and mild complexity sensitivity motivate stabilization and sparsity; strong shrinkage is not automatically appropriate and may increase bias. Regularization alone is not guaranteed to solve temporal distribution changes.

### Saved artifact and PDF checklist

[artifacts/phase_three.json](artifacts/phase_three.json) contains the evidence-based diagnosis, target degree **4**, expanded feature family, all numeric comparison rows, target weights/preprocessing, original seeded IDs, single chronological cutoff and IDs, optimizer settings and source/upstream hashes. It is the input to Phase Four.

- Phase Two's degree and expansion choice anchor the experiment.
- Training and held-out R²/RMSE are compared across degrees 1–5.
- Exactly one chronological split trains on earlier dates and evaluates later dates.
- Random/chronological differences, limits, future-data relevance and their effect on the diagnosis are explained.
- A justified target degree and diagnosis are saved for Phase Four.
- Expectation, measured Outcome, numerical checks, plots and written decision justifications are included.

Phase Four follows below. Phase Five remains unimplemented. No hidden-test tuning, final submission predictions, penalty fitting or classification is performed in Phase Three itself.

## Phase Four

### Goal and expectation

Compare **L1/Lasso, L2/Ridge and Elastic Net**, consuming Phase Three's diagnosis and target degree **4**. The expectation is that mild shrinkage may stabilize redundant features, while strong penalties may worsen the remaining approximation bias. No assumption is made that regularization fixes the chronological score gap. The notebook includes Expectation and measured Outcome cells and saves a Phase Five handoff.

### 1. Preserve the inherited pipeline

Verify the Phase Three artifact hash, source CSV hashes, seed **29578**, exact split IDs and feature order. Reuse its 62 degree-4 predictors, including the selected powers of temperature, humidity and wind speed. Every scale and category mapping still comes only from training. No new split or test-file tuning is introduced. Earlier phase cell sources are preserved.

### 2. A common regularized objective

The from-scratch `fit_regularized` solver minimizes:

```text
mean((y - intercept - Xw)^2) / 2
  + lambda * ((1 - rho) * sum(w^2) / 2 + rho * sum(abs(w)))
```

For Ridge, `rho=0`: squared coefficients are penalized, usually shrinking rather than zeroing them. For Lasso, `rho=1`: absolute coefficients are penalized and soft thresholding produces exact zeros. Elastic Net combines both, testing `rho=0.2, 0.5, 0.8`. Lambda controls overall penalty strength. The common objective makes the definitions explicit rather than comparing incompatible library conventions.

Training predictor means and the training target mean are subtracted inside the solver. The intercept is reconstructed afterward and is **never penalized**. The slope gradient is `G @ w - b + lambda*(1-rho)*w`. A soft-threshold operation handles the nondifferentiable L1 part. Accelerated proximal-gradient updates use the reciprocal largest smooth-Hessian eigenvalue as the step, with acceleration restart when its direction is unsuitable. Fits initialize from Phase Three's actual coefficients.

Convergence is checked through the KKT optimality conditions: nonzero coefficients must balance gradient and penalty, while zero coefficients may have a gradient inside the L1 subgradient interval. The maximum residual must be below `1e-7*(1+max(abs(b)))`; the cap is 100,000 iterations. This avoids declaring success just because loss changes slowly. Direct objectives are recorded. An independent closed-form Ridge solution checks the solver's predictions within 0.01 rentals/hour; it does not supply the fitted weights.

### 3. Search and compare fairly

The predefined lambda grid is `0.0001, 0.001, 0.01, 0.1, 1, 10, 100`. This spans near-zero through strong shrinkage, following Phase Three's evidence of only mild random-split complexity sensitivity. Ridge and Lasso each have seven candidates; Elastic Net has 21 across its three mixing ratios. All 35 candidates use the same rows, predictors, scaling, starting vector and optimality tolerance. The inherited unpenalized degree-4 model is a separate reference.

Select the lowest validation RMSE per method, report both training/validation R² and RMSE, and compare all three winners with the unpenalized model. The RMSE plot shows each method's curve; at each lambda, the Elastic Net point uses the mixing ratio with lowest validation RMSE. The adjacent plot shows that same candidate's nonzero count. This is validation tuning, not independent test evidence, and no statistical significance is inferred from small differences.

### 4. Verdicts for every original column

The PDF requires numbers from our own models, so coefficients alone are not the entire argument. Map encoded features back to each source: `temp`, `hum` and `windspeed` include their powers; categorical inputs include all their indicators; `dteday` contributes day-of-month, while year/month are also represented by their supplied columns. Remove each entire group, refit the winning regularized method at the **same** lambda/rho, and measure the increase in validation RMSE. This is a conditional fixed-hyperparameter ablation, not a causal effect or retuning each reduced model.

A predeclared practical sensitivity threshold of **0.1 rentals/hour** classifies a group as useful if its removal worsens RMSE by more than that amount. Otherwise, classify it as redundant when other retained predictors reconstruct its encoded values with validation R² at least **0.90**, or uninformative within this tested basis otherwise. Reconstruction is fitted on training predictors only, without demand labels. These thresholds are operational conventions, not significance tests or universal feature truths; exact numeric deltas remain visible.

`instant` was excluded by earlier phases. A separate add-feature diagnostic standardizes it on training only, refits the fixed regularized model with the index added and measures its incremental validation contribution. The table's positive delta for that row means adding the index improves RMSE. This tests the identifier assumption rather than merely repeating it. It is not added to the inherited production feature set by this diagnostic. `cnt` is explicitly target-only and never enters a predictor matrix.

The verdict table also reports each group's surviving Lasso and Elastic Net terms. Correlated inputs can substitute for each other, so a zero coefficient or a small conditional ablation effect does not prove that the original information is useless everywhere. Similarly, the representation of `dteday` is assessed incrementally alongside existing year/month features, not as a claim that all date information is uninformative.

### 5. Save the classifier feature handoff

[artifacts/phase_four.json](artifacts/phase_four.json) stores final lambda and mixing settings for **each** method, complete fitted weights/intercepts, optimum residuals, candidate scores, original-column verdicts, source/upstream hashes and split IDs. It preserves the inherited preprocessing and degree-4 feature order.

The explicit Phase Five selection rule is the **nonzero subset from the validation-selected Lasso**, excluding the intercept. Lasso is chosen as the pure L1 sparsity mechanism, not because a classifier has already been evaluated. Elastic Net's separate nonzero subset is also saved. The numerical ablation verdicts are complementary conditional evidence, not an additional silent filtering rule. The best regression method can differ from the method supplying classifier sparsity. Phase Five must consume the saved subset and transformations; no binary target or classifier is trained here.

### Results and interpretation

| Method | Lambda | L1 ratio | Training RMSE | Validation R² | Validation RMSE | Nonzero terms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unpenalized degree 4 | 0 | — | 98.98077 | 0.704710 | 98.34580 | 62 |
| Lasso — selected | 0.01 | 1.0 | 98.98098 | 0.704721 | 98.34383 | 57 |
| Ridge | 0.0001 | 0.0 | 98.98079 | 0.704707 | 98.34627 | 62 |
| Elastic Net | 0.0001 | 0.8 | 98.98077 | 0.704709 | 98.34589 | 62 |

All 35 candidates passed the numerical optimality check. Lasso is the recorded regression winner, but its **0.00197 rentals/hour** RMSE improvement is practically negligible and does not demonstrate statistically significant superiority. The main observed benefit is reducing the representation from **62 to 57 terms**. Ridge and Elastic Net prefer the weakest tested penalties and do not beat the unpenalized baseline. Their boundary choices indicate little support for shrinkage in this grid, not evidence of a globally optimal positive penalty. Stronger penalties worsen validation error. These results agree with Phase Three's finding of no large random-split overfitting problem, and provide no claim that its chronological gap has been repaired.

### Original-column verdicts and numeric evidence

Positive delta means removal worsens RMSE. For `instant` only, delta instead measures the improvement from adding it, since it is absent from the inherited basis. Reconstruction R² is auxiliary feature reconstruction, not demand-prediction R². Labels follow the predefined 0.1/0.90 policy above.

| Original column | Verdict in this tested model | RMSE delta (rentals/hour) | Reconstruction R² | Lasso terms retained |
| --- | --- | ---: | ---: | ---: |
| `instant` | Redundant | +0.00912 (addition gain) | 0.99999 | 0; not included |
| `dteday` | Uninformative increment under threshold | +0.08919 | 0.02710 | 1 |
| `season` | Redundant | -0.00179 | 1.00000 | 3 |
| `yr` | Useful | +9.43777 | 0.04640 | 1 |
| `mnth` | Useful | +1.19885 | 0.35750 | 8 |
| `hr` | Useful | +45.72496 | 0.01519 | 23 |
| `holiday` | Redundant | -0.00521 | 1.00000 | 1 |
| `weekday` | Useful | +0.11684 | 0.14247 | 5 |
| `workingday` | Redundant | -0.00143 | 1.00000 | 1 |
| `weathersit` | Useful | +1.17817 | 0.15457 | 2 |
| `temp` | Useful | +1.08211 | 0.61497 | 4 |
| `atemp` | Redundant | +0.04207 | 0.97150 | 1 |
| `hum` | Useful | +0.39033 | 0.29536 | 3 |
| `windspeed` | Useful | +0.11358 | 0.10468 | 4 |
| `cnt` | Target only, not an input feature | Not applicable | Not applicable | Never included |

`dteday` here means the incremental day-of-month feature, not all date information: useful year/month predictors remain present. Its 0.08919 effect lies just below the chosen 0.1 threshold; a different practical threshold could change the label. Weekday and wind speed are also marginally above it. These are conditional sensitivity labels, not assertions that a feature has exactly zero information.

Season can be reconstructed from the retained month indicators, and holiday/working-day information overlaps with the weekday/calendar representation. Those findings support redundancy **conditional on keeping the other columns**. Removing every redundant group simultaneously is not justified by one-at-a-time ablations. Lasso also may retain some terms from a group labeled redundant because correlated representations can substitute for one another; the coefficient mask and ablation question are different. The exact nonzero Lasso subset is therefore saved intact, without silently dropping every low-impact group.

The added record index changes validation RMSE only from 98.34383 to 98.33472, below the sensitivity threshold, and is almost completely reconstructible from existing calendar information. It remains excluded from the inherited pipeline. No identifier experiment or column verdict uses hidden-test labels.

Phase Five remains pending. It will use the 57 stored Lasso features; Phase Four does not define the binary target, train a classifier or overwrite the submission template.

The five exact-zero Lasso terms are `weekday_2`, `mnth_6`, `mnth_7`, `mnth_12` and `hum^2`. The original humidity input is still useful through its retained linear/cubic/quartic terms; removing one encoded term is not the same as removing the whole original column.
