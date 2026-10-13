# Bike-sharing demand
Machine Learning Project 1: a chained pipeline using the supplied hourly bike-sharing records. The current implementation completes **Phase One only**.

Open this folder in VS Code, then select `.venv\Scripts\python.exe` through
**Python: Select Interpreter**. In `bike_sharing_demand.ipynb`, use
**Select Kernel → Python Environments** to choose the same environment.
Restart the notebook kernel and run all cells in order from the project folder. The notebook reads the CSV files from `data/`, trains the Phase One baseline and saves its handoff artifact.

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

The **Outcome** summary is calculated from the run, with actual seed, partition sizes, convergence settings, training/validation scores and limitations. Saved notebook outputs provide the measured tables and plots. Re-running all cells regenerates the handoff artifact. The optional asymmetric-cost bonus and Phases Two through Five are outside this implementation; no final test predictions overwrite the sample submission.
