# Final requirement check

Reviewed on **17 October 2026** against the supplied seven-page project brief.

**Status: Share with caveats. Complete within the checked repository scope.** No missing mandatory technical item or unresolved numerical implementation error was found. This is a file/code review, not a promised grade or a certification of external course requirements.

## What is done and what remains

- **Done:** all five required phases; code explanations and reasons; notebook execution; six-page report; final 574-row count CSV; numerical and requirement checks.
- **Bonus done:** our own 3:1 asymmetric-loss gradient descent, gradient derivation in the report, and a measured comparison/plot against the Phase One MSE model.
- **Team tasks left:** confirm registration and roster, submit the required files, and have every member prepare to explain/change any phase live.
- **Project status:** technical deliverables are ready, including the optional bonus. The course submission and live evaluation are still team responsibilities. No full-mark or hidden-score guarantee is possible.

## Mandatory technical requirements

| PDF reference | Requirement | Evidence and result |
|---|---|---|
| 1.7, Phase 1 | Configurable gradient descent from scratch; convergence shown | Notebook Phase One: our updates, tested rates/loss plot, derivative and optimum checks; no library model fit/predict calls |
| Phase 2 | Polynomial model uses Phase One weights and its optimizer | Exact weight-plus-zero warm-start checks; same gradient-descent function; selected degree 3 and scores saved |
| Phase 3 | Empirical complexity comparison | Degrees 1–5, training/validation R² and RMSE; degree-4 target and stated diagnosis |
| Phase 3 | One earlier/later chronological split and explanation | Cutoff 2012-08-05; 8,703 earlier / 2,183 later rows; fresh earlier-only preprocessing; random/time estimates compared |
| Phase 4 | Implement and compare L1, L2 and Elastic Net | All three implemented; 35 candidate fits; winning settings and model-based column checks saved |
| Phase 4 | Verdict for every original input | 14 original inputs each have a verdict and numeric evidence; `cnt` is explicitly target-only |
| Phase 5 | Logistic regression uses only selected features | Exact ordered 57-term Lasso subset; no dropped input reintroduced |
| Phase 5 | Derive a meaningful high-demand label | Training-only hour/day-type 75th percentile; global-threshold bias explained; class balance reported |
| 1.6 | Required metrics on the team's validation rows | Phases 1–4: R²/RMSE; Phase 5: accuracy/F1/ROC-AUC, plus precision/recall and baseline |
| 1.7–1.10 | Numeric chaining between all phases | Actual weights, degrees, diagnoses and masks passed forward; upstream hashes and feature order verified |
| 1.8 | Correct roster seed and one seeded 80/20 split | IDs 16003585, 16003110, 14004179 → seed 29578; exactly one split call; 8,708 / 2,178 rows |
| 1.8 | Training-only transformations; no hidden-test tuning | Means/scales, categories, powers and label cutoffs independently checked; test data only checked for schema/order before final predictions |
| 1.7, 1.9 | Written reasons for open choices | Each README phase has code steps, choices, settings, result and explanation questions; notebook/report preserve the mathematical details |
| 1.7 | Expectation before execution and final Outcome | Five original expectations preserved; each phase now has a separate Expectation and ends with an explicit numeric Outcome markdown cell. File inspection cannot independently certify historical writing time. |
| 1.11 | Notebook runs top to bottom | Full revised run passed: 36 sequential code cells, zero errors. The bonus and updated report were included in this full run. |
| 1.11 | Report has at most six pages | Six pages; all pages rendered and checked; chosen settings and actual phase results included |
| 1.11 | Filled count submission in exact format | Root sample_submission.csv: 574 rows, instant/cnt columns, original test order, finite nonnegative predictions |
| 1.8, 1.11 | Final refit and postprocessing justified | Selected Lasso refitted on all labeled rows after tuning; zero clipping stated; raw/clipped validation scores refer to earlier training-only model |
| 1.12 | No external training data | All five artifacts reference the unchanged supplied CSV hashes |

## Optional bonus: completed

PDF Phase One's +5% bonus is implemented in the notebook and explained in README Phase One. It uses `a=3` for underprediction and `a=1` otherwise, with loss `sum(a*e²)/(2m)` and gradient `X.T @ (a*e)/m`. The piecewise derivative is derived on report page one; both sides have derivative zero at zero. This is explicitly a weighted squared-cost interpretation.

The bonus uses the original split/features/scaling and its own gradient descent. The validation comparison and hourly plot show the upward shift: mean predictions rise 42.22 rentals/hour, asymmetric cost falls 20.05%, and ordinary RMSE changes from 99.46 to 108.77. It is a separate experiment; all five original phase artifacts and final count CSV remain unchanged.

[bonus_verification.json](bonus_verification.json) records 43 incremental checks, including independent reconstruction of the bonus scores and training gradient, unchanged main calculation code/artifact hashes, full execution and the six-page report limit. This supplements the earlier 120-check main-pipeline audit; it is not an independently refitted bonus or a promised grade.

## Important interpretation limits

- The random validation set was reused to select degree, penalties, features and decision threshold. Its scores are tuned estimates, not an independent final test score.
- The chronological result is materially worse. The model is not claimed to be robust to every future period; regularization did not prove this gap solved.
- The `dteday` verdict is for its added day-of-month representation, with year/month present separately. It does not mean all date information is uninformative.
- Feature-removal verdicts are conditional on other inputs and the chosen 0.1-rental threshold. Do not remove every overlapping group at once or interpret coefficients causally.
- The high-demand classifier predicts a relative flag among comparable hours; it is not an absolute station-capacity rule.
- Original expectations were kept, not rewritten to match results. Their historical timing cannot be proved from the final files alone.

## Items a file audit cannot finish for the team

1. Confirm team registration and that the registered roster matches the three IDs used for the seed.
2. Submit the required notebook, PDF and root CSV through the course form by the course deadline.
3. Every member must explain **any phase** and change/re-run code live; knowing only an assigned phase is insufficient under the PDF.
4. Hidden-test performance and the instructor's assessment cannot be guaranteed.

## What changed in the final revision

- Replaced repeated README text with a shorter phase-by-phase team guide and glossary.
- Simplified notebook Markdown while keeping formulas, choices, reasons and the original expectations.
- Added a clear final Outcome markdown cell to every phase.
- Added the missing selected Phase Two learning-rate/iteration detail to the report and clarified other optimizer settings.
- Kept the original model-calculation syntax trees unchanged. The later bonus adds its own side experiment and report content.
- Removed temporary recovery/work folders and the revision script after checking their resolved paths; kept `.venv`, `.git`, `.vscode`, `data` and `artifacts`.
- Added a root `.gitignore` to keep environments, caches and temporary work out of Git.

## Independent numerical evidence

Saved [numeric_audit.json](numeric_audit.json) records 120 independent checks with no failures. Its checker rebuilt the saved transformations from the raw CSVs instead of calling the notebook's preparation helpers. It reproduced phase scores, the 48 training label cutoffs, classifier metrics, the exact 57-feature mask, and all 574 count predictions. It also checked regularized-model optimality. This does not mean 120 rubric points or guarantee a mark.

[final_review.json](final_review.json) records the final file hashes and structural checks. [review_coverage.json](review_coverage.json) maps the scoped review items to evidence. Grid fits were regenerated in the notebook; each candidate was not separately refitted by the independent checker. The ten main-pipeline PNG outputs are byte-identical to the earlier reviewed figures. The bonus adds one separately inspected comparison figure. All six current report pages were inspected.

## Review coverage

The following totals count scoped items, not grader points or an accuracy percentage. Categories overlap. `0 / N` means no remaining defect observed in that inventory; it does not promise perfect or complete behavior outside the reviewed scope. Dashboard-only source panels are not applicable to this notebook/PDF project.

### Organization and presentation

| Category | Observed defects | Assessment |
|---|---|---|
| Usefulness and completeness | 0 / 8 | All five phases and three submission files checked against the supplied brief. Registration and live evaluation are outside the file audit. |
| Analytical clarity | 0 / 6 | Five phase guides and the README simplified; formulas, settings, reasons and limits retained. Team understanding is not certified. |
| Visual and interaction consistency | 0 / 17 | Six current PDF pages inspected; eleven saved static figures checked using current exports and unchanged plotting/data evidence. No interactive dashboard. |

### Calculations and conclusions

| Category | Observed defects | Assessment |
|---|---|---|
| Source authority and confidence | 0 / 6 | All models use the supplied CSVs; original-source hashes match across all five artifacts. |
| Calculation and value accuracy | 0 / 6 | Independent reconstruction verifies five phase-score groups and the count submission; all numerical checks pass. Grid fits were regenerated, not each independently refitted. |
| Within-chart agreement | 0 / 11 | Ten main-pipeline figures retain their calculations; the new bonus figure agrees with saved cost/shift results. |
| Complete source details | N/A | This notebook/PDF project has no dashboard source panels. CSV paths, fitted transformations and hashes are recorded in the notebook and artifacts. |
| Cross-artifact consistency | 0 / 6 | Five phase-result groups agree across saved artifacts, displayed summaries and the report; submission reproduces final saved weights. |
| Data-quality controls | 0 / 6 | Phase inputs preserve split IDs and training-only preprocessing; original hashes and final submission schema/order checked. |
| Conclusion support | 0 / 5 | All five phase interpretations retain the small gains, conflicting temporal ranking and repeated-validation limits. No full-mark or causal claim is made. |
