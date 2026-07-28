# Week11.ipynb — Apple Quarterly Revenue Regression

**Goal:** Build a linear regression model of Apple's quarterly revenue (`saleq`) over time, then extend it with a seasonal dummy for Q1 releases.

## Data
- Source: `qSales_2024.csv` — quarterly financials by ticker (`gvkey`, `datadate`, `fqtr`, `tic`, `saleq`, etc.)
- Filtered to Apple only:
  ```python
  apple_sales = qSales.loc[qSales['tic']=='AAPL']
  ```

## Workflow

1. **Dates** — convert `datadate` to datetime with `pd.to_datetime`.

2. **Visualize** — plot `saleq` vs. `datadate` to eyeball trend/seasonality.

3. **Time index** — create a `time` column as the independent variable (each row = one quarter):
   ```python
   apple_sales['time'] = range(1, len(apple_sales) + 1)
   ```

4. **Train/test split** — 75/25, taken by row position (not random):
   ```python
   dt4training = apple_sales[:int(0.75*len(apple_sales))]
   dt4testing = apple_sales[int(0.75*len(apple_sales)):]
   ```

5. **Model 1 (trend only)** — OLS of `saleq` on `time` (+ constant via `sm.add_constant`):
   ```
   saleq = -13,536.82 + 1,077.61 × time
   ```

6. **Predictions** — `model1.predict(x_test)` for point forecasts; `model1.get_prediction(x_test).summary_frame(alpha=0.2)` for 80% confidence/prediction intervals (`obs_ci_lower` / `obs_ci_upper` = prediction interval).

7. **Seasonal dummy** — `release_dummy_variable = 1` if `fqtr==1` (fiscal Q1, Apple's holiday-quarter release), else 0.

8. **Interaction term** — `release_dummmy_interaction = time × release_dummy_variable` (note: this spelling with 3 m's matches the notebook's actual column name).

9. **Model 2 (trend + dummy + interaction)**:
   ```
   saleq = -11,044.75 + 933.21 × time − 10,422.13 × release_dummy + 578.33 × (time × release_dummy)
   ```

10. **Predictions (Model 2)** — same `get_prediction().summary_frame(alpha=0.2)` pattern on the interaction-model test set.

## ⚠️ Known bug

In the later cells, `dt4testing` is redefined incorrectly — it reuses the same slice as `dt4training`:

```python
dt4training = apple_sales[:int(0.75*len(apple_sales)):]
dt4testing = apple_sales[:int(0.75*len(apple_sales)):]   # should be [int(0.75*len(apple_sales)):]
```

This means Model 2's "test" predictions are actually generated on the training set, not a holdout. Worth fixing if accuracy/MAPE comparisons matter downstream.

## Notes / caveats

- The `statsmodels.api.OLS`, `get_prediction`, and `summary_frame` method names/usage shown here match the notebook as written. I have not independently re-verified them against the current `statsmodels` docs — worth a quick check if you're citing exact API behavior elsewhere.
- Coefficient values above are read directly from the notebook's printed output as of the upload date (2026-07-28); re-run to confirm if the underlying CSV has changed.
