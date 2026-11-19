# Drift Detector

Streaming concept drift detection and online model adaptation using **ADWIN** (Adaptive Windowing) and **Page-Hinkley** cumulative-deviation tests.

## Problem Statement

When the conditional target relationship $P(y \mid X)$ changes over time (concept drift), a frozen model suffers severe post-drift degradation. `drift-detector` performs prequential (predict-then-observe) streaming evaluation, monitors scalar prediction errors in real time, isolates the post-drift regime upon detection, retrains an online regressor on clean post-drift data, and benchmarks adaptive recovery against a static non-retrained baseline.

## Algorithms

### 1. `ADWINDetector(delta=0.002, min_window=10)`
- Maintains a variable-length sliding window $W$ of scalar prediction errors.
- Evaluates split points $(W_0, W_1)$ with $|W_0|, |W_1| \ge \text{min\_window}$ against the Hoeffding cut criterion:
  $$|\hat{\mu}_0 - \hat{\mu}_1| \ge \epsilon_{\text{cut}}, \quad \epsilon_{\text{cut}} = R \sqrt{\frac{1}{2m} \ln \frac{4}{\delta'}}, \quad m = \frac{n_0 n_1}{n_0 + n_1}, \quad \delta' = \frac{\delta}{\ln n}$$
  where $R = \max(W) - \min(W)$ is the empirical window range.
- Upon detection, drops stale pre-drift subwindows $W_0$ and records pre/post window means.

### 2. `PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=1.0, min_instances=30, mode="both")`
- Tracks the running mean $\bar{x}_t$ and $\alpha$-weighted cumulative deviations:
  $$U_t = \alpha U_{t-1} + (x_t - \bar{x}_t - \delta), \quad PH_t^{\uparrow} = U_t - \min_{0 \le k \le t} U_k$$
  $$L_t = \alpha L_{t-1} + (x_t - \bar{x}_t + \delta), \quad PH_t^{\downarrow} = \max_{0 \le k \le t} L_k - L_t$$
- Supports two-sided (`"both"`) and directional (`"up"`, `"down"`) detection; triggers once `n >= min_instances` and resets accumulators cleanly for the new regime.

### 3. `DriftDetectorPipeline`
- Warm-up fits an `OnlineRegressor` (Ridge-initialized `SGDRegressor` with incremental `partial_fit` updates).
- Monitors streaming absolute prediction errors $|y_t - \hat{y}_t|$ using ADWIN, Page-Hinkley, or both in parallel.
- Records structured `DriftEvent(step, detector, pre_mean, post_mean, magnitude)` records, retrains cleanly on the isolated post-drift slice, and reports overall and post-drift MAE/MSE against a static baseline.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the default comparison with both ADWIN and Page-Hinkley detectors:

```bash
python main.py --detector both
```

Additional CLI options:

```bash
python main.py \
  --detector both \
  --data-size 1000 \
  --drift-at 500 \
  --shift-magnitude 3.0 \
  --drift-type abrupt \
  --seed 42
```

Supported flags:
- `--detector {adwin,page_hinkley,both}`
- `--data-size INT`
- `--drift-at INT`
- `--shift-magnitude FLOAT`
- `--drift-type {abrupt,gradual}`
- `--seed INT`

### Sample Output

Output from `python main.py --detector both`:

```text
INFO src.pipeline: Drift detected at step 554 (adwin): pre_mean=0.1634, post_mean=4.1641, magnitude=4.0007
INFO src.pipeline: Drift detected at step 514 (page_hinkley): pre_mean=0.1634, post_mean=3.7055, magnitude=3.5421
Stream length: 1000 (warmup: 100, evaluated: 900)
Drift point: 500 | Mode: both

Detected Drift Events:
  - step=514  detector=page_hinkley pre_mean=0.1634 post_mean=3.7055 magnitude=3.5421
  - step=554  detector=adwin        pre_mean=0.1634 post_mean=4.1641 magnitude=4.0007

Model Error Comparison (Static Baseline vs. Adaptive Retraining):
  Model                      Overall MAE  Overall MSE   Post-Drift MAE   Post-Drift MSE   Retrains
  -----------------------------------------------------------------------------------------------
  static_baseline                 2.2430      12.9137           3.9125          23.2140          0
  adaptive (adwin)                0.4053       1.5878           0.6050           2.8276          1
  adaptive (page_hinkley)         0.2293       0.4490           0.2882           0.7778          1
```

## Gating a pipeline with `--fail-on`

By default the CLI always exits `0`, so a monitoring job cannot alert on drift.
`--fail-on drift` exits `1` when the stream produced any drift event (and `0` on a
clean, stationary stream), so the tool can gate or alert in a scheduled pipeline:

```bash
python main.py --detector both --fail-on drift
```

An unknown level is rejected by argparse rather than silently disabling the gate.

## Running Tests

```bash
pytest -v
```

## License

MIT — see [LICENSE](LICENSE).
