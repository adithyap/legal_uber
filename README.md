# Judge speed analysis (EWHC/KB)

Methodology (current pipeline)
- **Measure speed:** Hearing → judgment turnaround days; drop/clip obvious bad values and optionally trim/winsorize extreme tails.
- **Model:** Fixed-effects regression `speed_ij = α + γ_judge + controls + ε_ij`, with judgment-year (and optional hearing-year/area) controls. One judge is the baseline; γ_judge gives the day difference vs. that baseline (negative = faster).
- **Inference:** Joint F-test on all γ_judge answers “do judges differ systematically?” Dispersion stats (std, percentiles, IQR) show economic magnitude.
- **Robustness to tails:** Re-run on `log(turnaround+1)`; significance here indicates results aren’t driven solely by a few very long cases.
- **Visualization:** Turnaround histograms (raw/log) and judge-effect bar/box plots to reveal tails and spread.
