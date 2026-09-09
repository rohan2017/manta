# Archived investigation scripts

These are design evidence, including rejected hypotheses. They are not the
acceptance runners for the current estimator. Several monkeypatch the then
current implementation and are intentionally preserved in their original form;
running them against a newer chart or packet state layout changes the experiment
or can make it fail to construct.

The recovered product-covariance baseline is `79e39c7`. The first integrated
nonlinear chart is `1fa959c`; the v1 chart with active boundary estimation and
stable Earth mechanics is `fa017e7`. The affine-gyro and fixed-reference controls
were developed against that v1 implementation. Some earlier exploratory scripts
predate a committed intermediate API and are archival rather than maintained
command-line interfaces.

For reproducible current acceptance runs, use the `examples.qualification`
modules and commands in [the qualification report](../ins-nonlinear-2026-09-09.md).
`earth_ins_verify` independently recomputes the primary confidence intervals
from the committed held-out reports. The report distinguishes current accepted
results from historical failures and the invalidated old packet fixture.
