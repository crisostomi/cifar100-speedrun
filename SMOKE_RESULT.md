# Smoke Result

Job `48374080` ran on Leonardo `IscrC_SIMP` on 2026-07-03.

Purpose: verify that the benchmark executes end-to-end. This is not target-selection or baseline evidence.

Hardware: NVIDIA A100-SXM-64GB on `lrdn1402`.

Command path: `slurm/smoke.sh`, which runs one warmup and one tiny `C100_EPOCHS=0.05` run.

Observed output:

```text
config model=simple_resnet_muon runs=1 epochs=0.05 batch=1024 target=0.01 no_tta=1
|  warmup  |   eval  |     0.0243  |   0.0195  |       1.0000  |      28.2364  |
|       1  |   eval  |     0.0146  |   0.0145  |       1.0000  |       0.2221  |
```

Conclusion: data staging, model construction, Muon step, training loop, plain no-TTA validation, and Slurm account guard all execute.
