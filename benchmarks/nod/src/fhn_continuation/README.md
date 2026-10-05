# Current FHN three-source continuation

The current paper uses source seeds **42/43/44**. This directory adds the actual RMSprop-restart wrapper and frozen K-history/report-recipient evaluation code to the earlier shared `fhn_minimal` implementation. The public `replay_selected.py` dispatcher binds portable paths and reproduces the frozen stage commands; it replaces the original machine queue only.

The recipe was recovered from selected checkpoint ancestry and cross-checked against six completed historical run records. It is not merely a planned paper budget. Full training and original checkpoints were not rerun or redistributed during source packaging.

| Method | Cumulative stage endpoints | Alignment | Final update budget |
|---|---|---|---|
| NOD | 20,000 → 30,000 → 40,000 → 45,000 | Prediction only throughout | 45,000 |
| SPRII | 10,000 → 30,000 → 40,000 → 50,000 | 0.01, 2,000-step warmup, only through 10,000; then prediction only | 50,000 |

Both use batch size 8, latent dimension 2, RMSprop initial learning rate `0.5/sqrt(P)` for P trainable parameters, and StepLR decay 0.95 every 500 updates. At the 40,000-step boundary, `fhn_lr_restart.py` resets only the optimizer-group learning rate to `1e-5`, preserving RMSprop moments and scheduler state. The seeded sampling stream restarts at every continuation. The budgets differ between methods; do not describe this as an equal-update comparison.

The machine-readable recipe is `../../configs/fhn_selected_three_seed.json`. It preserves the actual stage parser values, including the inert NOD first-stage alignment default: `method=nod` disables alignment regardless of that numeric parser default. `effective_alignment_weight` makes the distinction explicit.

Seed 42 selected a recipe using ID target initial condition 5 and mean relative-L2 over H1/H5/H50. Seeds 43 and 44 repeated that recipe and endpoint without checkpoint reselection. Final report recipients are initial conditions 15/24/45; initial condition 5 must be excluded from reported final rows.

From the repository root, inspect commands for one run:

```sh
python benchmarks/nod/src/fhn_continuation/replay_selected.py \
  --method sprii --seed 43 --data-dir data/fhn \
  --official-code benchmarks/nod/third_party/nod_original/code/DR2D \
  --output runs/fhn/sprii_seed43
```

Add `--execute` to perform this run. Repeat the explicit method/seed combinations as needed. This is a full training recipe, not a smoke test.

For a completed checkpoint, evaluate matched history budgets and then apply the final reporting restriction:

```sh
PYTHONPATH=benchmarks/nod/src \
  python benchmarks/nod/src/fhn_continuation/fhn_multi_history.py \
  --checkpoint CHECKPOINT --output runs/fhn/multi_history.json \
  --data-dir data/fhn --official-code benchmarks/nod/third_party/nod_original/code/DR2D
PYTHONPATH=benchmarks/nod/src \
  python benchmarks/nod/src/fhn_continuation/fhn_report_probe.py \
  CHECKPOINT runs/fhn/multi_history.json runs/fhn/report.json \
  --data-dir data/fhn --official-code benchmarks/nod/third_party/nod_original/code/DR2D
```

K=1/2/4 codes are averages of independent same-system histories. Donors exclude the recipient. K=1 is numerically checked against the native rollout, and source weights must remain unchanged. Physical-parameter probes use training histories 50–89, with same-K aggregation, and are refitted after excluding the selection recipient from report rows. These are historical evaluation results (`test_read=false`), not a newly sealed test.

The separate `fhn_minimal_closure.json` is retained only as the historical one-seed precursor. It does not describe this three-source result.
