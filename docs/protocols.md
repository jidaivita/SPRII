# Experimental protocols

The benchmark configuration and implementation are authoritative for exact
seeds, budgets, data partitions and command-line arguments. This guide describes
the common interpretation; it does not overwrite benchmark-specific protocols.

## Independent interactions and relation supervision

Same-system pairs use distinct interactions from a shared physical system.
Keep system identity separate from rollout identity. The D-Clean sampler in
`persistent_jepa.sampling` constructs same-system pairs, disjoint windows for a
same-rollout control, collision-free wrong relations, and deterministic donor
derangements. Other environments expose their own relation construction.

Native prediction, persistent alignment and cross-context prediction must have
matched observation access and the declared training budget. A Random relation
control changes the relation without silently changing the data or model.
Supervised physical labels belong to explicitly named oracle references and
probes; they are not unannounced input features for the unsupervised method.

## Formation

Formation examines whether persistent physical properties can be recovered from
learned representations. Use the physical-parameter or invariant probes paired
with each selected source checkpoint. Fit preprocessing, regression and probe
selection on the prescribed training/validation partition. Good task utility
alone does not establish physical-parameter recovery.

## Use

Use tests hold the predictor fixed and change only the declared persistent
context or update. Preserve recipient transient state, observations, actions,
case identities and frozen weights across conditions. Correct, Null and Wrong
donors answer different questions. Decode and Oracle must use the same physical
coordinate convention and training-only normalization.

The Swimmer prospective experiment retains the same predictor and adapter
checkpoints across own-system and cell-matched wrong-system updates. The core
formation/use and CoPhy modules provide their own intervention contracts.

## Value

Prediction quality and closed-loop utility are distinct endpoints. Describe an
offline prediction or decision-cost evaluation as such. An environment replay
or visualization is not evidence of closed-loop benefit. Utility reporting
should remain paired with the appropriate representation probes.

## Splits, selection and aggregation

- Preserve the declared system-disjoint and within-system splits.
- Use validation data for configuration and checkpoint selection.
- Keep sealed-test admission separate from development commands.
- Preserve paired cases when comparing interventions.
- Aggregate and bootstrap at the declared unit, often the physical system.
- Report source seeds and reader/adapter seeds as separate levels.
- Distinguish a new training run from an exact frozen-checkpoint reproduction.

Historical orchestration may check artifact hashes or provenance documents.
These checks are part of the original scientific boundary. Do not fabricate
missing records to make a command pass; generate new artifacts through the
declared preparation and training route or supply the required frozen inputs.

For exact per-benchmark details, follow the source table in the root README,
the [core guide](core.md), and the local benchmark configurations.

