# Verification of this source distribution

The following checks were performed while preparing version 0.1.0:

| Check | Result |
|---|---|
| Core and formation/use tests | 59 passed; optional native integrations require their documented inputs |
| CoPhy and native Spring synthetic integration | 34 passed |
| CoPhy multiseed wrapper/source checks | 18 kind/seed bindings passed; J2 source CLI and three reader transforms checked |
| Additional bound Spring contrastive integration | 2 passed |
| Core CPU learning/probe example | Passed: finite objective, gradients and analytic physical-parameter probe |
| RH20T video preprocessing and split checks | 6 passed |
| FHN numerical tests | 5 passed |
| Pendulum reference equivalence | 200 resets and 3,600 steps; zero observation discrepancy, maximum reward discrepancy below 1e-15 |
| DALI/FCRL additional adapters | Finite gradients, CoPhy shape/padding, bitwise Adam/RNG continuation and paired raw-MSE checks passed; 40 source bindings checked |
| Swimmer numerical and protocol tests | 99 passed; 4 tests requiring omitted historical fixtures skipped |
| Swimmer MuJoCo simulation smoke test | Passed |
| Overcooked dependency bootstrap and source bindings | Fixed upstream archive and patch verified; 10 training-source bindings checked per variant |
| Overcooked numerical smoke checks | Different-identity pairing, fit-only ridge probes and fixed-query history intervention passed |

Optional tests that require unavailable native sources, trained checkpoints or
large datasets are distinct from these passing checks. Source compilation,
configuration parsing, relative-document-link checks and anonymous-content
checks are performed on the final assembled archive.

The tested core package versions are recorded in
[`requirements/tested-core.json`](../requirements/tested-core.json). This records
the validation environment, not a claim that all benchmarks share one dependency
lockfile.

These checks validate implementation contracts and source distribution
integrity. They do not certify a new full-budget training run, every optional
dependency combination, or exact reproduction of all reported paper numbers.

## Version 0.1.1 presentation and documentation update

This update changes the gallery, release metadata, and documentation. Training
implementations and experimental configurations are unchanged from 0.1.0. The
checks above are the earlier implementation checks, not a new full training run.

The update adds twelve equal 720 × 480 SVG frames, verifies that their embedded
images match the included PNG sources, checks image metadata and relative
Markdown/HTML links, and regenerates the source-integrity manifest. The public
download is checked against that manifest after publication.

## Shared environment gallery — 2026-09-26

The gallery now imports thirteen thumbnails from the frozen shared environment
media registry, including Pendulum. All thirteen PNG files and the images
embedded in their SVG frames match the registry hashes. Local rebuilding is
deterministic. Registry version and SHA-256 are recorded in
[`gallery.json`](../assets/environments/gallery.json).

Image metadata, relative links, anonymous-content patterns, and the refreshed
source-integrity manifest were checked. These presentation changes leave
training implementations, experimental configurations, and numerical results
unchanged; the earlier implementation checks above remain the relevant record.

## Version 0.1.2 privacy and attribution update

Nineteen implementation/configuration files received only declared string
replacements for internal execution identifiers, protected-scope labels and
launcher help text. Python syntax trees and JSON structures match the previous
version after those exact substitutions; scientific numbers, booleans and
conditions are unchanged. The generated CLI reference also removes the launcher
brand and adds the existing gallery-import argument that its prior snapshot
omitted. Nineteen synthetic configuration-tampering cases
remain rejected before and after the change. Three isolated receipt-integrity
checks cover valid historical markers, changed hashes and unexpected output
files. These checks use synthetic fixtures, not omitted historical run data.

All Python sources are parsed, all JSON files are decoded, relative documentation
links are checked and the source manifest is regenerated for this version.
The gallery keeps the thirteen existing thumbnail bytes. Media credits record
documented upstream licenses and transformations while distinguishing
source-code terms from upstream media rights.

The full Torch-dependent suite and full-budget training were not rerun for this
metadata change. Historical source hashes are preserved, so the normalized
distribution must not be passed off as the original frozen execution inputs.
See [source privacy and provenance boundaries](source_privacy.md).

## Version 0.2.0 public release — 2026-10-06

The public release starts from the integrity-verified 0.1.2 source distribution.
All 670 scientific/other Python files and all scientific JSON configurations
retain their 0.1.2 bytes. The gallery builder is the sole changed Python file:
it retains CoPhy as text-only guide links and skips its media on registry imports.
Other changes are public documentation, citation metadata, package version,
project URLs, original-contribution copyright naming, and the removal of six
CoPhy gallery media files.

Checks performed for this release:

- Editable installation in a new Python 3.13 environment with NumPy 2.5.3 and PyYAML 6.0.3.
- `python examples/dclean_pairs.py`: independent-interaction example, disjoint system splits, and collision-free wrong relations passed.
- `python -m pytest -q tests/core/test_preflight.py tests/core/test_pokeworld_ood.py tests/core/test_pokeworld_interaction_mechanism.py`: **14 passed** with pytest 9.1.1. One collection warning comes from the imported `TestSealedError` exception class; it is not a failed test.
- All 671 Python source files parsed; all JSON files and package TOML parsed.
- Relative Markdown target paths resolve; `CITATION.cff` parses and contains the seven authors in the paper order.
- Gallery rebuilding is deterministic, preserves all retained image bytes, and leaves no references to the removed CoPhy images. Registry import also preserves the three text-only entries.
- The regenerated release manifest verifies every published file listed within it.

These are package and CPU implementation checks. Torch-dependent tests,
framework-specific optional integrations, historical checkpoint analyses, and
full-budget training were not rerun for this documentation-only publication update.
