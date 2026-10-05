# Source privacy and provenance boundaries

This source distribution uses neutral scope labels and stable case names for
execution metadata. In the Swimmer configuration, `protected_scope_1` and
`protected_scope_2` are distinct resources outside the declared experiment.
Their access and mutation restrictions remain in force; these labels do not
refer to a public account, host, or dataset.

Entries in `failed_parity_jobs` use public case aliases formed from the
execution revision and scientific seed. They are not original launcher IDs.
The earlier failures, numerical compatibility conditions, frozen-artifact
hashes and the fact that no scientific artifacts were produced remain recorded.

The `paper_c` import namespace, relative `protocol/`, `data/` and `runs/` paths,
and frozen protocol filenames remain for compatibility with the included
implementation. They are not author home directories. Historical artifact
hashes remain historical commitments: this normalized source distribution is
not byte-identical to those original private execution inputs. A frozen-input
check must continue to reject an input with a different hash. New runs require
their own identified configuration and provenance; this release does not
invent replacement historical receipts or silently accept changed inputs.

Some programs write local runtime receipts or resource locks containing the
current hostname, process ID, paths, software versions and selected computing
environment settings. The source distribution does not include those generated
receipts. Their existence is not webpage visitor tracking. Review and remove
identifying values before sharing any future run output; do not disable machine
ownership checks or locks merely to suppress this information.

The two dependency-preparation scripts fetch explicitly requested public
upstream sources. Source-file privacy does not describe the separate anonymous
hosting platform's cookies, analytics or server logs. A new public snapshot
must be downloaded and checked after publication; a clean local tree alone
is not publication evidence.
