# EPU-QA1 development protocol

Specified before implementation and new data access, 2026-10-15. This protocol
permits only exposed-data development, after resource and implementation checks.
There is no confirmatory execution mode. Historical EPU sources, frozen protocols
and results remain intact. Any changed scientific choice requires a new version.

## Data and split

Use the installed scikit-learn digits.csv.gz (1,797 rows, 64 integer pixels 0–16
and one integer class 0–9). This is the UCI optical handwritten digits test subset,
not the complete 5,620-row UCI corpus. Source: Alpaydin and Kaynak (1998),
[UCI DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49);
[official description/license](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits),
CC BY 4.0. [scikit-learn loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
and the installed loader identify the subset. Credit these sources in any export.

The installed wheel RECORD declares the compressed file SHA-256
`09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22`
and 57,523 bytes. This is a metadata declaration until runtime verifies the file;
no payload has been opened while preparing the protocol. No download/install is
needed. All 1,797 rows are treated as historically exposed development data;
neither a new seed nor a new split makes them untouched confirmation. Old raw
result files are not inputs to this runner.

Stable ID is digits:<zero-based CSV row>. Group identical raw pixel tuples before
splitting; conflicting labels within a duplicate group invalidate the export.
Order groups by SHA-256 of `EPU-QA1/split/` plus their comma-joined integer pixels,
breaking a hash tie by first row ID. Fill the query set by whole groups until at
least 450 rows are assigned; all other groups form training. Both sets must have
every class represented. No row or feature group may overlap. No group is dropped.
For each query group evaluate its lowest row ID once and retain all aliases.
Metrics weight each distinct raw image once. This removes exact duplication, not
writer dependence. Writer IDs are unavailable in this file, so population-level
independence and generalization remain unestablished.

Dry run: first 32 distinct query groups in the fixed order, noise=0 only, all
methods, plus native trajectories for these 32. This is real-data engineering
verification and exploratory exposure, never a substitute for the full matrix.
Development: all query groups, noise standard deviations 0, .25, .5, clip to
[0,1], then center/normalize. For every query/noise pair initialize Python
Random with the integer SHA-256 of `EPU-QA1/noise/<row_id>/<noise>`; generate 64
gauss(0,noise) draws. The same corrupted query feeds every method. This is a new
development schedule, not a reproduction of NumPy's original noise stream.

## Fixed matrix and outputs

1. Direct nearest prototype (same ten class means).
2. Original native source, eight steps, alpha=.6, beta=6, threshold=0.
3. Original native source, one step.
4. Original native source, uniform weights, eight steps.
5. Candidate anchor lambda=.1, eight steps (primary candidate).
6. Candidate anchor lambda=.1, one step (iteration ablation).
7. Candidate anchor lambda=.1, eight steps with alpha=0 (mixing ablation).
8. Direct prediction repeated eight times (budget-only control; wasted work is
   labeled as such, never presented as a strong alternative algorithm).
9. Full training-bank nearest neighbor with centered unit vectors, k=1.
10. Same bank, k=3, majority vote; ties use smaller class label. Rank neighbors by
    descending dot product then ascending row ID. These stronger alternatives
    use more memory; their gains cannot be attributed to recurrence.

No tuning, encoder, whitening, new loss, threshold calibration, adaptive stopping,
temperature grid or post-hoc subgroup is included. The model has zero trained
free parameters beyond stored data summaries. Scalar configuration counts, bank
bytes, score/dot counts and elapsed method time are reported separately.

Retain one JSONL record for every query/noise/method: true development label,
prediction, initial prediction, exact/near ties when applicable, elapsed time,
state norm, query displacement, and operation accounting. For first 32 clean
queries retain unchanged-source diagnostic traces plus candidate trajectories.
Save prototypes/config as a checkpoint, split/alias membership, data/code/native
source hashes, environment, resources and all failures. Save a CSV summary and
regenerable Markdown table. A failed partial run is not analyzed as complete.

Primary development endpoint: mean of per-noise accuracies (equal weighting),
anchor8 minus direct. Secondary: each noise's accuracy, prediction changes,
winner concentration, harmful/beneficial changes, measured timings and memory.
Uncertainty is deliberately not a population confidence interval: report the
finite-cohort paired discordant counts and a sensitivity envelope across the
three declared corruption conditions. There are no independent seeds or valid
writer clusters here. A significance claim would require a separately powered
and genuinely protected study. No p-value or certainty can be manufactured by
treating noise replicas as independent subjects.

H2's 1 percentage point screen is an operational development target, not evidence
that this sample has power for that effect. If it fails, retain the result and
close QA1's utility branch. If it passes, nominate it for a fresh confirmatory
design; do not call it supported research. If direct is stronger, direct remains
the preferred classifier on the tested cohort. H1 and H2 are reported separately.

## Budget, admission and failure rules

One data-opening dry-run attempt and one full development attempt. A failure after
opening data consumes its attempt; any scientifically affected retry requires a
versioned amendment, never silent repetition. Pre-data resource refusals do not
consume the data-opening budget but each is retained. Maximum wall time 600 s per
attempt, single process, no paid compute, installs, GPU or concurrent benchmark.
No further automatic search is authorized by this protocol.

Before data access: verify the package freeze, original native source, protocol
role and resource guard. Two samples at least one second apart must both have
>=512 MiB free+speculative memory, unchanged pageouts and nonincreasing swap use.
Check the same guard every 25 queries and at finish. Failed measurement refuses;
there is no override. Do not stop unrelated applications to create headroom.

Before full development: require a complete dry-run receipt with the same freeze,
data hash and valid outputs. Metadata-only success never grants RUN READY. An
exclusive output directory and flock-protected attempt ledger prevent overwrites
and concurrent attempts. Interruptions/errors retain partial raw output and a
nonzero exit. Regeneration verifies the raw hash before summarizing it.

## Confirmation and release boundary

The intended future confirmation must use separately provenance-reviewed,
untouched data with writer grouping where available, predeclared endpoints,
practical effect/cost thresholds and a power/precision calculation using only
development information. It must compare the selected model with direct and the
strongest appropriate feature/memory-matched baselines and report all outcomes.
Dataset identity, writer IDs and sample-size justification are currently absent.
Consequently CONFIRMATORY_PROTOCOL_FROZEN, VERIFIED and RESEARCH_COMPLETE cannot
pass in this branch today. No confirmatory data is acquired or opened by this
package. A positive development screen does not relax that boundary.
