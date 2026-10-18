# Public-readiness code review

## Outcome

The inherited E5-R1 implementation is a strong evidence-preserving local
controller, but its frozen package is not a portable public runner. This
successor keeps the scientific computation and historical receipts intact and
addresses the code defects that can be fixed without opening data or changing
the study.

## Repairs applied

- All claim-bearing vector and replay `zip` operations now use `strict=True`;
  length mismatches fail instead of silently truncating arithmetic.
- The independent replay dot product now checks dimensions explicitly.
- `fit_class_means` no longer constructs a configuration object at function
  definition time and validates an explicitly supplied configuration.
- Raw JSONL is flushed and `fsync`ed before manifests are calculated.
- Artifact manifests use the bounded, stable-file hashing helper instead of
  unbounded `Path.read_bytes()` calls.
- `KeyboardInterrupt` and `SystemExit` are recorded as `INTERRUPTED` and then
  propagated. Ordinary failures are caught as `Exception`; the runner no
  longer swallows process-control exceptions as ordinary study failures.
- Completed-run verification can bind the expected engineering revision and
  recovery version. Recovery runs must include a unique ledger path and the
  complete 31-sample, 30-second, 1 GiB admission receipt.
- Unused imports and avoidable lint defects were removed from claim-bearing
  source.

## Deliberately unresolved execution gates

- `recovery_budget.py` still names the exact R1 native package and predecessor
  receipt identities. The canonical state is resolved relative to the checkout,
  but these retained R1 bindings are evidence rather than new execution authority.
- The resource sampler is Darwin-specific. Unsupported platforms must refuse
  until an independently reviewed adapter provides equivalent free-memory,
  pageout, and swap semantics.
- The copied runner cannot pass its original freeze or native-install checks
  from this directory. That is intentional: review completion is not execution
  authority.
- The internal trajectory replay tests arithmetic consistency. It is not an
  independent reproduction or confirmatory analysis.

The exact digest-pinned reference implementation and predecessor compatibility
fixtures are bundled under `reference/` and `tests/fixtures/`, so public checks
do not depend on the original developer workstation layout.

## Claim boundary

Passing public preflight establishes source integrity and generated-fixture
contract behavior only. It does not establish efficacy, a positive H1/H2
outcome, external reproducibility, or publication readiness of a scientific
claim.

