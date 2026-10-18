# EPU-QA1 public-readiness candidate

This directory is an append-only engineering successor to the frozen E5-R1
package. It is prepared for public source review and data-free verification.
It is **not** an admitted scientific runner and does not authorize another data
opening, consume or reset an attempt ledger, or report a positive result.

The scientific matrix remains fixed: the declared digits payload, grouped
split, 450 query groups, noise levels `0`, `0.25`, and `0.5`, ten declared
methods, and the retained H1/H2 arithmetic. The changes here strengthen failure
handling, artifact durability, receipt validation, and public reviewability.

Run the bounded public checks from the repository root:

```sh
python3 -B state/epu-public-readiness-20261018/public_preflight.py \
  --output state/epu-public-readiness-20261018/verification/local
```

The command compiles source, checks fail-closed AST invariants, runs generated
and contract fixtures only, and writes `RECEIPT.json`. It never opens the digits
payload or an efficacy result. If `ruff` is installed, the command also runs the
pinned lint rule set.

Before any public exposed-data execution, a separately reviewed release must:

1. bind every admitted file in a new immutable freeze;
2. name a portable authority input instead of a developer-specific absolute
   path;
3. install the candidate at the exact bound source identity;
4. establish a platform-specific resource guard (the inherited guard is
   reviewed only for Darwin/macOS);
5. explicitly state the remaining attempt budget and ledger lineage; and
6. receive canonical authorization without modifying the E5 or E5-R1 history.

A development H2 pass would only nominate a fresh confirmatory design. A
failure remains a reportable result and must not be tuned away.

