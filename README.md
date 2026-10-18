# EPU-QA1 public-readiness candidate

[![Data-free verification](https://github.com/build-the-future-11/epu-qa1-public-readiness/actions/workflows/verify.yml/badge.svg)](https://github.com/build-the-future-11/epu-qa1-public-readiness/actions/workflows/verify.yml)

This directory is an append-only engineering successor to the frozen E5-R1
package. It is prepared for public source review and data-free verification.
It is **not** an admitted scientific runner and does not authorize another data
opening, consume or reset an attempt ledger, or report a positive result.

The scientific matrix remains fixed: the declared digits payload, grouped
split, 450 query groups, noise levels `0`, `0.25`, and `0.5`, ten declared
methods, and the retained H1/H2 arithmetic. The changes here strengthen failure
handling, artifact durability, receipt validation, and public reviewability.

## Evidence map

| Question | Authoritative public file |
|---|---|
| What is fixed scientifically? | [`SCIENTIFIC_SPEC.md`](SCIENTIFIC_SPEC.md) and [`PROTOCOL.md`](PROTOCOL.md) |
| What does verification cover? | [`TEST_CONTRACT.md`](TEST_CONTRACT.md) |
| What changed and why? | [`CHANGELOG.md`](CHANGELOG.md) and [`REVIEW.md`](REVIEW.md) |
| How are security reports handled? | [`SECURITY.md`](SECURITY.md) |
| How should changes preserve evidence? | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| What do current checks establish? | Engineering source integrity and data-free contract behavior only |
| What remains scientifically unresolved? | Efficacy and confirmation; disposition `NOT_EVALUATED` |

## Verify

Run the bounded public checks from the repository root:

```sh
python3 -B public_preflight.py --output verification/local-unique-id
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

The verifier is dependency-free. Its optional Ruff check runs when `ruff` is
already available; the receipt records an explicit skip when it is not. GitHub
Actions runs Ruff separately before accepting the verifier receipt.

For containment and auditability, `--output` must name a new directory below
`verification/`. Public receipts replace checkout- and run-specific absolute
paths with `<REPOSITORY>` and `<OUTPUT>` markers.

## Evidence boundary

A passing receipt establishes that the named public source manifest compiled,
passed its data-free contract suite, satisfied the static fail-closed checks,
and—when available—passed the pinned Ruff policy. It does not establish a
scientific improvement, independent reproduction, successful exposed-data run,
production suitability, or publication acceptance.

