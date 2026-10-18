# Public verification test contract

Run from the package directory:

```sh
python3 -B public_preflight.py --output verification/local-unique-id
```

The v0.1.1 public gate passes 88 small standard-library tests. No real dataset
was opened. Native-reference parity cases read only the bundled, digest-pinned
reference source. `public_preflight.py` supplies the isolated import and fixture
environment used by the contract, so raw discovery without that environment is
not the supported complete gate.

| Current suite | Cases | Evidence covered |
|---|---:|---|
| `test_recovery_contract.py` | 29 | Sustained sampled resources, historical/shared budget integrity, clone refusal, lock contention, canonical pause during admission, dry compatibility, no retry, synchronization failure, generated full control flow, query/finalization deadlines |
| `test_numerical_contract.py` | 7 | Unchanged E5 normalization and supported beta domain |
| `test_retained_science.py` | 27 | 14 unchanged model cases, five data/matrix cases, eight endpoint/statistics cases selected explicitly from retained legacy sources |
| `test_completed_bundle_contract.py` | 17 | Six unchanged accumulator cases and eleven completed-bundle integrity/regeneration cases adapted to call the unchanged completed-run validator |
| `test_public_hardening.py` | 8 | Public manifest exclusions, link refusal, exclusive durable receipts, interrupt classification, strict vector alignment, and configuration validation |

The original E5 test files remain byte-identical under `legacy_test_*.py`. They
are not default-discovered as standalone R1 suites because their old dry-run CLI,
temporary-package admission and `verify_dry_run(path, freeze)` interface no longer
describe the R1 contract. This is explicit migration, not deletion of failing
scientific tests. The copied numerical test file remains directly discoverable.

The 19 old orchestration cases not run verbatim are replaced as follows:

| Legacy source/cases | Reason and replacement |
|---|---|
| `legacy_test_epu_query_anchor.ExperimentTests`: resource refusal, measurement failure, data-failure consumption, full dry prerequisite | Old temporary-package/dry CLI replaced by exact-native full-only R1 admission. R1 resource, measurement, post-reservation, clone and dry-bundle tests exercise the new path. |
| Same class: raw regeneration/tampering, confirmation/bad IDs, source-freeze tampering, dry tampering/metadata-only | Integrity assertions now run through completed-bundle cases, scientific-source/runner compatibility mutations, and explicit R1 no-new-dry rejection. Production mode/run-ID validation remains in `run_study.py`. |
| `legacy_test_prerun_integrity`: expired admission, last-query timeout, durable reservation, sync failure, corrupt phase, regenerated dry phase | Replaced by R1 expired-window, last-query, post-reservation/sync-failure and malformed-ledger cases; adapted complete-bundle test retains dry-phase regeneration. |
| `legacy_test_result_contract`: mocked dry-run integration | R1 has no new dry mode. Replaced by generated full-loop integration using all ten methods and three noise conditions, plus separate complete-bundle tests. |
| `legacy_test_run_completion`: final hash deadline and fresh admission | Replaced by R1 final-hash deadline, blocked sustained admission, and pause-during-admission cases. |
| `legacy_test_staged_integrity`: two E5-specific shared-budget freeze cases | Replaced by R1 immutable binding, historical-ledger identity, clone/shared-ledger and expanded-budget refusal tests. |

The original E5 compatibility tests and scientific sources are retained under
`tests/legacy_test_*.py` and `tests/fixtures/predecessor/`. Their historical
orchestration interfaces are preserved for auditability, while the adapted
public tests exercise the current completed-bundle and recovery contracts.
No production recurrence, dataset logic, scientific matrix, or recovery control
flow is changed by this public test wrapper.

These checks establish bounded implementation behavior. They do not establish
real R1 execution, predictive improvement, independent reproduction or confirmation.
