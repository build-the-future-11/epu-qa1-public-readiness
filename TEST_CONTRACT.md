# Executable E5-R1 engineering test contract

Run from the package directory:

```sh
rtk proxy python3 -B -m unittest discover -v
```

The final default-discovery run passed 80 small stdlib tests. No real dataset was
opened. Native-reference parity cases read only the pinned original Python source;
the staged wrapper resolves that path explicitly, with `EPU_REFERENCE_ROOT` as an
optional relocation setting. Production reference hash checking is unchanged.

| Current suite | Cases | Evidence covered |
|---|---:|---|
| `test_recovery_contract.py` | 29 | Sustained sampled resources, historical/shared budget integrity, clone refusal, lock contention, canonical pause during admission, dry compatibility, no retry, synchronization failure, generated full control flow, query/finalization deadlines |
| `test_numerical_contract.py` | 7 | Unchanged E5 normalization and supported beta domain |
| `test_retained_science.py` | 27 | 14 unchanged model cases, five data/matrix cases, eight endpoint/statistics cases selected explicitly from retained legacy sources |
| `test_completed_bundle_contract.py` | 17 | Six unchanged accumulator cases and eleven completed-bundle integrity/regeneration cases adapted to call the unchanged completed-run validator |

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

The previously frozen staged revision is preserved in `staged_v1/`, including its
freeze `3c2ee69d387212236f3052bf2b81b23b6b578b84dbf59b960909bb944ccba9cc`,
23-test receipt and original copied tests. The first new default-discovery run
reported two source-path errors because old parity fixtures inferred a native
repository from the staged file's parent directory. `TESTS.discovery.log` retains
both errors. The wrapper's explicit pinned-source resolution fixed this fixture
location defect; `TESTS.discovery.final.log` retains the passing 80-test run.
No production recurrence, dataset logic or recovery control flow changed during
this test-contract revision.

These checks establish bounded implementation behavior. They do not establish
real R1 execution, predictive improvement, independent reproduction or confirmation.
