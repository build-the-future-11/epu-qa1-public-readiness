# Contributing

Use Python 3.11 or newer. Run the complete data-free gate before proposing a
change:

```sh
python3 -B public_preflight.py --output verification/local-unique-id
```

Keep scientific and engineering changes separate. A pull request that changes
the dataset, split, noise schedule, methods, thresholds, statistics, resource
budget, or attempt budget must create a clearly versioned protocol successor.
It must not rewrite an existing freeze, ledger, receipt, negative outcome, or
resource refusal.

Every change should state:

- which claim-bearing files changed;
- whether the scientific matrix changed;
- which generated or contract tests cover it; and
- the exact verification receipt produced from the proposed source.

Do not add datasets, credentials, machine-specific paths, private receipts, or
unredacted host diagnostics to this repository.

