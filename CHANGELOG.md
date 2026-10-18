# Changelog

## 0.1.1 - 2026-10-18

- Corrected the fresh-clone verification command and removed stale references
  to private workspace-only evidence files.
- Hardened source hashing against symlink and path-replacement races.
- Made verification receipts durable and explicitly record interruptions.
- Added a canonical manifest digest and regression tests for the new verifier
  guarantees.
- Pinned CI actions to immutable commits, disabled persisted checkout
  credentials, bounded job time, and set explicit artifact retention.

## 0.1.0 - 2026-10-18

- Added a self-contained, data-free public verification command.
- Bundled the digest-pinned reference and predecessor compatibility fixtures.
- Added strict length checking throughout claim-bearing vector arithmetic.
- Hardened raw artifact durability, stable hashing, interrupt receipts, and
  recovery provenance validation.
- Added 84 generated and contract tests plus bounded lint and compile gates.
- Preserved the E5 and E5-R1 scientific history and attempt ledgers unchanged.

