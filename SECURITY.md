# Security policy

This package intentionally executes a digest-pinned local reference module.
Treat changes to the reference digest, source loader, path containment,
receipt parser, artifact hashing, or subprocess environment as security-sensitive.

Do not publish credentials, private datasets, host paths, or mutable download
URLs in an issue. Report a suspected vulnerability privately to the repository
owner through GitHub's private vulnerability reporting feature when available.

Only the latest tagged `0.x` release is supported. Scientific validity issues
should be reported separately from software-security issues and must preserve
the affected receipt and source identity.

