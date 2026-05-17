# Security Policy

## Reporting a vulnerability

If you find a security issue in `primary-reason`, please **do not open a public issue**.

Use one of the following private channels:

1. **GitHub private vulnerability reporting**: open a private advisory at
   https://github.com/hinanohart/primary-reason/security/advisories/new
2. **Email**: contact the maintainer via the email address listed in the
   `CITATION.cff` author block.

We will acknowledge receipt within 7 days and aim to publish a fix within 30
days for confirmed reports, faster for actively exploited issues.

## Scope

In-scope:
- Dependency-introduced vulnerabilities (e.g. tenacity / pydantic / openai SDK)
- Adapter implementations leaking API keys to log streams / process listings
- Prompt-injection paths in the extractor or STB that allow an attacker
  controlling the CoT to escape JSON-mode constraints
- Distribution artifact tampering (wheel / sdist content not matching
  the tagged source)

Out-of-scope:
- The Swampman Test Battery score being gamed by prompt engineering. The
  metric is acknowledged to be gameable in the README; the control baseline
  blunts but does not eliminate the attack. Submit findings as research, not
  as a vulnerability.
- The Davidson primary-reason extractor producing implausible decompositions
  on adversarial CoTs. This is a quality issue, not a security issue.

## Supply chain

- Published wheels are built via GitHub Actions OIDC trusted publishing
  (see `.github/workflows/release.yml`). They are **not** signed with
  developer keys.
- `pre-commit` + `gitleaks` are run on every PR and on every push to `main`.
- All dependencies are pinned to compatible ranges in `pyproject.toml`. We
  do not run automated dependency upgrades during a release window.

## Disclosure timeline (template)

| Day | Action |
|-----|--------|
| 0 | Report received |
| 7 | Triage complete, severity assigned |
| 30 | Fix released or extension requested with reason |
| 30 + 30 | Public advisory + CVE filed (for confirmed reports) |

Thank you for helping keep `primary-reason` honest.
