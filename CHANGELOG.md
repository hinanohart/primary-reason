# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-18

### Added
- T1 Primary Reason Extractor (belief + pro-attitude + causal-role + confidence decomposition of CoT steps)
- T2 Counterfactual Faithfulness metric (delete / paraphrase / negate interventions; exact / lexical / embedding distance)
- T1.5 Swampman Test Battery minimal (causal-history vs no-history variant comparison, with control-baseline correction and sentinel-word filter; bootstrap 95% CI)
- LLM adapter Protocol with anthropic / openai / ollama / mock implementations
- CLI (`primary-reason verify | extract | stb | version`)
- Langfuse plugin (optional, lazy import)
- Pre-commit hooks: ruff, ruff-format, mypy strict, gitleaks, banned-phrase guard, image-URL hardcode guard
- GitHub Actions CI: lint / typecheck / test (py3.11, py3.12) / gitleaks / docs
- PyPI trusted publishing workflow on `v*` tags (OIDC, no token)
- 94 tests including 7 property tests and 20 mock-integration fixtures
- Apache-2.0 license, CITATION.cff for academic use
