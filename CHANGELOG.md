# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-18

Post-release re-audit (3 sub-agent monitor: architect / verifier / critic) surfaced statistical
and parsing issues in v0.1.0. This release fixes them without scope change.

### Fixed
- **STB clipping bias**: ``fpa_score`` is now the signed control-adjusted mean. The previous
  ``max(0, d - ctrl_d)`` clip biased the null distribution upward (false-positive
  ``discriminates`` flag possible under no real effect). ``SwampmanScore.fpa_score`` and
  ``bootstrap_ci`` are now ``Field(ge=-1.0, le=1.0)``.
- **STB ``discriminates`` floor**: now requires ``n_trials >= 20`` in addition to
  ``bootstrap_ci[0] > 0``. The default 5-task battery is exploratory and never
  ``discriminates`` regardless of CI — bootstrap on n=5 has no statistical force.
- **STB sentinel filter**: now tokenises on ``\\w+`` so hyphenated echoes
  ("causal-history", "memory-trace") are caught by the sentinel set.
- **STB bootstrap percentile**: uses ``numpy.quantile`` (linear-interpolated) instead of integer
  index lookup; the previous lookup was anti-conservatively narrow by a fraction of a percent.
- **CoT splitter arithmetic break**: numbered-list split is now anchored to line starts only.
  Previously ``"We compute 17 * 4. Step 1: factor"`` was split inside the math sentence. The
  ``Step N:`` label form still splits inline (the explicit colon is contextually distinctive).
- **JSON wrap drift**: ``_extract_json`` now wraps top-level JSON lists as
  ``{"primary_reasons": [...]}`` (matching the schema) instead of ``{"_value": [...]}``;
  previously a model returning the natural ``[ {step_index: 0, ...}, ... ]`` form caused the
  extractor to silently fall back to ``(extraction_failed)`` stubs.
- **Layered retry cost**: extractor's own retry loop default lowered from 3 to 1; the adapter
  already retries transport errors via tenacity. Worst-case verify is now N×1 instead of N×9.
- **Honest README**: removed false claim that the package ships a Davidson-vs-naive ablation
  (it does not). Ablation harness is moved to v0.2.0 roadmap. PyPI install hint now points to
  the GitHub Release wheel until trusted-publisher provisioning is complete.

### Added
- 7 new regression tests for the fixes above (splitter arithmetic preservation, signed STB
  score, hyphenated sentinel, top-level JSON list wrap, minimum-n discriminates gate, etc).

### Documentation
- ``ReasonCauseVerifier.cache_dir`` and ``max_concurrency`` are now documented as
  forward-compatibility placeholders (accepted but not yet wired into the request path).

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
