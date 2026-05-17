# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- T1 Primary Reason Extractor (belief + pro-attitude decomposition of CoT steps)
- T2 Counterfactual Faithfulness metric (delete / paraphrase / negate interventions)
- T1.5 Swampman Test Battery minimal (causal-history vs no-history variant comparison)
- LLM adapter Protocol with anthropic / openai / ollama / mock implementations
- CLI (`primary-reason verify | extract | stb`)
- Langfuse plugin (optional)
