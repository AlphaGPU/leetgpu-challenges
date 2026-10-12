# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases, and starter templates for multiple GPU programming frameworks.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for each supported framework

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## PR review bot

`.github/workflows/review-bot.yml` runs on every non-draft PR (opened, synchronize, reopened, ready_for_review). The `checks` job runs the same checks as CI (black/isort/flake8, clang-format, `mojo format` when `.mojo` files change, `python scripts/validate_challenges.py`). Then the bot decides:

1. **Deterministic rules** (`.github/review-bot.yml`, easy to tune). Needs a human if the PR touches `.github/**`, `scripts/**`, `challenge_base.py`, runner/grader/checker code, dependency files, `.clang-format`, `.flake8`, `.pre-commit-config.yaml` or `CLAUDE.md`; edits an **existing** challenge's `challenge.py` (its `reference_impl` is the grader); adds more than one new challenge; adds unexpected files under `challenges/`; renames or deletes files; changes more than 400 counted lines; or any check fails. In a **new** challenge folder only `challenge.py` counts toward the limit (html and the 6 starters are boilerplate).
2. **LLM review** of the diff (only if the rules pass), with the list of existing challenges so it can catch duplicates. For new challenges it checks the quality bar from `CLAUDE.md`. Diff over 80k chars ⇒ not safe. The diff is treated as untrusted data; the model must return strict JSON `{safe, risk, reasons}`.

Safe + low risk ⇒ **APPROVE** review. Anything else ⇒ COMMENT review with reasons, review requested from `ishaan-arya` and `shxjames`, `needs-human-review` label (created if missing). Missing key, failed call, or invalid JSON ⇒ human review (fail closed).

The review script and config run from the **base** branch and PR code is never executed in the job that has secrets. Fork PRs are skipped by the review job (no secrets / write token on forks), so they need a normal human review.

### Setup
- **Setting:** Settings → Actions → General → Workflow permissions → *Allow GitHub Actions to create and approve pull requests*.
- **Secret** `REVIEW_LLM_API_KEY`; **variables** `REVIEW_LLM_PROVIDER` (`openai` for any OpenAI-compatible API, or `anthropic`), `REVIEW_LLM_MODEL`, optional `REVIEW_LLM_BASE_URL`.
- **Optional GitHub App identity:** secrets `REVIEW_APP_ID` and `REVIEW_APP_PRIVATE_KEY`.
- Unit tests: `node --test .github/review-bot/`.

## License

This problem set is licensed under the [CC BY-NC-ND 4.0 license](LICENSE).

Copyright 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.
