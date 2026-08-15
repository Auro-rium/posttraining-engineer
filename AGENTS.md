# Repository Instructions for AI Agents

These rules apply to every change in this repository.

## Start-of-run checklist

1. Read `Decisions.md` before choosing an architecture or implementation approach.
2. Read the relevant portions of `Flow.md` before changing control flow, state, APIs, or events.
3. Review the latest `ChangeLog.md` entries.
4. Inspect the current repository and working tree; preserve unrelated work.

## Change discipline

- Keep this repository scoped to the backend hackathon demonstration. Do not add a visualizer, user accounts, billing, or unrelated production-platform features.
- Keep secrets, tokens, private prompts, and held-out task contents out of source, tests, logs, telemetry, and the living documents.
- Treat held-out evaluation data as sealed: it must not enter RAG, training data, agent prompts, or repair generation.
- Use deterministic code—not an LLM—to verify repairs, enforce budgets, calculate evaluation metrics, and decide checkpoint promotion.
- Label evidence as `LIVE`, `PRIOR_VERIFIED_RUN`, or `EXPLANATION`; never present fixtures or simulated values as live results.
- Add or update tests in proportion to each behavioral change.

## End-of-run checklist

1. Update `Flow.md` when a function, API, event, state transition, dependency, or failure path changes.
2. Append a decision to `Decisions.md` for meaningful product or technical choices. Never rewrite an old decision; supersede it with a new ID.
3. Run relevant tests and `python backend/scripts/check_docs_sync.py`.
4. Append one accurate entry to `ChangeLog.md` after verification. Record failures and unverified work honestly.
5. Confirm the implementation, tests, and living documents agree before reporting completion.

Runs that only inspect the repository and make no changes do not need a changelog entry.
