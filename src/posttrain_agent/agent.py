"""Assemble the deep agent: Backboard LLM + memory + persistence + fallback + approval gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from deepagents import create_deep_agent

from .config import Settings
from .experiments.domain import ExperimentBackend, LocalStubBackend, cost_summary, make_tools
from .experiments.nebius import NebiusBackend
from .llm.backboard_model import Cassette
from .llm.memory import ensure_assistant, memory_tools
from .llm.resilience import build_middleware, build_model
from .observability.events import Journal
from .observability.telemetry import telemetry_tool
from .runtime.budget_guard import BudgetGuard, RunTokenLedger
from .runtime.concurrency import SubagentConcurrencyMiddleware
from .runtime.exit_gate import ExitGate, NotesInjector
from .runtime.persistence import SEED_MEMORY, Runtime
from .subagents import ROLES, build_subagents, load_prompt
from .tools import budget
from .tools import dataset as dataset_mod
from .tools.hf import hf_tools
from .tools.web import web_tools

ORCHESTRATOR_PROMPT = """You are the coordinator of an autonomous post-training engineer.

Goal: improve a target checkpoint within the experiment and cost budget you are given, and only
promote a candidate when the deterministic gate says PROMOTE.

Loop per experiment (delegate each step with the `task` tool, one specialist at a time):
1. benchmark -> 2. failure_analyst -> 3. researcher -> 4. data_curator ->
5. training_designer (must call estimate_budget) -> 6. training_executor (a human approves the
job before it launches; if rejected, revise the plan, do not retry unchanged). Pass the
data_curator's `dataset_ref` AND `sha256` on to the designer and the executor unchanged: the
approval is bound to that sha256, and a launch without it (or after the file changed) is refused. ->
7. evaluator (champion AND candidate on the same split) -> 8. champion_manager.

Parallelism: at most 2 subagents run at once. Issue independent `task` calls together in ONE
message so they run concurrently: (a) in step 3, a researcher literature/prior-art scan alongside
the failure_analyst; (b) in step 7, one evaluator task for the champion and one for the candidate.
Everything else is sequential because each step needs the previous result. Never run two
training_executor tasks at once; one job per experiment.
Plan with write_todos first. Keep specialists' JSON results in files under /work/ so later steps
can read them. Use `recall` before planning and `remember` for durable lessons. Use
`read_telemetry` when calls are slow or failing. Never fabricate numbers: if a tool result is
labelled EXPLANATION, say so in your final report.

What is being improved: a small base model (default Qwen/Qwen3-0.6B, LoRA fine-tuned on Nebius;
base and checkpoints are evaluated by serving them on CPU in a Nebius Sandbox, which takes minutes
per evaluation: wait for results, do not re-issue slow calls) at tool-calling on small deterministic games (tic-tac-toe, Nim, number guessing, rock-paper-scissors).
Success means the held-out score beats the champion's by the gate's margin without regressing; a
rejected candidate is a valid, honest outcome, not a failure to hide.

Task briefs: each `task` call must give the specialist everything it needs and nothing sealed:
the exact checkpoint name, the path of the previous step's result under /work/, and what to return.
Never include held-out or regression task contents in any brief.

Budgets: every subagent has a hard cap of model calls and tokens, and the whole run has a token cap.
A subagent that runs out returns {"status": "BUDGET_EXHAUSTED", ...}. Treat that as a partial result:
read what it saved in /work/, then either continue with a narrower brief or stop and report. Do not
re-dispatch the same task unchanged, and do not loop. Stop after at most 3 experiments, or sooner if
the run budget is nearly spent, and write the final report from what you have.

Proving the win over the base model: the first champion is `base` (the untouched base model).
Evaluate `base` and the candidate on the same held-out and regression splits (same judge), and run
the gate on those. In every experiment's final report show a table: base vs candidate, held-out and
regression scores with `n`, per-game accuracy (`by_game`), `judged_by`, the gate's `z` and decision.
Also state, in words: (a) the test the gate used, exactly as `test` reports it (paired exact McNemar
when both results carry per-task verdicts over the same `tasks_sha256`, otherwise the unpaired
two-proportion z); (b) `mde_80pct_power` and whether the observed gain is below it (the `power`
block's `adequately_powered`, plus `power_note` when present); and (c) the candidate's parse rate. A
gain below the MDE is underpowered, not a win. If a later champion exists, also gate the new
candidate against `base` so the cumulative gain over base is stated, not only the gain over the
previous champion. Say plainly when the evidence is not strong enough to claim a win. Training data
is capped at 1200 rows and is generated by code from the game rules (`generate_game_sft`);
evaluation states never appear in it.
Termination rule (enforced by code): once a training job exists you may NOT finish with a final
message until a gate verdict exists for that candidate (champion_manager ran `run_gate`) or you
called `abort_experiment(reason)`. A final message without either is re-prompted up to 3 times and
then the run is marked INCOMPLETE. Do not write the final report first.

Handling specialist results (never ignore a non-SUCCEEDED status):
| result                                  | do this                                                       |
| training_executor REJECTED              | revise the plan; never retry unchanged                        |
| training_executor BLOCKED               | fix the named cause; if unfixable `abort_experiment(reason)`  |
| training_executor FAILED/CANCELLED      | read its log lines, then `abort_experiment` or revise + relaunch |
| training_executor RUNNING/SUBMITTED     | re-dispatch it WITH the job id (it waits); never launch again |
| evaluator BLOCKED or FAILED             | re-dispatch the evaluator with the SAME job id (or checkpoint)|
|                                         | once; if it fails again `abort_experiment(reason)`            |
| gate decision REJECT                    | valid outcome: report it honestly                             |
Always pass the training job id (`ftjob-...`) and checkpoint name verbatim in later briefs; the
evaluator also accepts `candidate`. Evaluate and gate by job id, not by retyped scores:
champion_manager reads the evaluation ledger itself.

Operator notes: the human may leave notes (`posttrain note`); they arrive as "[operator note]"
messages. Read and follow them before your next step, and say in your report that you did.

Parallelism limit: never more than 2 subagents at once. Never use the `general-purpose` subagent
(it is disabled); only the named specialists.

Resuming: reuse a /work/ artifact (or a training job id) only if the input hashes recorded inside
it (dataset sha256, hyperparameter hash, checkpoint) match the current plan; otherwise redo the
step. Never launch a second training job for the same dataset and hyperparameters (code refuses
a duplicate unless the earlier job failed).

Final report: state the goal, each experiment's benchmark / candidate / gate numbers copied from the
results, the gate decision and its reasons, the gate test used, `mde_80pct_power` versus the
observed gain, the candidate's parse rate, the spend against the budget, and every caveat
(simulated backend, small samples, budget stops). A REJECT is a valid outcome: describe it as a
rejection, never as success and never as if the candidate were promoted.
"""


def default_backend(settings: Settings) -> ExperimentBackend:
    """Nebius Token Factory when a key is configured, otherwise the labelled simulation."""
    return NebiusBackend(settings) if settings.nebius_api_key else LocalStubBackend()


def _live_dataset_sha(settings: Settings, ref: str) -> str:
    """sha256 of the dataset as it is on disk right now, or a readable reason it is unavailable."""
    try:
        report = dataset_mod.verify_file(settings.data_dir, ref, settings.max_dataset_rows)
        return report["sha256"] or f"UNAVAILABLE ({'; '.join(report['errors'][:1])})"
    except Exception as exc:  # noqa: BLE001 - never block the prompt on a formatting error
        return f"UNAVAILABLE ({exc})"


def approval_description(settings: Settings) -> Callable[..., str]:
    """Text a human sees before approving a training job, including the cost estimate."""

    def describe(tool_call: dict[str, Any], state: Any, runtime: Any) -> str:
        args = tool_call.get("args", {})
        hp = args.get("hyperparameters", {})
        if isinstance(hp, str):
            try:
                hp = json.loads(hp)
            except ValueError:
                pass
        try:
            est = budget.estimate(settings, args.get("dataset_ref", ""), hp)
        except Exception as exc:  # noqa: BLE001 - never block the prompt on a formatting error
            est = {"error": str(exc)}
        unpriced = ""
        if isinstance(est, dict) and "error" not in est and est.get("price_per_mtok") is None:
            unpriced = ("UNPRICED: no USD price is configured, so the dollar cost is UNKNOWN. This job "
                        f"is limited only by the cumulative trained-token cap "
                        f"({est.get('spent_trained_tokens')} already used + "
                        f"{est.get('estimated_trained_tokens')} for this job, cap "
                        f"{est.get('max_trained_tokens')}).\n")
        try:
            cost = cost_summary(settings, args.get("dataset_ref", ""),
                                hp if isinstance(hp, dict) else {})["cost_line"]
        except Exception as exc:  # noqa: BLE001
            cost = f"cost UNKNOWN: {exc}"
        live_sha = _live_dataset_sha(settings, args.get("dataset_ref", ""))
        claimed = args.get("dataset_sha256")
        if not claimed:
            bind = "NOT BOUND: the call carries no dataset_sha256, so the launch will be refused"
        elif claimed.strip().lower() != live_sha:
            bind = f"MISMATCH: the call claims {claimed}, the file on disk is {live_sha}; REJECT"
        else:
            bind = "bound: the launch is refused if the file changes after approval"
        return ("Approve launching a fine-tuning job?\n" + unpriced + f"COST: {cost}\n" +
                f"base model: {args.get('base_model') or settings.base_model}\n"
                f"dataset: {args.get('dataset_ref')}\n"
                f"dataset sha256 (live, {bind}): {live_sha}\n"
                f"hyperparameters: {json.dumps(hp, sort_keys=True)}\n"
                f"budget estimate: {json.dumps(est, sort_keys=True)}")

    return describe


async def build_agent(
    settings: Settings,
    runtime: Runtime,
    *,
    experiments: ExperimentBackend | None = None,
    client_factory: Callable[[], Any] | None = None,
    http_client_factory: Callable[[], Any] | None = None,
    limiter: SubagentConcurrencyMiddleware | None = None,
    journal: Journal | None = None,
):
    """Return a compiled deep-agent graph wired to ``runtime``'s checkpointer and store."""
    assistant_id = await ensure_assistant(settings, client_factory)
    settings = replace(settings, assistant_id=assistant_id)
    extra: dict[str, Any] = {}
    if client_factory:
        extra["client_factory"] = client_factory
    if settings.cassette_path:
        extra["cassette"] = Cassette(settings.cassette_path, settings.cassette_mode)

    middleware = build_middleware(settings, **extra)
    remember, recall = memory_tools(settings, assistant_id, client_factory)
    tools = make_tools(experiments or default_backend(settings), settings, journal=journal)
    tools.update(web_tools(settings, http_client_factory))
    tools.update(hf_tools(settings, http_client_factory))
    tools["read_telemetry"] = telemetry_tool(settings.state_dir / "telemetry.jsonl")

    ledger = RunTokenLedger(settings.state_dir / "run_tokens.json", settings.run_max_tokens)
    interrupt_on = None
    if settings.require_approval:
        interrupt_on = {"launch_training": {
            "allowed_decisions": ["approve", "reject"],
            "description": approval_description(settings),
        }}
    return create_deep_agent(
        model=build_model(settings, **extra),
        tools=[remember, recall, tools["web_search"], tools["read_telemetry"],
               tools["abort_experiment"]],
        system_prompt=ORCHESTRATOR_PROMPT + "\n" + load_prompt("mission.md"),
        subagents=build_subagents(
            tools, (remember, recall),
            [*middleware, BudgetGuard("subagent", settings.subagent_max_calls,
                                      settings.subagent_max_tokens, ledger)],
            {r: build_model(settings, provider=p, model=m, **extra)
             for r, (p, m) in settings.role_models.items() if r in ROLES},
            disable_general_purpose=True),
        middleware=[limiter or SubagentConcurrencyMiddleware(settings.max_parallel_subagents, journal=journal),
                    BudgetGuard("coordinator", settings.coordinator_max_calls,
                                settings.coordinator_max_tokens, ledger),
                    NotesInjector(settings.state_dir, journal),
                    ExitGate(settings.state_dir),
                    *middleware],
        backend=runtime.backend,
        memory=[SEED_MEMORY],
        interrupt_on=interrupt_on,
        checkpointer=runtime.checkpointer,
        store=runtime.store,
        name="posttrain-coordinator",
    )
