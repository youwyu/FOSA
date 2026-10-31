"""LLM-CL: the LLM-only curriculum baseline (paper appendix, Baseline Implementation Details).

At each step the LLM sees the robot's capability context and the history of proposed tasks with their training
outcomes, and proposes the next task, which is compiled into an environment and trained with PPO.

usage: python -m fosa.discovery.curriculum --robot ROBOT [--iterations N] ...
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fosa import llm, utils
from fosa.compiler import contracts, generate_env
from fosa.discovery import capability_context, common, objectives
from fosa.sim import registry

METHOD = "curriculum"
MODULE_PREFIX = "curr"  # module names reach the prompt through the history
SYSTEM_PROMPT = (
    "You design feasible robot reinforcement-learning curricula for the active simulator backend. Output JSON only."
)
PROPOSAL_ATTEMPTS = 3
REJECTED_PROMPT_ROWS = 20  # rejected proposals kept in the prompt so the model stops repeating them
REJECTED_BATCH_LIMIT = 4  # give up a batch after this many rejections per slot and train what compiled
# Every other method optimizes the capability-frontier score, so LLM-CL is told the same objective.
OBJECTIVE = (
    "Propose the next task at the robot's capability frontier: one it cannot do yet "
    "(success_rate near 0 at the start of training) but can learn within the budget "
    "(success_rate between 0.1 and 0.9 at the end). Tasks already solved, and tasks that "
    "stay at zero, are both worthless."
)
INSTRUCTIONS = [
    "Propose exactly one next concrete robot task.",
    "Use only backend-supported capabilities.",
    "Select task_id from registered_task_adapters and keep its execution semantics consistent with the description "
    "and rationale. The selected adapter is the execution substrate; downstream generation will not silently replace "
    "it.",
    "Describe the task goal and only the objects, support, terrain, or spatial constraints physically needed to "
    "execute it. Do not add scenery or props merely for novelty.",
    "Cross-embodiment tasks are allowed when deliberately requested and backed by the selected adapter; never "
    "introduce locomotion, manipulation, or a free base as an implicit fallback.",
    "Vary physical context only when it changes the learning objective or is supported by recent empirical outcomes.",
    "Return JSON only.",
]
MANIFEST_INSTRUCTION = (  # third instruction of the most detailed prompt, which has the manifest
    "Use the simulator_capability_manifest as the concrete simulator contract. Do not invent physics mechanisms, "
    "reset domains, observables, assets, or interaction effects absent from that manifest."
)
PROPOSAL_SCHEMA = {
    "task_id": "exact key from registered_task_adapters that will execute this task",
    "execution": "must exactly match registered_task_adapters[task_id].execution",
    "description": "natural-language concrete task description compatible with the backend generator",
    "module_name": "short_snake_case_name",
    "rationale": "why this is the right next task",
    "expected_difficulty": "number from 0 to 1",
    "family_hint": "one backend-supported task family if known",
}


@dataclass(kw_only=True)
class CurriculumRecord(common.TaskRecord):
    task_id: str | None
    proposed: dict[str, Any]  # the LLM's proposal
    primitive_hint: str | None = None
    family_hint: str | None = None


def registered_task_adapters(robot: str) -> dict[str, dict[str, str]]:
    return {
        name: {"primitive": spec.primitive, "execution": spec.execution, "description": spec.description}
        for name, spec in sorted(registry.TASKS.items())
        if spec.robot == robot
    }


def prompt_payload(history: list[dict[str, Any]], args: argparse.Namespace, level: int) -> dict[str, Any]:
    """The prompt's context at a detail level: 0 has everything; 1 drops the robot profile, the simulator manifest
    and the task families; 2 also the capability lists. The history shrinks from 20 tasks to 10 and 5."""
    backend, context = args.backend_spec, args.capability_context
    payload: dict[str, Any] = {
        "backend": backend.backend,
        "robot": backend.robot,
        "robot_display_name": backend.robot_display_name,
        "generator_robot_name": backend.generator_robot_name,
        "registered_task_adapters": registered_task_adapters(backend.generator_robot_name),
        "backend_supported_families": context.get("backend_supported_families"),
        "objective": OBJECTIVE,
        "history": history_for_prompt(history, max(2, 20 >> level)),
    }
    if level <= 1:
        payload["backend_capabilities"] = context.get("backend_capabilities")
        payload["unsupported_families_with_reasons"] = context.get("unsupported_families_with_reasons")
    if level == 0:
        payload["robot_profile"] = context.get("robot_profile")
        payload["simulator_capability_manifest"] = contracts.llm_simulator_capability_manifest(
            backend.generator_robot_name
        )
        payload["applicable_task_families"] = [
            item for item in context["applicable_task_families"] if item["backend_supported"]
        ]
    return payload


def history_for_prompt(history: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "iteration": row.get("iteration"),
            "description": row.get("description"),
            "module_name": row.get("module_name"),
            "task_id": row.get("task_id"),
            "primitive": row.get("primitive"),
            "primitive_hint": row.get("primitive_hint"),
            "family_hint": row.get("family_hint"),
            "status": row.get("status"),
            "normalized_reward": row.get("normalized_reward"),
            "success_rate": row.get("success_rate"),
            "success_rate_at_start": (row.get("success_curve") or [None])[0],
            "frontier_score": objectives.frontier_score(row.get("success_curve")),
            "error": llm.truncate_text(row.get("error")),
        }
        for row in sorted(history, key=lambda item: item["iteration"])[-limit:]
    ]


def propose_task(history: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """The LLM's next task. A proposal that fails validation is asked for again, with the reason, up to
    PROPOSAL_ATTEMPTS times; the last one is returned with its "validation_issue"."""
    backend = args.backend_spec
    used = {task_key(row["description"]) for row in history}
    adapters = registered_task_adapters(backend.generator_robot_name)
    runtime_tokens = capability_context.runtime_capability_tokens(args.capability_context["backend_capabilities"])
    issue: str | None = None

    def build_messages(level: int) -> list[dict[str, str]] | None:
        if level > 2:
            return None
        instructions = list(INSTRUCTIONS)
        if level == 0:
            instructions.insert(2, MANIFEST_INSTRUCTION)
        prompt = {
            "baseline": "VLM-only direct task proposal from empirical history",
            **prompt_payload(history, args, level),
            "instructions": instructions,
            "schema": PROPOSAL_SCHEMA,
            "fallback_first_task": backend.seed_task,
            "previous_validation_issue": issue,
        }
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": llm.compact_json(prompt)}]

    for _ in range(PROPOSAL_ATTEMPTS):
        proposed = llm.call_json_within_context(build_messages, args, max_tokens=768)
        description = str(proposed.get("description") or backend.seed_task).strip()
        proposed["description"] = ensure_robot_display(description, backend.robot_display_name)
        if len(adapters) == 1:
            proposed["task_id"], adapter = next(iter(adapters.items()))
            proposed["execution"] = adapter["execution"]
        issue = task_proposal_issue(proposed, used, set(adapters))
        if not issue:
            missing = capability_context.missing_runtime_capability(
                proposed["description"], runtime_tokens, category="task"
            )
            issue = f"backend lacks runtime capability {missing}" if missing else None
        if not issue:
            proposed["module_name"] = utils.safe_module_name(proposed.get("module_name") or proposed["description"])
            return proposed
    proposed["validation_issue"] = issue
    return proposed


def ensure_robot_display(description: str, robot_display_name: str) -> str:
    text, display = description.strip(), robot_display_name.strip()
    return text if display.lower() in text.lower() else f"{text} with the {display}"


def task_key(description: str) -> str:
    return re.sub(r"\s+", " ", description.strip().lower())


def task_proposal_issue(proposed: dict[str, Any], used: set[str], valid_task_ids: set[str]) -> str | None:
    """Why the proposal cannot be trained: an unknown adapter or execution, a description too short to define a task,
    or one that repeats an earlier task; None if it can."""
    task_id = str(proposed.get("task_id") or "").strip()
    if task_id not in valid_task_ids:
        return f"task_id must be one of {sorted(valid_task_ids)}"
    expected_execution = registry.TASKS[task_id].execution
    if str(proposed.get("execution") or "") != expected_execution:
        return f"execution must be {expected_execution!r} for task_id {task_id!r}"
    description = proposed["description"]
    if len(re.findall(r"\w+", description)) < 8:
        return "description is too short to define a concrete physical task and setting"
    key = task_key(description)
    if key in used:
        return "description duplicates a recent task"
    if any(SequenceMatcher(None, key, previous).ratio() >= 0.97 for previous in used):
        return "description is a near-duplicate of a recent task"
    return None


def compile_task(
    iteration: int, proposed: dict[str, Any], module_name: str, gpu_id: str, args: argparse.Namespace
) -> CurriculumRecord:
    """The record of the proposed task, compiled into an environment. Raises for a proposal that failed validation
    or a failed compile; the caller shows the error to the LLM."""
    if proposed.get("validation_issue"):
        raise RuntimeError(f"proposal rejected: {proposed['validation_issue']}")
    description = proposed["description"].strip()
    spec, module_path = generate_env.generate_environment(
        description,
        model=args.model,
        robot=args.backend_spec.generator_robot_name,
        task_id=proposed["task_id"],
        module_name=module_name,
        output_dir=common.generated_env_dir(args),
    )
    task_id = str(proposed.get("task_id"))
    family_hint = proposed.get("family_hint")
    return CurriculumRecord(
        iteration=iteration,
        timestamp=time.time(),
        backend=args.backend_spec.backend,
        robot=args.backend_spec.robot,
        description=description,
        module_name=spec.module_name,
        task_id=spec.task_id,
        primitive=spec.primitive,
        module_path=str(module_path),
        proposed=proposed,
        status="generated",
        gpu_id=gpu_id,
        primitive_hint=registry.TASKS[task_id].primitive if task_id in registry.TASKS else None,
        family_hint=None if family_hint is None else str(family_hint),
    )


def record_result(job: common.TrainingJob, history: list[dict[str, Any]], args: argparse.Namespace) -> None:
    record = job.record
    record.reward_budget = common.read_reward_budget(Path(record.module_path))
    common.finish_training(job, reward_budget=record.reward_budget, clip_normalized_reward=True)
    common.append_history(record, history, args.history)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.configure_gpus(args)
    common.setup_robot(args)
    common.set_run_paths(args, METHOD)
    history = common.load_history(args.history)
    gpu_ids = common.scheduler_gpu_ids(args.gpu_ids)
    iteration = common.next_iteration(history)
    rejected_rows: list[dict[str, Any]] = []
    trained = common.count_trained(history)
    while trained < args.iterations:
        batch_size = min(args.parallel_tasks, args.iterations - trained)
        records: list[CurriculumRecord] = []
        planning_history = [*history, *rejected_rows]
        rejected = 0
        while len(records) < batch_size:
            proposed = propose_task(planning_history, args)
            description = proposed["description"].strip()
            module_name = common.unique_module_name(
                proposed.get("module_name") or description, iteration, planning_history, prefix=MODULE_PREFIX
            )
            try:
                record = compile_task(
                    iteration, proposed, module_name, common.batch_gpu_id(gpu_ids, len(records)), args
                )
            except Exception as exc:  # an invalid proposal or a failed scene compile: show it to the LLM, don't die
                row = {
                    "iteration": iteration,
                    "description": description,
                    "module_name": module_name,
                    "status": "rejected",
                    "error": str(exc),
                }
                planning_history.append(row)
                rejected_rows = [*rejected_rows, row][-REJECTED_PROMPT_ROWS:]
                rejected += 1
                if rejected >= REJECTED_BATCH_LIMIT * batch_size:
                    break
                continue
            planning_history.append(asdict(record))
            records.append(record)
            iteration += 1

        jobs = [common.launch_training(record, args, METHOD) for record in records]
        common.wait_for_training(jobs, lambda job: record_result(job, history, args))
        trained += sum(record.status == "trained" and record.success_rate is not None for record in records)
    return 0


if __name__ == "__main__":
    main()
