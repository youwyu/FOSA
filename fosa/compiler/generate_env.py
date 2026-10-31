"""Compile a task description into the JSON environment spec that ``python -m fosa.learning.ppo --task`` trains.

The stages run in order, each an LLM call followed by normalization: simulation contract, critic observation, training
scene, domain randomization, reward, termination. A repair stage then fixes what the validator still reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fosa.sim import capabilities, registry
from fosa import llm, utils
from fosa.compiler import contracts, policy_specs, randomization, repair, scenes


@dataclass(frozen=True)
class GeneratedTaskSpec:
    """A compiled task; the written spec holds these fields plus reward_budget and the repair record."""

    task_description: str
    module_name: str
    task_id: str
    robot: str
    primitive: str
    execution: str
    simulation_contract: dict[str, Any]
    domain_randomization: dict[str, Any]
    critic_observation: dict[str, Any]
    task_reward: dict[str, Any]
    task_termination: dict[str, Any]
    render_scene: dict[str, Any]
    training_scene: dict[str, Any]
    backend: str = capabilities.BACKEND_ID


def _compiled_spec_cache_path(robot: str, atom_program: str) -> Path:
    """Where the environment compiled from this atom program is cached.

    LLM serving is not deterministic even at temperature 0, so the program, not the call, fixes the environment: every
    method compiles through here, and a program proposed again gets the same environment back.
    """
    digest = hashlib.sha256(atom_program.encode("utf-8")).hexdigest()[:32]
    return llm.model_log_root() / "compiled_specs" / robot / f"{digest}.json"


def set_goal_rotation_angle(
    scene_document: dict[str, Any], angle: float | None, axis: list[float] | None = None
) -> None:
    """The goal pose of an orientation task is the program's goal angle away from the reset pose (a compiler rule)."""
    domain = scene_document["domain_randomization"]
    domain.pop("goal_rotation_angle", None)
    domain.pop("goal_rotation_axis", None)
    if angle:
        domain["goal_rotation_angle"] = float(angle)
        if axis:
            domain["goal_rotation_axis"] = [float(x) for x in axis]


def generate_environment(
    description: str,
    *,
    success_predicate: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    robot: str | None = None,
    task_id: str | None = None,
    primitive: str | None = None,
    module_name: str | None = None,
    output_dir: Path | str | None = None,
    api_key: str | None = None,
    atom_program: str | None = None,
    goal_rotation_angle: float | None = None,
    goal_rotation_axis: list[float] | None = None,
) -> tuple[GeneratedTaskSpec, Path]:
    """Compile a task description into an environment spec; returns the spec and the path it was written to."""
    endpoint = endpoint or os.environ.get("FOSA_LLM_ENDPOINT", llm.DEFAULT_ENDPOINT)
    model = model or os.environ.get("FOSA_LLM_MODEL", llm.DEFAULT_MODEL)
    api_key = api_key or llm.default_api_key()
    robot_name = contracts.normalize_robot(robot)
    # Every stage that authors physics sees the program, so the environment is a function of the program.
    brief = (
        description
        if not atom_program
        else (
            f"{description}\n\nThis task was composed from the symbolic program below. The contract, "
            f"scene, randomization, reward and termination must reflect every part of it, and two "
            f"tasks with different programs must not compile to the same environment.\n{atom_program}"
        )
    )
    task_id = resolve_task_id(
        description,
        robot=robot_name,
        task_id=task_id,
        primitive=primitive,
        endpoint=endpoint,
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    clean_module_name = utils.safe_module_name(module_name or description)
    out_dir = Path(output_dir) if output_dir is not None else llm.model_log_root() / "generated_envs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clean_module_name}.json"
    cache_path = _compiled_spec_cache_path(robot_name, atom_program) if atom_program else None
    if cache_path is not None and cache_path.exists():
        scene_document = json.loads(cache_path.read_text(encoding="utf-8"))
        # The environment is the program's; the description and the file name are this task's.
        scene_document["task_description"] = description.strip()
        scene_document["module_name"] = clean_module_name
        set_goal_rotation_angle(scene_document, goal_rotation_angle, goal_rotation_axis)
        path.write_text(json.dumps(scene_document, indent=2), encoding="utf-8")
        return _spec_from_document(scene_document), path

    llm_args = {
        "endpoint": endpoint,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
    }
    scene_document = _compile_document(
        brief,
        description,
        robot=robot_name,
        task_id=task_id,
        module_name=clean_module_name,
        success_predicate=success_predicate,
        llm_args=llm_args,
    )
    set_goal_rotation_angle(scene_document, goal_rotation_angle, goal_rotation_axis)
    scene_document["reward_budget"] = task_reward_budget(
        scene_document["task_id"],
        scene_document["robot"],
        scene_document["task_reward"],
    )
    spec = _spec_from_document(scene_document)
    validate_executable_scene(scene_document)
    path.write_text(json.dumps(scene_document, indent=2), encoding="utf-8")
    issues = scenes.scene_spec_issues(scene_document)
    if issues:  # the same check PPO runs; rejecting here costs no GPU slot
        raise RuntimeError(f"generated scene spec is not trainable: {path} issues={issues}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(scene_document, indent=2), encoding="utf-8")
        os.replace(temporary, cache_path)
    return spec, path


def _compile_document(
    brief: str,
    description: str,
    *,
    robot: str,
    task_id: str,
    module_name: str,
    success_predicate: str | None,
    llm_args: dict[str, Any],
) -> dict[str, Any]:
    """Run the LLM stages in order, then the repair; the goal pose and the reward budget are set by the caller."""
    stage_args = {"robot": robot, "task_id": task_id, **llm_args}
    simulation_contract = contracts.generate_simulation_contract_spec(brief, **stage_args)
    # Videos are rendered outside compilation; every spec carries the same deferred render-scene placeholder.
    render_scene = scenes.deferred_render_scene(description, robot=robot, task_id=task_id)
    critic_observation = policy_specs.generate_critic_observation_spec(
        brief, **stage_args, simulation_contract=simulation_contract
    )
    training_scene = scenes.generate_training_scene_spec(
        brief, **stage_args, render_scene=render_scene, simulation_contract=simulation_contract
    )
    domain_randomization = randomization.generate_domain_randomization_spec(
        brief, **stage_args, simulation_contract=simulation_contract, training_scene=training_scene
    )
    policy_inputs = {
        "simulation_contract": simulation_contract,
        "domain_randomization": domain_randomization,
        "training_scene": training_scene,
    }
    task_reward = policy_specs.generate_task_reward_spec(
        brief, **stage_args, success_predicate=success_predicate, **policy_inputs
    )
    task_termination = policy_specs.generate_task_termination_spec(brief, **stage_args, **policy_inputs)
    spec = GeneratedTaskSpec(
        task_description=description.strip(),
        module_name=module_name,
        task_id=task_id,
        robot=robot,
        primitive=registry.TASKS[task_id].primitive,
        execution=registry.TASKS[task_id].execution,
        simulation_contract=simulation_contract,
        domain_randomization=domain_randomization,
        critic_observation=critic_observation,
        task_reward=task_reward,
        task_termination=task_termination,
        render_scene=render_scene,
        training_scene=training_scene,
    )
    # reward_budget is set once the repair has settled the reward; its key stays right after the spec fields.
    document = {**asdict(spec), "reward_budget": None}
    repair_compiled_model_randomization(document)
    document = repair.repair_generated_scene_document(document, **llm_args)
    repair_compiled_model_randomization(document)
    return document


def validate_executable_scene(scene_document: dict[str, Any]) -> None:
    """Build the training environment and its randomization function, so a spec that cannot compile fails here."""
    from fosa.sim.domain_randomization import make_domain_randomization_fn

    env = registry.make_env(scene_document["task_id"], impl="warp", scene_spec=scene_document)
    make_domain_randomization_fn(env)


def repair_compiled_model_randomization(scene_document: dict[str, Any]) -> None:
    """Remove model randomization operations that cannot compile against the scene."""
    from fosa.sim.domain_randomization import drop_uncompilable_model_operations

    env = registry.make_env(scene_document["task_id"], impl="warp", scene_spec=scene_document)
    drop_uncompilable_model_operations(env, scene_document["domain_randomization"])


def task_reward_budget(task_id: str, robot: str, task_reward: dict[str, Any]) -> float:
    """Normalizer for episode returns: episode length x summed positive weights, x ctrl_dt when rewards scale by dt."""
    components = task_reward["reward_function"]["components"]
    positive_weight = sum(max(0.0, item["weight"]) for item in components)
    step_scale = 1.0
    if task_reward["scale_by_dt"]:
        step_scale = float(contracts.simulator_capability_manifest(robot, task_id)["robot_spec"]["ctrl_dt"])
    return float(registry.TASKS[task_id].episode_length) * step_scale * (positive_weight or 1.0)


def select_task_id(
    description: str,
    *,
    robot: str,
    endpoint: str,
    model: str,
    temperature: float,
    api_key: str | None,
) -> str | None:
    """The registered task the LLM picks for the description, or None when its answer is not one of them."""
    candidates = {
        task: registry.TASKS[task].description for task in sorted(registry.TASKS) if registry.TASKS[task].robot == robot
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    messages = [
        {
            "role": "system",
            "content": (
                "Select the single registered simulator task that best backs the requested robot task. "
                "Do not restrict the robot to embodiment-typical tasks; choose purely by task semantics. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": contracts.prompt_json(
                {
                    "robot": robot,
                    "task_description": description,
                    "registered_tasks": candidates,
                    "return_json": {
                        "task_id": "one key from registered_tasks",
                        "reason": "one short sentence",
                    },
                }
            ),
        },
    ]
    response = contracts.request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=256,
        api_key=api_key,
    )
    task_id = str(response.get("task_id") or "")
    return task_id if task_id in candidates else None


def resolve_task_id(
    description: str,
    *,
    robot: str,
    task_id: str | None,
    primitive: str | None,
    endpoint: str,
    model: str,
    temperature: float,
    api_key: str | None,
) -> str:
    """The given task, else the robot's one task with the given primitive, else the LLM's pick, else word overlap."""
    if task_id:
        return task_id
    candidates = {name: spec for name, spec in registry.TASKS.items() if spec.robot == robot}
    if primitive:
        matches = [name for name, spec in candidates.items() if spec.primitive == primitive]
        if len(matches) != 1:
            raise ValueError(f"Primitive {primitive!r} does not identify one registered task for {robot!r}")
        return matches[0]
    return select_task_id(
        description,
        robot=robot,
        endpoint=endpoint,
        model=model,
        temperature=temperature,
        api_key=api_key,
    ) or infer_task_id(robot, description)


def infer_task_id(robot: str, description: str) -> str:
    """The robot's task that shares the most words with the description, else its default task."""
    text = utils.safe_module_name(description)
    candidates = [task for task, spec in registry.TASKS.items() if spec.robot == robot]
    text_tokens = {token for token in text.split("_") if len(token) > 2}
    scored: list[tuple[int, str]] = []
    for task in candidates:
        spec = registry.TASKS[task]
        task_text = utils.safe_module_name(" ".join([task, spec.name, spec.description]))
        task_tokens = {token for token in task_text.split("_") if len(token) > 2}
        scored.append((len(text_tokens & task_tokens), task))
    best_score, best_task = max(scored, key=lambda item: (item[0], item[1]))
    return best_task if best_score > 0 else registry.default_task_for_robot(robot)


def _spec_from_document(document: dict[str, Any]) -> GeneratedTaskSpec:
    return GeneratedTaskSpec(
        task_description=document["task_description"],
        module_name=document["module_name"],
        task_id=document["task_id"],
        robot=document["robot"],
        primitive=document["primitive"],
        execution=document["execution"],
        simulation_contract=dict(document["simulation_contract"]),
        domain_randomization=dict(document["domain_randomization"]),
        critic_observation=dict(document["critic_observation"]),
        task_reward=dict(document["task_reward"]),
        task_termination=dict(document["task_termination"]),
        render_scene=dict(document["render_scene"]),
        training_scene=dict(document["training_scene"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec, path = generate_environment(
        args.description,
        endpoint=args.endpoint,
        model=args.model,
        robot=args.robot,
        task_id=args.task_id,
        primitive=args.primitive,
        module_name=args.module_name,
        output_dir=args.output_dir,
        api_key=args.api_key,
    )
    print(f"Generated MJP task spec: {spec.task_id}")
    print(path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description")
    parser.add_argument("--robot", default=registry.default_robot())
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--primitive", default=None)
    parser.add_argument("--module-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=llm.model_log_root() / "generated_envs")
    parser.add_argument("--endpoint", default=os.environ.get("FOSA_LLM_ENDPOINT", llm.DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.environ.get("FOSA_LLM_MODEL", llm.DEFAULT_MODEL))
    parser.add_argument("--api-key", default=llm.default_api_key())
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
