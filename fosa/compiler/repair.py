"""Repair stage: deterministic fixes first, then up to two LLM repair rounds that each must not add validator
issues; what was tried is recorded in the spec's scene_repair field."""

from __future__ import annotations

import copy
import math
from typing import Any

from fosa.sim import reward_signals, scene_contract
from fosa.compiler import contracts, policy_specs, randomization, scenes

SCENE_REPAIR_ATTEMPTS = 2
REPAIR_MIN_TEMPERATURE = 0.1
REPAIR_HISTORY_LIMIT = 12  # history entries kept in the spec
# An LLM-repaired section whose own source is empty or one of these is marked as coming from the repair.
PLACEHOLDER_SOURCES = {"", "heuristic", "schema_only_fallback", "schema_template"}


def repair_generated_scene_document(
    scene_document: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    attempts: int = SCENE_REPAIR_ATTEMPTS,
) -> dict[str, Any]:
    """The spec with its validator issues repaired where possible; a repair that adds issues is rejected."""
    current = copy.deepcopy(scene_document)
    initial_issues = scenes.scene_spec_issues(current)
    current, deterministic_repairs = _apply_deterministic_scene_repairs(current)
    issues = scenes.scene_spec_issues(current)
    initial_issues = list(dict.fromkeys([*initial_issues, *issues]))
    if not issues and not deterministic_repairs:
        return current

    history: list[dict[str, Any]] = []
    if deterministic_repairs:
        history.append(
            {
                "attempt": 0,
                "source": "deterministic_scene_contract_repair",
                "status": "accepted" if not issues else "partial",
                "repairs": deterministic_repairs,
                "issues": issues,
            }
        )
    for attempt in range(1, attempts + 1):
        if not issues:
            break
        proposal = request_llm_scene_repair_spec(
            current,
            issues=issues,
            endpoint=endpoint,
            model=model,
            temperature=max(temperature, REPAIR_MIN_TEMPERATURE),
            max_tokens=max_tokens,
            api_key=api_key,
        )
        candidate = apply_llm_scene_repair(current, proposal)
        candidate, deterministic_repairs = _apply_deterministic_scene_repairs(candidate)
        candidate_issues = scenes.scene_spec_issues(candidate)
        if len(candidate_issues) > len(issues):
            history.append(
                {
                    "attempt": attempt,
                    "source": "llm_scene_repair",
                    "status": "rejected_regression",
                    "issues": candidate_issues,
                }
            )
            continue
        current, issues = candidate, candidate_issues
        history.append(
            {
                "attempt": attempt,
                "source": "llm_scene_repair",
                "status": "accepted" if not issues else "still_invalid",
                "deterministic_repairs": deterministic_repairs,
                "issues": issues,
            }
        )
    if issues:
        history.append(
            {
                "attempt": len(history) + 1,
                "source": "llm_scene_repair",
                "status": "unresolved",
                "issues": issues,
            }
        )
    current["scene_repair"] = {
        "initial_issues": initial_issues,
        "final_issues": issues,
        "history": history[-REPAIR_HISTORY_LIMIT:],
    }
    return current


def _apply_deterministic_scene_repairs(
    scene_document: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Re-normalize the contract and scene, free the manipulated object, and move static obstacles off the spawn."""
    current = copy.deepcopy(scene_document)
    robot = current["robot"]
    task_id = current["task_id"]
    current["simulation_contract"] = contracts.normalize_simulation_contract_spec(
        current["simulation_contract"],
        fallback=contracts.simulation_contract_schema_template(robot, task_id),
    )
    repairs: list[str] = []
    training_scene = scenes.normalize_training_scene_spec(
        current["training_scene"],
        simulation_contract=current["simulation_contract"],
    )
    current["training_scene"] = training_scene
    objects = training_scene.get("objects", [])
    if current["simulation_contract"]["execution"] == "manipulation":
        manipulated = next((obj for obj in objects if obj["role"] == "manipulated_object"), None)
        if manipulated is None:
            manipulated = next((obj for obj in objects if obj["role"] in scene_contract.MANIPULATED_ROLES), None)
        if manipulated is not None and manipulated["motion"] not in {"dynamic", "carried"}:
            manipulated["motion"] = "dynamic"
            manipulated.pop("attachment", None)
            repairs.append(f"manipulated_object_made_dynamic:{manipulated['name']}")
    repaired, overlap_repairs = scene_contract.repair_static_spawn_overlaps(
        training_scene,
        spawn_radius_m=_scene_spawn_radius(current),
    )
    current["training_scene"] = repaired
    return current, list(dict.fromkeys([*repairs, *overlap_repairs]))


def _scene_spawn_radius(scene_document: dict[str, Any]) -> float:
    """The spawn clearance widened by how far base_xy resets can move the robot."""
    base_xy = scene_document["domain_randomization"]["reset"].get("base_xy")
    values = base_xy if isinstance(base_xy, list) else []
    reset_extent = max((abs(item) for item in values if math.isfinite(item)), default=0.0)
    return contracts.SPAWN_RADIUS_M + reset_extent


def request_llm_scene_repair_spec(
    current: dict[str, Any],
    *,
    issues: list[str],
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
) -> dict[str, Any]:
    robot = current["robot"]
    task_id = current["task_id"]
    description = current["task_description"] or task_id
    contract = current["simulation_contract"]
    manifest = contracts.llm_simulator_capability_manifest(robot, task_id)
    randomization_interface = contracts.backend_randomization_interface()
    randomization_interface["reset"] = manifest["reset_randomization"]
    compact_current = {
        "robot": robot,
        "task_id": task_id,
        "description": description,
        "simulator_capability_manifest": manifest,
        "simulation_contract": contract,
        "render_scene": current.get("render_scene"),
        "training_scene": current.get("training_scene"),
        "domain_randomization": current.get("domain_randomization"),
        "task_reward": current.get("task_reward"),
        "task_termination": current.get("task_termination"),
        "backend_randomization_interface": randomization_interface,
        "simulator_observables": list(reward_signals.reward_signals_for_task(task_id, current.get("training_scene"))),
        "reward_objectives": list(contracts.REWARD_FUNCTION_OBJECTIVES),
        "termination_condition_ops": list(contracts.TASK_TERMINATION_CONDITIONS),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Repair a generated FOSA MuJoCo scene spec. Return JSON only. "
                "Return only the top-level sections named by validator_issues; omit unchanged sections. "
                "Do not change the actor observation, robot model, action space, or task_id. "
                "Do not change an existing LLM-authored reward function unless validator_issues identifies a reward problem. "
                "You own the render_scene and training_scene design: make task-specific scene assets and physics geometry "
                "from the language instead of choosing a fixed preset. Use supported primitive geometry and composite parts. "
                "You may add or repair task_termination using supported simulator observables; simulator safety termination remains fixed."
            ),
        },
        {
            "role": "user",
            "content": contracts.prompt_json(
                {
                    "schema": "fosa_scene_repair_v1",
                    "validator_issues": issues,
                    "current_spec": compact_current,
                    "repair_rules": [
                        "If render_scene was heuristic or schema-only, replace it with LLM-authored task-specific metadata.",
                        "Infer a distinct real setting from the exact task; do not default unrelated tasks to the same office/lab/warehouse or repeat one primitive at different positions. Use 4-10 recognizable multi-part context assemblies with varied silhouettes and materials.",
                        "Use current_spec.simulation_contract.scene_requirements as authoritative for task-required terrain, task objects, context geometry, and spatial constraints.",
                        "If validator_issues contains task_terrain, task_terrain_missing, task_terrain_semantics_mismatch, or task_terrain_height_missing, add terrain matching scene_requirements.terrain, set satisfies=['terrain'], and include a visible nonzero height range only when must_be_nonflat is true.",
                        "If validator_issues contains task_owned_object, task_owned_object_missing, or render_scene_task_object, design robot-scale task-object geometry matching scene_requirements.task_objects. Use composite primitive parts when a single primitive would erase important semantics.",
                        "If validator_issues contains manipulated_object_missing_or_static, add exactly one primary hand task object with role=manipulated_object and motion=dynamic.",
                        "If validator_issues contains task_object_variants_missing, give every non-static task object at least two entries in its variants list (each with name, geometry, rgba and optionally mass), varying shape, size or identity meaningfully; variants=null does not fix it.",
                        "If validator_issues contains task_object_attachment or task_object_attachment_missing, attach the task object to a real robot body from simulator_capability_manifest.mujoco_model.body_names.",
                        "If validator_issues contains static_object_overlaps_spawn:<name>, move or resize that object so the origin has robot clearance; dimensions_m are full metric dimensions.",
                        "If validator_issues contains initial_penetration:<geom1>:<geom2>:<depth>, move or resize the corresponding training-scene objects until they are separated or merely resting in contact; preserve the requested task geometry.",
                        "If validator_issues contains render_scene_context_geometry_missing, add render-only geometry matching scene_requirements.context_geometry and list context_geometry in each realizing asset's satisfies field.",
                        "If validator_issues contains training_scene_visual_only_context:<path>, remove that peripheral decoration from training_scene; recognizable setting decoration belongs in render_scene. If it truly affects task physics, instead add the exact causal satisfies key such as spatial_constraint or task_objects and state why in task_relevance.",
                        "If validator_issues contains training_scene_spatial_constraint_missing, add physical geometry matching scene_requirements.spatial_constraint and list spatial_constraint in its satisfies field.",
                        "If validator_issues contains task_object_semantics_mismatch, list task_objects in the matching task object's satisfies field.",
                        "If validator_issues contains task_object_physics_mismatch, copy every scene_requirements.task_objects.required_physics field exactly into the satisfying training object's physics; do not encode center of mass as world placement.",
                        "If validator_issues contains render_scene_context_assets_sparse, add distinct task-relevant multi-part asset families; do not clone one primitive or assembly.",
                        "If validator_issues contains render_scene_context_dominates_frame:<name>:<view>, move or shrink that assembly so it remains visible beside/behind the task but occupies far less of every camera view.",
                        "If validator_issues contains render_scene_robot_occluded:<view>, remove the camera-ray occluder or revise that camera variant until the complete robot and task object are visible.",
                        "If validator_issues contains render_scene_asset_relevance_missing:<name>, add an honest task_relevance justification or delete that asset when it is not needed by the task or simulation contract.",
                        "If validator_issues contains training_scene_asset_relevance_missing:<name>, add its exact physical task_relevance or delete it if no such reason exists.",
                        "If validator_issues contains render_scene_camera_not_subject_relative, replace distance_m with distance_scale relative to simulator_capability_manifest.robot_spec.render_camera, normally near 1.0.",
                        "If validator_issues contains domain_randomization_reset_missing:<key>, add a nondegenerate value for that exact key under domain_randomization.reset using backend_randomization_interface.reset; adding only reset_rationale does not fix it.",
                        "If validator_issues contains domain_randomization_model_missing:<field>, add a model operation for that exact supported field and select the relevant generated scene or robot entity; geom_size requires mode='scale'. The range must be nondegenerate (min < max): a fixed value such as [0.7, 0.7] is discarded, so use a narrow range around the task's nominal value.",
                        "If validator_issues contains domain_randomization_empty, add at least one nondegenerate reset range or model operation that the task description motivates; every range needs min < max.",
                        "If validator_issues contains domain_randomization_rationale_missing:<path>, add a short task-specific reset_rationale or model-operation task_relevance for that exact variation; remove it instead if it is irrelevant.",
                        "If validator_issues contains task_reward_function_missing_or_non_llm, add task_reward as fosa_task_reward_function_v1 with source='llm'; use only simulator_observables and reward_objectives.",
                        "Use only the current scene-grounded simulator_observables. Object signals have the form object/<scene-name>/<quantity>.",
                        "If validator_issues contains task_success_predicate, add exactly one executable task_reward.success_condition using a supported simulator observable and termination_condition_op. This is an evaluation metric, not an episode termination.",
                        "If validator_issues contains task_success_not_task_aligned:<name> and <name> measures the primary task achievement, add a reward component using that same observable and keep the success condition. Replace the success condition only when <name> measures mere survival, uprightness, or absence of failure.",
                        "If validator_issues contains task_success_target_mismatch:<name>, make success_condition use the same dynamic target observable as the aligned reward component, with a norm comparison and task-appropriate positive tolerance.",
                        "If validator_issues contains task_success_tracking_condition_invalid:<name>, use condition='norm_less_than' with a positive tracking-error threshold.",
                        "If validator_issues contains task_success_does_not_complete_spatial_constraint:<name>, add root_displacement for traversal or target_position_error for an explicit waypoint to the reward, and use the same observable with a task-distance threshold as sustained success.",
                        "If validator_issues contains task_success_does_not_complete_mobile_carry:<name> and the existing reward tracks a dynamic command, use that same tracking observable, target, and components for sustained success. Use traversal or arrival only when the task explicitly specifies a distance or destination, and include the exact success observable in the reward.",
                        "If validator_issues contains task_success_spatial_threshold_invalid, use greater_than or norm_greater_than with a positive traversal distance for root_displacement, or norm_less_than with a positive arrival tolerance for target_position_error.",
                        "For locomotion while carrying a payload, success must measure motion or command tracking; dropped-payload safety already enforces retention, so payload retention alone is not task completion.",
                        "If validator_issues contains task_success_not_persistent, set success_condition.hold_steps to a task-appropriate sustained interval, normally 25-100 control steps.",
                        "If validator_issues contains task_reward_observable_unsupported:<name>, replace it with an exact signal from current_spec.simulator_observables.",
                        "If validator_issues contains task_termination_function_missing_or_non_llm, add task_termination as fosa_task_termination_v1 with source='llm'; use only simulator_observables and termination_condition_ops.",
                        "If validator_issues contains task_termination_observable_unsupported:<name>, replace or remove that condition using only current_spec.simulator_observables.",
                        "Keep safety termination for falls, NaNs, invalid root pose, and dropped objects implicit in the simulator; task_termination may add only unrecoverable task-specific failures, never success or ordinary early-training errors.",
                        "If domain_randomization used abstract domains, replace it with direct reset ranges and model.operations.",
                        "Keep train-time geometry low-count and MJX/Warp-friendly.",
                        "Keep repairs anchored to the simulation_contract and generated task-owned scene elements.",
                    ],
                    "return_json": {
                        "render_scene": "omit unless a render_scene issue is present",
                        "training_scene": "omit unless a training_scene issue is present",
                        "domain_randomization": "omit unless a domain_randomization issue is present",
                        "task_reward": "omit unless a reward issue is present",
                        "task_termination": "omit unless a termination issue is present",
                        "notes": "one short summary of the repair",
                    },
                },
            ),
        },
    ]
    return contracts.request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max(max_tokens, 8192),
        api_key=api_key,
    )


def apply_llm_scene_repair(scene_document: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """Merge the sections the repair LLM returned into the spec and normalize each one again."""
    current = copy.deepcopy(scene_document)
    task_id = current["task_id"]
    contract = current["simulation_contract"]
    training_candidate = proposal.get("training_scene")
    if isinstance(training_candidate, dict):
        current["training_scene"] = scenes.normalize_training_scene_spec(
            _with_repair_source(training_candidate, "llm_training_scene_repair"),
            simulation_contract=contract,
        )

    domain_candidate = proposal.get("domain_randomization")
    if isinstance(domain_candidate, dict):
        domain_candidate = _merge_domain_randomization_repair(
            current["domain_randomization"],
            _with_repair_source(domain_candidate, "llm_domain_randomization_repair"),
        )
        current["domain_randomization"] = randomization.enforce_domain_randomization_contract(
            randomization.normalize_domain_randomization_spec(domain_candidate),
            contract=contract,
        )

    reward_candidate = proposal.get("task_reward")
    if isinstance(reward_candidate, dict):
        merged_reward = _merge_mappings(current["task_reward"], reward_candidate)
        # A repaired success condition replaces the old one; it may also arrive nested under reward_function.
        success = reward_candidate.get("success_condition")
        if not isinstance(success, dict) and isinstance(reward_candidate.get("reward_function"), dict):
            success = reward_candidate["reward_function"].get("success_condition")
        if isinstance(success, dict):
            merged_reward["success_condition"] = copy.deepcopy(success)
        current["task_reward"] = policy_specs.normalize_task_reward_spec(
            _with_repair_source(merged_reward, "llm_reward_repair"),
            task_id=task_id,
            training_scene=current["training_scene"],
        )

    termination_candidate = proposal.get("task_termination")
    if isinstance(termination_candidate, dict):
        merged_termination = _merge_mappings(current["task_termination"], termination_candidate)
        current["task_termination"] = policy_specs.normalize_task_termination_spec(
            _with_repair_source(merged_termination, "llm_termination_repair"),
            task_id=task_id,
            training_scene=current["training_scene"],
        )
    return current


def _with_repair_source(section: dict[str, Any], source: str) -> dict[str, Any]:
    if str(section.get("source") or "").lower() in PLACEHOLDER_SOURCES:
        return {**section, "source": source}
    return section


def _merge_domain_randomization_repair(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial repair without discarding already-valid randomization."""
    merged = _merge_mappings(current, patch)
    patch_model = patch.get("model")
    if isinstance(patch_model, dict):
        replacements = patch_model.get("operations")
        if isinstance(replacements, list):
            merged["model"]["operations"] = _merge_operations(current["model"]["operations"], replacements)
    return merged


def _merge_mappings(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Deep merge: nested dicts merge, every other patch value replaces the base value."""
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_operations(existing: list[Any], replacements: list[Any]) -> list[Any]:
    """An operation with a known id is merged into that operation; any other is appended."""
    merged = copy.deepcopy(existing)
    positions = {operation["id"]: index for index, operation in enumerate(merged)}
    for operation in replacements:
        if not isinstance(operation, dict):
            continue
        operation_id = str(operation.get("id") or "")
        if operation_id and operation_id in positions:
            position = positions[operation_id]
            merged[position] = _merge_mappings(merged[position], operation)
        else:
            if operation_id:
                positions[operation_id] = len(merged)
            merged.append(copy.deepcopy(operation))
    return merged
