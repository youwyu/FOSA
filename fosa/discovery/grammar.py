"""The typed atom grammar searched by every grammar-based method (FOSA and its ablations, MCTS, LLM-MCTS, MCTS-Learn):
atoms and programs (task skeletons), the grammar built from the robot's task templates, validity and completeness,
duplicate and novelty measures, and the LLM verbalizer that renders a program as a task (paper appendix, Symbolic
Task Skeleton Language)."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from fosa import llm, utils
from fosa.discovery import capability_context, common, task_templates

ALLOWED_ATOM_CATEGORIES = {
    "capability",
    "embodiment",
    "entity",
    "environment",
    "action",
    "goal",
    "constraint",
    "condition",
    "difficulty",
    "composition",
    "observation",
}
SLOT_RULES = {  # the atom categories that can fill each slot of a program
    "capability": ["capability"],
    "goal": ["goal", "action"],
    "entity_or_environment": ["entity", "environment"],
    "constraint": ["constraint"],
    "condition": ["condition", "difficulty"],
    "observation": ["observation"],
    "composition": ["composition"],
}
CARRY_PATTERN = r"\b(?:carry|carried|carrying|payload|cargo)\b"
NOVELTY_WINDOW = 200  # recent history rows that novelty is measured against
# fmt: off
TEXT_KEY_STOPWORDS = {
    "the", "a", "an", "robot", "task", "train", "perform", "using", "with", "must",
    "should", "go2", "jackal", "booster", "t1", "leap", "hand", "inspire",
    "its", "and", "to", "of", "for",
    "while", "on", "by", "this", "capability", "control", "object", "state",
}
# fmt: on
VAGUE_PHRASES = (
    "perform contact control",
    "perform object state control",
    "perform an in-hand task",
    "perform in-hand",
    "perform a grasp",
    "demonstrate contact control",
    "demonstrate object_state_control",
)


@dataclass
class AtomSpec:
    name: str
    category: str
    description: str
    arguments: dict[str, Any] = field(default_factory=dict)
    incompatible_with: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)  # atom names, categories or slots
    difficulty_axes: list[str] = field(default_factory=list)
    backend_supported: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomGrammar:
    atoms: dict[str, AtomSpec]
    start_symbols: list[str]  # the capability atoms a program starts from
    slot_rules: dict[str, list[str]]
    global_constraints: list[str]


@dataclass
class AtomBinding:
    atom: str
    role: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomProgram:
    bindings: list[AtomBinding]
    open_slots: list[str]
    constraints: list[str]
    depth: int


@dataclass
class AtomStats:
    """Visits and value of one atom over the trained tasks that use it."""

    visits: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    success_sum: float = 0.0
    success_mean: float | None = None


def request_json(
    messages: list[dict[str, str]], args: argparse.Namespace, *, max_tokens: int, timeout_s: float | None = None
) -> dict[str, Any]:
    """The LLM's JSON reply to a fixed prompt."""
    return llm.call_json_within_context(
        lambda level: messages if level == 0 else None, args, max_tokens=max_tokens, timeout_s=timeout_s
    )


# ----------------------------------------------------------------------------------------------------------------------
# The grammar


def build_atom_grammar(args: argparse.Namespace) -> AtomGrammar:
    """The robot's grammar: a capability atom per execution primitive, and a goal, an environment and a constraint
    atom per task family, with the atoms the simulator cannot run marked unsupported."""
    atom_grammar = task_template_atom_grammar(args)
    disable_unsupported_atoms(atom_grammar, args)
    return atom_grammar


def backend_capability_tokens(args: argparse.Namespace) -> list[str]:
    return sorted(capability_context.runtime_capability_tokens(args.capability_context["backend_capabilities"]))


def task_template_atom_grammar(args: argparse.Namespace) -> AtomGrammar:
    tokens = backend_capability_tokens(args)
    atoms: dict[str, AtomSpec] = {}

    def add(name: str, category: str, description: str, primitive: str, metadata: dict[str, Any], **fields: Any) -> str:
        name = utils.safe_module_name(name)
        metadata = {"required_backend_capabilities": [f"primitive:{primitive}"], **metadata}
        atoms[name] = AtomSpec(name=name, category=category, description=description, metadata=metadata, **fields)
        return name

    capability_atoms: dict[str, str] = {}  # primitive -> its capability atom
    for token in tokens:
        if token.startswith("primitive:"):
            primitive = token.split(":", 1)[1]
            description = f"Execute the registered {primitive.replace('_', ' ')} control primitive."
            capability_atoms[primitive] = add(
                f"{primitive}_capability", "capability", description, primitive, {"primitive": primitive}
            )

    family_atoms: dict[str, list[str]] = {}  # family -> its goal, environment and constraint atoms
    templates: dict[str, task_templates.TaskFamilyTemplate] = args.task_templates
    for family, template in sorted(templates.items()):
        candidates = [template.default_primitive, *template.primitives]
        primitive = next((item for item in candidates if item in capability_atoms), None)
        if primitive is None:
            continue
        prefix, label = utils.safe_module_name(family), family.replace("_", " ")
        goal_arguments = dimension_arguments({**template.entity_dimensions, **template.goal_dimensions})
        environment_arguments = dimension_arguments(template.environment_dimensions)
        difficulty_axes = [
            name
            for name, spec in {**goal_arguments, **environment_arguments}.items()
            if spec.get("kind") in {"continuous", "integer", "ordinal"}
        ]
        metadata = {
            "task_family": family,
            "primitive": primitive,
            "constraints": list(template.constraints),
            "success_predicates": list(template.success_predicates),
        }
        constraint_text = "; ".join([*template.constraints, *template.success_predicates])
        family_atoms[family] = [
            add(
                f"{prefix}_goal",
                "goal",
                template.description,
                primitive,
                metadata,
                arguments=goal_arguments,
                difficulty_axes=[name for name in difficulty_axes if name in goal_arguments],
            ),
            add(
                f"{prefix}_environment",
                "environment",
                f"Use the environment specified by the {label} task family.",
                primitive,
                metadata,
                arguments=environment_arguments,
                required_context=["goal"],
                difficulty_axes=[name for name in difficulty_axes if name in environment_arguments],
            ),
            add(
                f"{prefix}_constraint",
                "constraint",
                constraint_text or f"Satisfy the measurable {label} success condition.",
                primitive,
                metadata,
                required_context=["goal"],
            ),
        ]

    # families compose freely; only atoms of different execution primitives exclude each other
    primitive_of = {name: atoms[names[0]].metadata["primitive"] for names in family_atoms.values() for name in names}
    for names in family_atoms.values():
        primitive = atoms[names[0]].metadata["primitive"]
        incompatible = sorted(name for name, item in primitive_of.items() if item != primitive)
        incompatible += [name for item, name in capability_atoms.items() if item != primitive]
        for name in names:
            atoms[name].incompatible_with = incompatible

    global_constraints = [
        "Combine task families freely, but use a single execution primitive and only "
        "capabilities proven by the active backend."
    ]
    if "asset:supports_carried_payload" not in tokens and "primitive:manipulation" not in tokens:
        global_constraints.append("Do not propose grasping, lifting, holding, or carrying objects or payloads.")
    return AtomGrammar(atoms, list(capability_atoms.values()), SLOT_RULES, global_constraints)


def dimension_arguments(dimensions: dict[str, task_templates.DimensionSpec]) -> dict[str, Any]:
    """The template dimensions as atom arguments: each dimension's set fields, by dimension name."""
    return {
        name: {key: value for key, value in asdict(spec).items() if key != "name" and value is not None}
        for name, spec in sorted(dimensions.items())
    }


def disable_unsupported_atoms(atom_grammar: AtomGrammar, args: argparse.Namespace) -> None:
    """Mark unsupported the atoms that need an observation or runtime effect the simulator lacks, carry atoms without
    a payload support frame, and every atom of a family with an unsupported atom."""
    available = set(backend_capability_tokens(args))
    observations = args.capability_context["backend_capabilities"]["supported_observation_terms"]
    atoms = atom_grammar.atoms.values()
    for atom in atoms:
        text = f"{atom.name} {atom.description}"
        missing_modality = capability_context.missing_observation_modality(text, observations)
        missing_runtime = capability_context.missing_runtime_capability(text, available, category=atom.category)
        if missing_modality or missing_runtime:
            atom.backend_supported = False
            atom.metadata["missing_backend_capabilities"] = (
                [f"observation:{missing_modality}"] if missing_modality else [f"runtime:{missing_runtime}"]
            )
    for atom in atoms:
        is_carry = re.search(CARRY_PATTERN, atom.description.lower())
        if atom.backend_supported and atom.metadata.get("primitive") != "manipulation" and is_carry:
            atom.backend_supported = False
            atom.metadata["missing_backend_capabilities"] = ["payload_support_position_m"]
    disabled_families = {
        str(atom.metadata["task_family"])
        for atom in atoms
        if not atom.backend_supported and atom.metadata.get("task_family")
    }
    for atom in atoms:
        if atom.backend_supported and atom.metadata.get("task_family") in disabled_families:
            atom.backend_supported = False
            atom.metadata["missing_backend_context"] = ["task_family"]
    atom_grammar.start_symbols = [
        name for name in atom_grammar.start_symbols if atom_grammar.atoms[name].backend_supported
    ]


# ----------------------------------------------------------------------------------------------------------------------
# Programs


def serialize_atom_program(program: AtomProgram, grammar: AtomGrammar) -> str:
    """The program as text, every binding in order: the scene compiler's and the state encoder's input."""
    bindings = []
    for index, binding in enumerate(program.bindings, 1):
        description = " ".join(grammar.atoms[binding.atom].description.split())[:120]
        arguments = json.dumps(binding.arguments, sort_keys=True, separators=(",", ":"))
        bindings.append(
            f"{index}: role {binding.role}; atom {binding.atom.replace('_', ' ')}; "
            f"meaning {description}; arguments {arguments}"
        )
    constraints = json.dumps(program.constraints, sort_keys=True, separators=(",", ":"))
    return (
        f"Ordered AtomProgram; depth {program.depth}; "
        f"bindings [{' | '.join(bindings)}]; "
        f"open slots {', '.join(program.open_slots) or 'none'}; "
        f"constraints {constraints}"
    )


def atom_program_to_dict(program: AtomProgram) -> dict[str, Any]:
    return utils.json_safe(asdict(program))


def atom_program_from_dict(raw: dict[str, Any]) -> AtomProgram:
    return AtomProgram(
        bindings=[AtomBinding(item["atom"], item["role"], dict(item["arguments"])) for item in raw["bindings"]],
        open_slots=list(raw["open_slots"]),
        constraints=list(raw["constraints"]),
        depth=raw["depth"],
    )


def goal_rotation_angle(program: AtomProgram, grammar: AtomGrammar) -> float | None:
    """The program's goal rotation angle: a goal atom's argument measured in radians whose name contains "angle"."""
    for binding in program.bindings:
        atom = grammar.atoms[binding.atom]
        if atom.category != "goal":
            continue
        for name, spec in atom.arguments.items():
            if "angle" in name and spec.get("unit") == "rad" and name in binding.arguments:
                return float(binding.arguments[name])
    return None


def goal_rotation_axis(program: AtomProgram, grammar: AtomGrammar) -> list[float] | None:
    """The vertical axis when the program's goal angle is a yaw (argument named "yaw"), else None (a random axis)."""
    for binding in program.bindings:
        atom = grammar.atoms[binding.atom]
        if atom.category == "goal" and any("yaw" in name and name in binding.arguments for name in atom.arguments):
            return [0.0, 0.0, 1.0]
    return None


def atom_program_key(program: AtomProgram) -> str:
    # what the program is, not how it was built: depth, open slots and binding order differ between edit paths
    payload = atom_program_to_dict(program)
    payload = {
        "bindings": sorted(json.dumps(binding, sort_keys=True) for binding in payload["bindings"]),
        "constraints": payload["constraints"],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def atom_names(program: AtomProgram) -> list[str]:
    return [binding.atom for binding in program.bindings]


def program_summary(program: AtomProgram) -> str:
    return " -> ".join(f"{binding.role}:{binding.atom}" for binding in program.bindings) or "root"


def is_program_valid(program: AtomProgram, grammar: AtomGrammar) -> tuple[bool, str | None]:
    if not program.bindings:
        return True, None
    if len(program.bindings) > common.MAX_ATOMS_PER_PROGRAM:
        return False, "program exceeds max atoms"
    names = atom_names(program)
    if any(name not in grammar.atoms for name in names):
        return False, "program references atom outside grammar"
    if any(not grammar.atoms[name].backend_supported for name in names):
        return False, "program uses backend-unsupported atom"
    name_set = set(names)
    categories = {grammar.atoms[name].category for name in names}
    roles = {binding.role for binding in program.bindings}
    if "capability" not in roles and not roles & {"goal", "action"}:
        return False, "program lacks capability or goal atom"
    for name in names:
        atom = grammar.atoms[name]
        if set(atom.incompatible_with) & name_set:
            return False, f"atom {name} is incompatible with another selected atom"
        for requirement in atom.required_context:
            if not required_context_satisfied(requirement, name_set, categories, program.open_slots, grammar):
                return False, f"atom {name} requires missing context {requirement}"
    return True, None


def program_completeness_score(program: AtomProgram) -> float:
    """How much of a task the program specifies, in [0, 1]: weighted filled roles and bound arguments."""
    if not program.bindings:
        return 0.0
    roles = {binding.role for binding in program.bindings}
    score = 0.0
    if "capability" in roles:
        score += 0.18
    if roles & {"goal", "action"}:
        score += 0.22
    if roles & {"entity", "environment"}:
        score += 0.18
    if "constraint" in roles:
        score += 0.14
    bound_args = sum(
        1 for binding in program.bindings for value in binding.arguments.values() if is_bound_argument(value)
    )
    if bound_args:
        score += min(0.12, 0.04 * bound_args)
    if "observation" in roles:
        score += 0.04
    essential_open = set(program.open_slots) & {"capability", "goal", "entity_or_environment", "constraint"}
    score += max(0.0, 0.08 - 0.02 * len(essential_open))
    if program.constraints or "constraint" in roles:
        score += 0.04
    if roles & {"goal", "action"} and ("constraint" in roles or program.constraints):
        score += 0.08
    return min(max(score, 0.0), 1.0)


def is_bound_argument(value: Any) -> bool:
    return value is not None and value != "" and not (isinstance(value, list) and not value)


def is_program_complete_for_evaluation(program: AtomProgram, grammar: AtomGrammar) -> tuple[bool, str | None]:
    """Whether the program specifies a whole task, so it can be verbalized, compiled and trained; else why not."""
    valid, reason = is_program_valid(program, grammar)
    if not valid:
        return False, reason
    missing_context = missing_required_contexts(program, grammar)
    if missing_context:
        return False, f"missing required context: {', '.join(missing_context)}"
    score = program_completeness_score(program)
    roles = {binding.role for binding in program.bindings}
    if score < 0.70:
        return False, f"completeness score {score:.2f} < 0.70"
    if len(program.bindings) < common.MIN_COMPLETE_BINDINGS:
        return False, f"bindings {len(program.bindings)} < {common.MIN_COMPLETE_BINDINGS}"
    if not roles & {"goal", "action"}:
        return False, "missing goal/action atom"
    if not roles & {"entity", "environment"}:
        return False, "missing entity/environment atom"
    if "constraint" not in roles and not program.constraints:
        return False, "missing constraint or success-related atom"
    return True, None


def missing_required_contexts(program: AtomProgram, grammar: AtomGrammar) -> list[str]:
    names = set(atom_names(program))
    roles = {binding.role for binding in program.bindings}
    missing: list[str] = []
    for binding in program.bindings:
        for requirement in grammar.atoms[binding.atom].required_context:
            if requirement not in missing and not required_context_satisfied(requirement, names, roles, [], grammar):
                missing.append(requirement)
    return missing


def required_context_slot(requirement: str, grammar: AtomGrammar) -> str | None:
    atom = grammar.atoms.get(requirement)
    category = atom.category if atom is not None else requirement
    return next((slot for slot, categories in grammar.slot_rules.items() if category in categories), None)


def required_context_satisfied(
    requirement: str, names: set[str], roles: set[str], open_slots: list[str], grammar: AtomGrammar
) -> bool:
    """Whether an atom, role or category the program has meets the requirement, or an open slot can still meet it.
    A requirement the grammar does not know is met."""
    known = requirement in grammar.atoms or requirement in ALLOWED_ATOM_CATEGORIES or requirement in grammar.slot_rules
    if not known:
        return True
    if (
        requirement in names
        or requirement in roles
        or any(grammar.atoms[name].category == requirement for name in names)
    ):
        return True
    slot = required_context_slot(requirement, grammar)
    return slot in open_slots if slot is not None else False


def bind_default_arguments(atom: AtomSpec) -> dict[str, Any]:
    """Each argument's default, else its first listed value (True for a boolean)."""
    bound: dict[str, Any] = {}
    for name, spec in atom.arguments.items():
        if "default" in spec:
            bound[name] = spec["default"]
        elif spec.get("values"):
            bound[name] = spec["values"][0]
        elif spec.get("kind") == "boolean":
            bound[name] = True
    return bound


def role_for_slot(slot: str, atom: AtomSpec) -> str:
    if slot == "entity_or_environment":
        return atom.category
    if slot == "goal" and atom.category == "action":
        return "goal"
    if slot == "condition" and atom.category == "difficulty":
        return "condition"
    return slot


# ----------------------------------------------------------------------------------------------------------------------
# Duplicates and novelty


def task_text_key(text: str) -> str:
    text = text.lower().translate(str.maketrans(dict.fromkeys(string.punctuation, " ")))
    return " ".join(token for token in text.split() if token not in TEXT_KEY_STOPWORDS)


def atom_jaccard(program_a: AtomProgram, program_b: AtomProgram) -> float:
    a = {f"{binding.role}:{binding.atom}" for binding in program_a.bindings}
    b = {f"{binding.role}:{binding.atom}" for binding in program_b.bindings}
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def is_duplicate_program_or_task(
    program: AtomProgram,
    description: str | None,
    programs: list[AtomProgram],
    history: list[dict[str, Any]],
    grammar: AtomGrammar,
) -> tuple[bool, float, str | None]:
    """Whether the program, or its description, repeats one of `programs` or a recent history row; with a duplicate
    score in [0, 1] and the reason."""
    program_key = atom_program_key(program)
    recent = history[-common.RECENT_DUPLICATE_WINDOW :]
    comparisons = programs + [atom_program_from_dict(row["atom_program"]) for row in recent]

    for other in comparisons:
        if atom_program_key(other) == program_key:
            return True, 1.0, "exact atom program already exists"
    # near-duplicate rules apply to complete tasks, not to search prefixes
    if not is_program_complete_for_evaluation(program, grammar)[0]:
        return False, 0.0, None

    max_atom_score = 0.0
    meaningful_variant = False
    for other in comparisons:
        score = atom_jaccard(program, other)
        max_atom_score = max(max_atom_score, score)
        if score < common.DUPLICATE_ATOM_JACCARD_THRESHOLD:
            continue
        if not has_meaningful_new_difficulty(program, other):
            return True, score, "atom-program Jaccard duplicate"
        meaningful_variant = True
    if meaningful_variant:
        return False, max_atom_score, None

    text_score = 0.0
    if description:
        key = task_text_key(description)
        for row in recent:
            other = task_text_key(row["description"])
            if not other:
                continue
            if key == other:
                return True, 1.0, "recent description duplicate"
            ratio = difflib.SequenceMatcher(None, key, other).ratio()
            text_score = max(text_score, ratio)
            if ratio >= common.DUPLICATE_TEXT_THRESHOLD:
                return True, ratio, "recent description too similar"

    signature = main_action_object_signature(program)
    if signature[0] and signature[1]:
        repeats = sum(1 for row in recent if row_signature(row) == signature)
        if repeats >= 3 and not program_has_condition_or_difficulty(program):
            return True, min(1.0, 0.75 + 0.05 * repeats), "action-object-condition signature repeated"
    return False, max(max_atom_score, text_score), None


def has_meaningful_new_difficulty(program: AtomProgram, other: AtomProgram) -> bool:
    """Whether a binding the programs share differs in a condition or in a numeric argument."""
    other_bindings = {(binding.role, binding.atom): binding.arguments for binding in other.bindings}
    for binding in program.bindings:
        previous = other_bindings.get((binding.role, binding.atom))
        if previous is None or binding.arguments == previous:
            continue
        if binding.role in {"condition", "difficulty"}:
            return True
        changed = {
            name
            for name in binding.arguments.keys() | previous.keys()
            if binding.arguments.get(name) != previous.get(name)
        }
        if any(isinstance(binding.arguments.get(name, previous.get(name)), (int, float)) for name in changed):
            return True
    return False


def program_has_condition_or_difficulty(program: AtomProgram) -> bool:
    return any(binding.role in {"condition", "difficulty"} for binding in program.bindings)


def main_action_object_signature(program: AtomProgram) -> tuple[str | None, str | None, str | None]:
    """The program's first goal, entity and condition-like atoms."""
    action = entity = condition = None
    for binding in program.bindings:
        if binding.role in {"goal", "action"} and action is None:
            action = binding.atom
        elif binding.role == "entity" and entity is None:
            entity = binding.atom
        elif binding.role in {"condition", "constraint", "environment"} and condition is None:
            condition = binding.atom
    return action, entity, condition


def row_signature(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("main_action"), row.get("main_entity"), row.get("main_condition")


def atom_visit_novelty(names: list[str], atom_stats: dict[str, AtomStats]) -> float:
    return sum(1.0 / math.sqrt(1.0 + atom_stats.get(name, AtomStats()).visits) for name in names) / len(names)


def atom_pair_novelty(program: AtomProgram, history: list[dict[str, Any]]) -> float:
    pairs = atom_pair_keys(program)
    if not pairs:
        return 0.5
    counts = Counter(key for row in history[-NOVELTY_WINDOW:] for key in row["atom_pair_keys"])
    return sum(1.0 / math.sqrt(1.0 + counts[key]) for key in pairs) / len(pairs)


def atom_pair_keys(program: AtomProgram) -> list[str]:
    names = sorted(set(atom_names(program)))
    return [f"{a}|{b}" for idx, a in enumerate(names) for b in names[idx + 1 :]]


def novelty_bonus_from_program(
    program: AtomProgram,
    atom_stats: dict[str, AtomStats],
    history: list[dict[str, Any]],
    programs: list[AtomProgram],
    grammar: AtomGrammar,
) -> float:
    """N(x): how rarely the program's atoms, atom pairs and main signature occur in the recent history, less its
    duplicate score if it duplicates one of `programs` or the history."""
    single = atom_visit_novelty(atom_names(program), atom_stats)
    pair = atom_pair_novelty(program, history)
    signature = main_action_object_signature(program)
    signature_count = sum(1 for row in history[-NOVELTY_WINDOW:] if row_signature(row) == signature)
    signature_novelty = 1.0 / math.sqrt(1.0 + signature_count)
    duplicate, duplicate_score, _reason = is_duplicate_program_or_task(program, None, programs, history, grammar)
    duplicate_penalty = duplicate_score if duplicate else 0.0
    novelty = 0.4 * single + 0.4 * pair + 0.2 * signature_novelty
    return max(0.0, novelty - duplicate_penalty)


def is_vague_abstract_description(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in VAGUE_PHRASES)


# ----------------------------------------------------------------------------------------------------------------------
# Verbalization


def verbalize_atom_program(
    program: AtomProgram, grammar: AtomGrammar, args: argparse.Namespace, proposal: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The program as a task: description, primitive and success predicate. An LLM-MCTS proposal (the node's
    created_from) is used as it is unless its description is vague; otherwise the LLM renders the program."""
    created = proposal or {}
    proposal_description = created.get("proposal_description") or ""
    if proposal_description and not is_vague_abstract_description(proposal_description):
        task = {
            "description": proposal_description,
            "primitive": created.get("proposal_primitive"),
            "success_predicate": created.get("proposal_success_predicate"),
            "notes": "Used canonicalized VLM expansion proposal.",
        }
        return normalize_verbalized_task(task, program, grammar, args)

    prompt = {
        "request": "Render one typed atom program into one concrete robot RL task.",
        "robot": args.robot,
        "robot_display_name": args.robot_display_name,
        "backend": args.backend,
        "backend_capabilities": args.capability_context["backend_capabilities"],
        "atom_program": atom_program_to_dict(program),
        "atom_definitions": {name: asdict(grammar.atoms[name]) for name in atom_names(program)},
        "proposal_description": proposal_description or None,
        "proposal_success_predicate": created.get("proposal_success_predicate"),
        "instructions": [
            "Render the specific concrete task represented by the atom program.",
            "Preserve all atoms and arguments from the atom program.",
            "Do not silently add unsupported assets.",
            "Output one concrete robot task, not a list.",
            "Mention the robot display name.",
            "Mention the concrete object or environment, concrete action, constraint, and success predicate.",
            "Do not say 'perform contact control', 'perform object state control', 'perform an in-hand task', or "
            "similarly vague wording.",
            "Mention a measurable goal and success condition.",
            "Use only backend-supported primitives/assets.",
            "Natural language is only a rendering; the atom program remains canonical.",
            "Return JSON only.",
        ],
        "schema": {
            "description": "one natural-language task",
            "primitive": "one supported primitive",
            "success_predicate": "measurable success predicate",
            "notes": "optional",
        },
    }
    messages = [
        {
            "role": "system",
            "content": "You render typed robot atom programs into concrete RL task descriptions. Output JSON only.",
        },
        {"role": "user", "content": llm.compact_json(prompt)},
    ]
    raw = request_json(messages, args, max_tokens=1024)
    if not raw.get("description"):
        raw = template_task(program, grammar, args)
    return normalize_verbalized_task(raw, program, grammar, args)


def normalize_verbalized_task(
    raw: dict[str, Any], program: AtomProgram, grammar: AtomGrammar, args: argparse.Namespace
) -> dict[str, Any]:
    """The rendered task with a vague description, an unsupported primitive or a missing success predicate replaced
    from the template task."""
    task = dict(raw)
    supported = set(args.backend_spec.supported_primitives)
    primitive = str(task.get("primitive") or "").strip()
    template = template_task(program, grammar, args)
    if is_vague_abstract_description(str(task.get("description") or "")):
        task.update(template)
        task.setdefault("notes", "")
        task["notes"] = f"{task['notes']} vague VLM rendering replaced by concrete fallback.".strip()
        primitive = str(task.get("primitive") or "").strip()
    if primitive not in supported:
        task["primitive"] = template["primitive"]
        task.setdefault("notes", "")
        note = f" primitive coerced from {primitive!r} to backend-supported {task['primitive']!r}."
        task["notes"] = f"{task['notes']}{note}".strip()
    if not str(task.get("success_predicate") or "").strip():
        task["success_predicate"] = template["success_predicate"]
    return task


def template_task(program: AtomProgram, grammar: AtomGrammar, args: argparse.Namespace) -> dict[str, Any]:
    """The program as a task without the LLM: its goal atom's text and bound arguments, its family's primitive and
    success predicates."""
    supported = set(args.backend_spec.supported_primitives)
    family_atom = next(
        (
            grammar.atoms[binding.atom]
            for binding in program.bindings
            if grammar.atoms[binding.atom].metadata.get("task_family")
        ),
        None,
    )
    metadata = family_atom.metadata if family_atom is not None else {}
    primitive = str(metadata.get("primitive") or next(iter(sorted(supported)), ""))
    goal = next(
        (
            grammar.atoms[binding.atom].description.rstrip(".")
            for binding in program.bindings
            if grammar.atoms[binding.atom].category == "goal"
        ),
        "complete the simulator-feasible task",
    )
    parameters: list[str] = []
    for binding in program.bindings:
        atom = grammar.atoms[binding.atom]
        for name, value in binding.arguments.items():
            if is_bound_argument(value):
                unit = atom.arguments.get(name, {}).get("unit", "")
                parameters.append(f"{name.replace('_', ' ')}={value}{f' {unit}' if unit else ''}")
    description = f"Train the {args.robot_display_name} for this task: {goal}"
    if parameters:
        description += "; use " + ", ".join(parameters)
    description += "."
    predicates = metadata.get("success_predicates") or metadata.get("constraints", [])
    return {
        "description": description,
        "primitive": primitive,
        "success_predicate": " and ".join(predicates)
        or "the task goal is satisfied without simulator safety termination",
    }
