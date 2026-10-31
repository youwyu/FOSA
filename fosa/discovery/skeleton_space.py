"""The construction space of FOSA's GFlowNet (Sec. 4.2, App. A1): a task skeleton is built from the empty program by
typed edits (add an atom, compose, refine an argument) and ends with STOP once it is complete."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from fosa import utils
from fosa.discovery import common, grammar

SLOTS = ["capability", "goal", "entity_or_environment", "constraint", "condition", "observation", "composition"]
ADDITIONS = {"add_atom", "compose"}  # edits that add a binding; they commute


@dataclass
class Edit:
    action_type: str  # add_atom | compose | refine_arg | stop
    role: str | None = None
    atom: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


def edit_to_dict(edit: Edit) -> dict[str, Any]:
    """The edit as histories and replays store it (their "actions")."""
    return {
        "action_type": edit.action_type,
        "role": edit.role,
        "atom": edit.atom,
        "arguments": utils.json_safe(edit.arguments),
    }


def edit_from_dict(raw: dict[str, Any]) -> Edit:
    return Edit(raw["action_type"], raw["role"], raw["atom"], dict(raw["arguments"]))


def edit_key(edit: Edit) -> str:
    return json.dumps(edit_to_dict(edit), sort_keys=True, separators=(",", ":"))


def first_of_each(edits: list[Edit]) -> list[Edit]:
    """The edits without repeats, each at its first position."""
    unique: dict[str, Edit] = {}
    for edit in edits:
        unique.setdefault(edit_key(edit), edit)
    return list(unique.values())


# ----------------------------------------------------------------------------------------------------------------------
# Programs and the edits valid on them


def empty_program() -> Any:
    return grammar.AtomProgram(bindings=[], open_slots=list(SLOTS), constraints=[], depth=0)


def role_for_atom(atom: Any, slot: str | None = None) -> str:
    if slot:
        return grammar.role_for_slot(slot, atom)
    return {"action": "goal", "difficulty": "condition"}.get(atom.category, atom.category)


def roles_for_atom(atom: Any, atom_grammar: Any) -> list[str]:
    slots = [slot for slot, categories in atom_grammar.slot_rules.items() if atom.category in categories]
    return sorted({role_for_atom(atom)} | {role_for_atom(atom, slot) for slot in slots})


def addition(atom: Any, role: str) -> Edit:
    return Edit("compose" if atom.category == "composition" else "add_atom", role, atom.name)


def _open_slots(roles: set[str], has_constraints: bool) -> list[str]:
    """Slots still unfilled, in grammar order (composition is decided by the caller)."""
    filled = {
        "capability": "capability" in roles,
        "goal": bool(roles & {"goal", "action"}),
        "entity_or_environment": bool(roles & {"entity", "environment"}),
        "constraint": "constraint" in roles or has_constraints,
        "condition": bool(roles & {"condition", "difficulty"}),
        "observation": "observation" in roles,
    }
    return [slot for slot, done in filled.items() if not done]


def next_slots(program: Any) -> list[str]:
    """Slots the next added atom may fill: the capability first, composition only once four atoms are in."""
    if not program.bindings:
        return ["capability"]
    roles = {binding.role for binding in program.bindings}
    slots = _open_slots(roles, bool(program.constraints))
    return slots + ["composition"] if len(program.bindings) >= 4 and "composition" not in roles else slots


def is_partial_program_valid(program: Any, atom_grammar: Any) -> bool:
    """The constraints that can be checked before every required role is filled."""
    names = [binding.atom for binding in program.bindings]
    if len(names) > common.MAX_ATOMS_PER_PROGRAM or len(names) != len(set(names)):
        return False
    if any(name not in atom_grammar.atoms or not atom_grammar.atoms[name].backend_supported for name in names):
        return False
    selected = set(names)
    if any(set(atom_grammar.atoms[name].incompatible_with) & selected for name in names):
        return False
    categories = {atom_grammar.atoms[name].category for name in names}
    return all(
        grammar.required_context_satisfied(requirement, selected, categories, program.open_slots, atom_grammar)
        for name in names
        for requirement in atom_grammar.atoms[name].required_context
    )


def refresh_program_metadata(program: Any, atom_grammar: Any) -> Any:
    constraints = list(program.constraints)
    for binding in program.bindings:
        atom = atom_grammar.atoms[binding.atom]
        if atom.category == "constraint" and atom.description not in constraints:
            constraints.append(atom.description)
    roles = {binding.role for binding in program.bindings}
    open_slots = _open_slots(roles, bool(constraints)) + ([] if "composition" in roles else ["composition"])
    return grammar.AtomProgram(program.bindings, open_slots, constraints, max(program.depth, len(program.bindings)))


def apply_edit(program: Any, edit: Edit, atom_grammar: Any) -> Any:
    """The program after a concrete edit (a refinement names its binding by `target_index` and its new `value`)."""
    new_program = grammar.atom_program_from_dict(grammar.atom_program_to_dict(program))
    if edit.action_type == "stop":
        return new_program
    if edit.action_type in ADDITIONS:
        atom = atom_grammar.atoms[edit.atom]
        arguments = dict(edit.arguments) or grammar.bind_default_arguments(atom)
        new_program.bindings.append(grammar.AtomBinding(atom=edit.atom, role=edit.role, arguments=arguments))
        new_program.depth = max(new_program.depth + 1, len(new_program.bindings))
        return refresh_program_metadata(new_program, atom_grammar)
    new_program.bindings[edit.arguments["target_index"]].arguments[edit.arguments["argument"]] = copy.deepcopy(
        edit.arguments["value"]
    )
    new_program.depth += 1
    return refresh_program_metadata(new_program, atom_grammar)


def valid_edits(program: Any, atom_grammar: Any) -> list[Edit]:
    """Every concrete edit the grammar accepts on the program; neither the GFlowNet nor the LLM goes outside this set.
    STOP is valid once the program is complete, and refinements only then, so every terminal state is a task."""
    if not is_partial_program_valid(program, atom_grammar):
        return []
    complete, _reason = grammar.is_program_complete_for_evaluation(program, atom_grammar)
    edits = [Edit("stop")] if complete else []
    if len(program.bindings) >= common.MAX_ATOMS_PER_PROGRAM or program.depth >= common.MAX_DEPTH:
        return edits
    present = {binding.atom for binding in program.bindings}
    for slot in next_slots(program):
        categories = set(atom_grammar.slot_rules.get(slot, []))
        for atom in atom_grammar.atoms.values():
            if atom.name in present or not atom.backend_supported or atom.category not in categories:
                continue
            edit = addition(atom, role_for_atom(atom, slot))
            if is_partial_program_valid(apply_edit(program, edit, atom_grammar), atom_grammar):
                edits.append(edit)
    if not complete:
        return edits
    for idx, binding in enumerate(program.bindings):
        atom = atom_grammar.atoms[binding.atom]
        for name, value in binding.arguments.items():
            spec = atom.arguments[name]
            if spec.get("kind") in {"continuous", "integer"}:  # one difficulty level down or up
                alternatives = [(d, common.difficulty_step(spec, value, d)) for d in ("low", "high")]
            else:
                alternatives = [(None, item) for item in spec.get("values", [])]
            for direction, alternative in alternatives:
                if alternative is not None and alternative != value:
                    arguments = {
                        "target_index": idx,
                        "argument": name,
                        "value": alternative,
                        **({"direction": direction} if direction else {}),
                    }
                    edits.append(Edit("refine_arg", binding.role, binding.atom, arguments))
    return first_of_each(edits)


def matches(edit: Edit, concrete: Edit) -> bool:
    """Whether an indexed edit is the concrete edit `concrete`; a numeric refinement is indexed by its direction."""
    if edit.action_type != "refine_arg":
        return edit_key(edit) == edit_key(concrete)
    same_argument = (
        concrete.action_type == "refine_arg"
        and edit.role == concrete.role
        and edit.atom == concrete.atom
        and edit.arguments.get("argument") == concrete.arguments.get("argument")
    )
    if "direction" in edit.arguments:
        return same_argument and edit.arguments.get("direction") == concrete.arguments.get("direction")
    return same_argument and edit.arguments.get("value") == concrete.arguments.get("value")


# ----------------------------------------------------------------------------------------------------------------------
# The space


State = tuple[int, ...]  # ids of the edits applied so far, in canonical order


class SkeletonSpace:
    """The construction DAG over one robot's atom grammar.

    Every possible edit has an id (`edits`); the policies output one logit per edit plus STOP (id `stop`). A state is
    the tuple of edit ids applied so far. Additions commute, so they are kept sorted, and different orders of adding the
    same atoms meet in one state; refinements keep the order they were applied in.
    """

    def __init__(self, atom_grammar: Any) -> None:
        self.grammar = atom_grammar
        edits = []
        for atom in atom_grammar.atoms.values():
            if not atom.backend_supported:
                continue
            for role in roles_for_atom(atom, atom_grammar):
                edits.append(addition(atom, role))
                for name, spec in atom.arguments.items():
                    if spec.get("kind") in {"continuous", "integer"}:
                        refinements = [{"argument": name, "direction": d} for d in ("low", "high")]
                    else:
                        refinements = [{"argument": name, "value": v} for v in spec.get("values", [])]
                    edits += [Edit("refine_arg", role, atom.name, r) for r in refinements]
        self.edits = first_of_each(edits)
        self.stop = len(self.edits)
        self.size = len(self.edits) + 1
        self.index = {edit_key(edit): idx for idx, edit in enumerate(self.edits)}
        # a goal family is a goal atom: every complete skeleton has exactly one
        self.goal_families = sorted(
            atom.name for atom in atom_grammar.atoms.values() if atom.backend_supported and atom.category == "goal"
        )

    def edit_id(self, action: dict[str, Any]) -> int:
        """The id of a recorded action."""
        edit = edit_from_dict(action)
        return self.stop if edit.action_type == "stop" else self.index[edit_key(edit)]

    def action(self, edit_id: int) -> dict[str, Any]:
        """The recorded action of an edit id."""
        return edit_to_dict(Edit("stop") if edit_id == self.stop else self.edits[edit_id])

    def step(self, state: State, edit_id: int) -> State:
        ids = [*state, edit_id]
        additions = sorted(
            (i for i in ids if self.edits[i].action_type in ADDITIONS),
            key=lambda i: (self.edits[i].role, self.edits[i].atom, edit_key(self.edits[i])),
        )
        return tuple(additions + [i for i in ids if i not in additions])

    def program(self, state: State) -> Any:
        program = empty_program()
        for edit_id in state:
            program = self._apply(program, self.edits[edit_id])
        return program

    def _apply(self, program: Any, edit: Edit) -> Any:
        """Apply an indexed edit: a refinement targets the binding of its atom and, if numeric, moves one level."""
        if edit.action_type != "refine_arg":
            return apply_edit(program, edit, self.grammar)
        for idx, binding in enumerate(program.bindings):
            if binding.atom != edit.atom:
                continue
            arguments = {"target_index": idx, **edit.arguments}
            if "direction" in edit.arguments:
                name = edit.arguments["argument"]
                arguments["value"] = common.difficulty_step(
                    self.grammar.atoms[binding.atom].arguments[name],
                    binding.arguments[name],
                    edit.arguments["direction"],
                )
                if arguments["value"] is None:
                    return program
            return apply_edit(program, Edit("refine_arg", binding.role, binding.atom, arguments), self.grammar)
        return program

    def valid(self, state: State, family: str | None = None, arm: dict[str, Any] | None = None) -> list[int]:
        """Valid next edits, STOP last. Inside a goal family, other goals and atoms incompatible with the family's goal
        are excluded (they lead to dead ends); inside an arm, so are refinements moving its categorical values
        elsewhere."""
        concrete = valid_edits(self.program(state), self.grammar)
        ids = [
            idx
            for idx, edit in enumerate(self.edits)
            if any(matches(edit, c) for c in concrete)
            and (family is None or not self._outside(edit, family, arm or {}))
        ]
        return ids + [self.stop] if any(c.action_type == "stop" for c in concrete) else ids

    def _outside(self, edit: Edit, family: str, arm: dict[str, Any]) -> bool:
        """Whether a drawn goal family and arm exclude the edit."""
        atom, goal = self.grammar.atoms[edit.atom], self.grammar.atoms[family]
        if atom.category == "goal" and atom.name != family:
            return True  # another goal
        if edit.action_type in ADDITIONS and (
            atom.name in goal.incompatible_with or goal.name in atom.incompatible_with
        ):
            return True  # an atom the goal cannot be combined with
        name = edit.arguments.get("argument")
        return atom.name == family and name in arm and edit.arguments.get("value") != arm[name]  # off the arm's value

    def parents(self, state: State) -> list[tuple[State, int]]:
        """The (parent, edit) pairs with step(parent, edit) == state: the support of P_B. Only the last refinement can
        be undone, and an atom only once its refinements are."""
        found = []
        for position, removed in enumerate(state):
            parent, edit = state[:position] + state[position + 1 :], self.edits[removed]
            if self.step(parent, removed) != state:
                continue  # a refinement followed by others
            if (parent, removed) in found:
                continue  # the same numeric refinement applied twice in a row: undoing either copy gives one parent
            refined = {self.edits[i].atom for i in parent if self.edits[i].action_type == "refine_arg"}
            if edit.action_type in ADDITIONS and edit.atom in refined:
                continue  # an atom that still has refinements
            if any(matches(edit, c) for c in valid_edits(self.program(parent), self.grammar)):
                found.append((parent, removed))
        return found

    def build_order(self, state: State) -> list[int] | None:
        """An order in which the grammar accepts the state's edits from the empty program; None if there is none."""
        current, order, remaining = (), [], list(state)
        while remaining:
            valid = set(self.valid(current))
            edit_id = next((i for i in remaining if i in valid), None)
            if edit_id is None:
                return None
            current = self.step(current, edit_id)
            order.append(edit_id)
            remaining.remove(edit_id)
        return order

    def replay(self, actions: list[dict[str, Any]]) -> tuple[list[State], list[int], list[list[int]]] | None:
        """Rebuild a recorded construction: the states s_0..s_n it visits before STOP, the edits e_0..e_n-1 between
        them, and the valid edits at each state. An edit not valid where it is replayed is skipped; None if STOP never
        is."""
        state: State = ()
        states, edits, valid = [], [], []
        for action in actions:
            edit_id, ids = self.edit_id(action), self.valid(state)
            if edit_id not in ids:
                continue
            states.append(state)
            valid.append(ids)
            if edit_id == self.stop:
                return states, edits, valid
            edits.append(edit_id)
            state = self.step(state, edit_id)
        return None
