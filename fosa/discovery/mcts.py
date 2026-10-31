"""MCTS, LLM-MCTS and MCTS-Learn: the tree-search baselines over FOSA's atom grammar (paper appendix, Baseline
Implementation Details).

A tree node is a program (task skeleton), partial or complete. Selection is UCT with progressive widening; a complete
selected program is verbalized, compiled into an environment and trained with PPO, and its utility V is backed up
along its path.

  mcts      children are grammar edits: an atom added to an open slot, or an argument refined
  llm_mcts  the LLM proposes the children; grammar edits are the fallback when none of its proposals survives
  muzero    MCTS-Learn: MuZero-style networks over edit operators, trained on the tree, score nodes and children

usage: python -m fosa.discovery.mcts --robot ROBOT --method {mcts,llm_mcts,muzero} [--iterations N] ...
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fosa import llm, utils
from fosa.compiler import generate_env
from fosa.discovery import capability_context, common, grammar, objectives

METHODS = ("mcts", "llm_mcts", "muzero")
LOG_PREFIX = "[mcts]"
SCENE_COMPILE_WORKERS = 4  # compile threads per run
OPEN_SLOTS = ["capability", "goal", "entity_or_environment", "constraint", "condition", "observation"]
IN_FLIGHT_STATUSES = {"training", "reserved"}
SELECTABLE_STATUSES = {"unvisited", "frontier", "mastered", "failed"}
FAILED_RUN_STATUSES = {"train_failed", "timeout"}
FAILED_RUN_REWARD = -0.25  # backed up for a run that produced no result
UCT_EXPLORATION = 1.4
CHILDREN_PER_EXPANSION = 4
LLM_CHILDREN = 8  # proposals asked of the LLM per expansion

# MCTS-Learn's action space: edit-operator types, so its size fixes the network shapes. An edit maps to the first
# type its operator names or contains; MCTS-Learn's own edits use five of them.
MUZERO_OPERATORS = (
    "add_atom",
    "add_goal",
    "bind_entity",
    "refine_difficulty",
    "add_constraint",
    "compose_skill",
    "harden",
    "simplify",
    "refine_numeric_argument",
    "grammar_expansion",
    "vlm_child_proposal",
)
LLM_MCTS_SYSTEM_PROMPT = "You propose concrete robot RL child tasks for atom-program MCTS. Output JSON only."
LLM_MCTS_INSTRUCTIONS = [
    "Output JSON only with up to 8 concrete child tasks.",
    "Propose concrete child tasks, not abstract task families or capability labels.",
    "Each proposal must be a one-step semantic expansion of the parent region.",
    "Preserve the parent's core capability unless the operator is compose_skill or simplify.",
    "Use only supported backend primitives/assets.",
    "Avoid duplicate descriptions from recent history.",
    "Avoid vague tasks like 'perform contact control', 'perform object state control', "
    "'perform an in-hand task', or 'perform a grasp'.",
    "Each proposal must include a measurable success predicate.",
    "Each proposal should be feasible for the robot morphology.",
    "Use the provided grammar atoms; if an atom is missing, choose the closest existing atom.",
]
LLM_MCTS_CHILD_SCHEMA = {
    "description": "one concrete natural-language task",
    "primitive": "supported primitive",
    "success_predicate": "measurable success condition",
    "atoms": [
        {
            "atom": "atom_name",
            "role": "capability|goal|entity|environment|constraint|condition|observation|composition",
            "arguments": {},
        }
    ],
    "difficulty": 0.0,
    "rationale": "why this is a useful child of the parent",
    "operator": "add_goal|bind_entity|refine_difficulty|add_constraint|compose_skill|harden|simplify",
}

MUZERO_HIDDEN_DIM = 128
MUZERO_LEARNING_RATE = 3e-4
MUZERO_DISCOUNT = 0.997
MUZERO_SIMULATIONS = 32
MUZERO_EXPLORATION = 1.5
MUZERO_TEMPERATURE = 0.25
MUZERO_BATCH_SIZE = 32
MUZERO_EPOCHS = 8  # passes over the replay each time it changes
MUZERO_MIN_REPLAY = 4  # transitions before the networks are first trained
FEATURE_DEPTH_SCALE = 6.0
FEATURE_ATOM_SCALE = 8.0


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    children: list[str]
    program: grammar.AtomProgram
    depth: int
    visits: int = 0
    value_sum: float = 0.0
    value_mean: float = 0.0
    policy_success_mean: float | None = None  # mean success rate of the tasks trained at or below the node
    status: str = "frontier"  # unvisited, frontier, mastered, failed, blocked, reserved or training
    created_from: dict[str, Any] = field(default_factory=dict)  # the edit or LLM proposal that made the node
    last_result: dict[str, Any] | None = None  # the record of the node's latest task


@dataclass(kw_only=True)
class MctsRecord(common.TaskRecord):
    node_id: str
    parent_id: str | None
    path: list[str]
    selected_by: str
    mcts_score: float | None
    atom_program: dict[str, Any]
    atom_names: list[str]
    success_predicate: str | None
    task_id: str | None
    proposal_description: str | None
    proposal_rationale: str | None
    proposal_operator: str | None
    completeness_score: float | None
    duplicate_score: float | None
    main_action: str | None
    main_entity: str | None
    main_condition: str | None
    atom_pair_keys: list[str]
    policy_score: float | None = None
    frontier_bonus: float | None = None
    novelty_bonus: float | None = None
    cost_penalty: float | None = None
    mcts_reward: float | None = None  # V clipped to [-1, 1], backed up along the path


def clip(value: float) -> float:
    return min(max(value, -1.0), 1.0)


# ----------------------------------------------------------------------------------------------------------------------
# MCTS-Learn's networks


class MuZeroNetworks(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.representation = nn.Sequential(  # h
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.dynamics = nn.Sequential(  # g
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.reward = nn.Linear(hidden_dim, 1)
        self.policy = nn.Linear(hidden_dim, action_dim)  # f: policy and value
        self.value = nn.Linear(hidden_dim, 1)


@dataclass
class MuZeroNode:
    latent: torch.Tensor
    prior: float = 1.0
    reward: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, MuZeroNode] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class NeuralMuZero:
    """MuZero's representation, dynamics and prediction networks, a latent PUCT search, and one-step-unroll
    training."""

    def __init__(self, observation_dim: int, action_dim: int, seed: int) -> None:
        torch.manual_seed(seed)
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.networks = MuZeroNetworks(observation_dim, action_dim, MUZERO_HIDDEN_DIM).to(self.device)
        self.optimizer = torch.optim.Adam(self.networks.parameters(), lr=MUZERO_LEARNING_RATE)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"networks": self.networks.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.networks.load_state_dict(payload["networks"])
        self.optimizer.load_state_dict(payload["optimizer"])

    def _tensor(self, observation: Any) -> torch.Tensor:
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device).flatten()

    def _predict(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.networks.policy(latent), torch.tanh(self.networks.value(latent)).squeeze(-1)

    def _recurrent(self, latent: torch.Tensor, action: int) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = nn.functional.one_hot(torch.tensor(action, device=self.device), self.action_dim).to(latent.dtype)
        next_latent = self.networks.dynamics(torch.cat((latent, one_hot)))
        return next_latent, torch.tanh(self.networks.reward(next_latent)).squeeze(-1)

    def evaluate_action(self, observation: list[float], action: int) -> tuple[float, float]:
        """Predicted reward of an edit and value of the program it leads to."""
        with torch.no_grad():
            latent = self.networks.representation(self._tensor(observation))
            next_latent, reward = self._recurrent(latent, action)
            _, value = self._predict(next_latent)
        return float(reward.cpu()), float(value.cpu())

    def _expand(self, node: MuZeroNode, legal_actions: list[int]) -> float:
        logits, value = self._predict(node.latent)
        actions = torch.tensor(legal_actions, device=self.device)
        priors = torch.softmax(logits[actions], dim=0).detach().cpu().tolist()
        one_hot = nn.functional.one_hot(actions, self.action_dim).to(node.latent.dtype)
        parents = node.latent.unsqueeze(0).expand(len(legal_actions), -1)
        latents = self.networks.dynamics(torch.cat((parents, one_hot), dim=1))
        rewards = torch.tanh(self.networks.reward(latents)).squeeze(-1)
        for action, prior, latent, reward in zip(legal_actions, priors, latents, rewards, strict=True):
            node.children[action] = MuZeroNode(latent.detach(), float(prior), float(reward.detach().cpu()))
        return float(value.detach().cpu())

    def search(self, observation: list[float], legal_actions: list[int]) -> tuple[int, list[float]]:
        """Latent PUCT from the observed program: a sampled action and the root's visit distribution."""
        with torch.no_grad():
            root = MuZeroNode(self.networks.representation(self._tensor(observation)))
            self._expand(root, legal_actions)
            for _ in range(MUZERO_SIMULATIONS):
                node, path = root, [root]
                while node.children:
                    sqrt_parent = math.sqrt(node.visits + 1.0)
                    node = max(
                        node.children.values(),
                        key=lambda child: child.value
                        + MUZERO_EXPLORATION * child.prior * sqrt_parent / (child.visits + 1),
                    )
                    path.append(node)
                backed = self._expand(node, list(range(self.action_dim)))
                for visited in reversed(path):
                    visited.visits += 1
                    visited.value_sum += backed
                    backed = visited.reward + MUZERO_DISCOUNT * backed
            visits = [0.0] * self.action_dim
            for action, child in root.children.items():
                visits[action] = float(child.visits)
            total = sum(visits)
            policy = [count / total for count in visits]
            actions = list(root.children)
            weights = torch.tensor(
                [max(1e-8, root.children[item].visits ** (1.0 / MUZERO_TEMPERATURE)) for item in actions],
                device=self.device,
            )
            return actions[int(torch.multinomial(weights, 1).item())], policy

    def train_replay(self, samples: list[dict[str, Any]]) -> None:
        """One pass over the transitions in minibatches: policy, value and reward losses, plus a latent-consistency
        loss between the predicted next latent and the representation of the next program."""
        for start in range(0, len(samples), MUZERO_BATCH_SIZE):
            chunk = samples[start : start + MUZERO_BATCH_SIZE]
            loss = torch.tensor(0.0, device=self.device)
            for sample in chunk:
                latent = self.networks.representation(self._tensor(sample["observation"]))
                logits, value = self._predict(latent)
                policy = self._tensor(sample["policy"])
                loss = loss - (policy * torch.log_softmax(logits, dim=0)).sum()
                loss = loss + (value - float(sample["value"])).square()
                next_latent, reward = self._recurrent(latent, int(sample["action"]))
                loss = loss + (reward - float(sample["reward"])).square()
                target = self.networks.representation(self._tensor(sample["next_observation"])).detach()
                loss = loss + 0.25 * nn.functional.mse_loss(next_latent, target)
            loss = loss / len(chunk)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.networks.parameters(), 5.0)
            self.optimizer.step()


def program_features(program: grammar.AtomProgram, atom_grammar: grammar.AtomGrammar) -> list[float]:
    """The networks' observation of a program: size, completeness, atom-category counts and which atoms it has."""
    names = set(grammar.atom_names(program))
    categories = sorted(grammar.ALLOWED_ATOM_CATEGORIES)
    counts = dict.fromkeys(categories, 0.0)
    for binding in program.bindings:
        counts[atom_grammar.atoms[binding.atom].category] += 1.0
    return [
        1.0,
        min(1.0, float(program.depth) / FEATURE_DEPTH_SCALE),
        min(1.0, len(program.bindings) / FEATURE_ATOM_SCALE),
        grammar.program_completeness_score(program),
        1.0 if grammar.is_program_complete_for_evaluation(program, atom_grammar)[0] else 0.0,
        *[min(1.0, counts[category] / FEATURE_ATOM_SCALE) for category in categories],
        *[1.0 if name in names else 0.0 for name in sorted(atom_grammar.atoms)],
    ]


def operator_index(created_from: dict[str, Any]) -> int:
    operator = created_from["operator"].lower().strip()
    return next(
        (index for index, name in enumerate(MUZERO_OPERATORS) if operator == name or name in operator),
        MUZERO_OPERATORS.index("grammar_expansion"),
    )


def row_reward(row: dict[str, Any]) -> float | None:
    if row["status"] == "trained" and row["success_rate"] is None:
        return None  # a trained row without a success rate has no frontier score to learn from
    values = (row.get(key) for key in ("mcts_reward", "policy_score", "normalized_reward_clipped", "normalized_reward"))
    return next((float(value) for value in values if value is not None), None)


def replay_samples(
    tree: dict[str, Node], atom_grammar: grammar.AtomGrammar, history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One transition per node with a result: parent program, edit operator, reward, node value, and the visit
    distribution over the parent's children's operators."""
    history_by_node = {row["node_id"]: row for row in history}
    samples = []
    for node in tree.values():
        if node.parent_id is None:
            continue
        parent = tree[node.parent_id]
        row = node.last_result or history_by_node.get(node.node_id)
        reward = row_reward(row) if row is not None else None
        if reward is None:
            continue
        policy = [0.0] * len(MUZERO_OPERATORS)
        for sibling_id in parent.children:
            policy[operator_index(tree[sibling_id].created_from)] += tree[sibling_id].visits
        action = operator_index(node.created_from)
        if sum(policy) <= 0:
            policy[action] = 1.0
        else:
            total = sum(policy)
            policy = [value / total for value in policy]
        samples.append(
            {
                "node_id": node.node_id,
                "observation": program_features(parent.program, atom_grammar),
                "action": action,
                "reward": clip(reward),
                "value": clip(node.value_mean),
                "policy": policy,
                "next_observation": program_features(node.program, atom_grammar),
            }
        )
    return samples


class MuZeroScorer:
    """MCTS-Learn's learned scores. Before each batch the networks are trained on the tree's transitions if they
    changed; until the networks are first trained (or loaded), plain UCT and the heuristic candidate score are used."""

    def __init__(self, path: Path, atom_grammar: grammar.AtomGrammar) -> None:
        self.path, self.grammar = path, atom_grammar
        self.network: NeuralMuZero | None = None  # built at the first batch
        self.ready = False
        self.replay_signature: tuple[Any, ...] | None = None
        self.policy_cache: dict[tuple[str, tuple[int, ...]], list[float]] = {}
        self.tree: dict[str, Node] = {}

    def prepare(self, tree: dict[str, Node], history: list[dict[str, Any]]) -> None:
        self.tree = tree
        samples = replay_samples(tree, self.grammar, history)
        if self.network is None:
            observation_dim = len(program_features(tree["root"].program, self.grammar))
            self.network = NeuralMuZero(observation_dim, len(MUZERO_OPERATORS), common.SEED)
            if self.path.exists():
                self.network.load(self.path)
                self.ready = True
        signature = tuple((item["node_id"], item["reward"], item["value"], tuple(item["policy"])) for item in samples)
        if len(samples) >= MUZERO_MIN_REPLAY and signature != self.replay_signature:
            rng = random.Random(common.SEED + len(samples))
            for _ in range(MUZERO_EPOCHS):
                rng.shuffle(samples)
                self.network.train_replay(samples)
            self.network.save(self.path)
            self.replay_signature = signature
            self.ready = True
        self.policy_cache = {}

    def policy(self, parent: Node) -> list[float]:
        """The search's visit distribution over operators at parent, cached per batch. (The search also samples an
        action, which is not used.)"""
        legal_actions = sorted(
            {
                operator_index(self.tree[child_id].created_from)
                for child_id in parent.children
                if self.tree[child_id].status in SELECTABLE_STATUSES
            }
        ) or list(range(len(MUZERO_OPERATORS)))
        key = (parent.node_id, tuple(legal_actions))
        if key not in self.policy_cache:
            observation = program_features(parent.program, self.grammar)
            _action, self.policy_cache[key] = self.network.search(observation, legal_actions)
        return self.policy_cache[key]

    def node_score(self, child: Node, parent: Node) -> float:
        """Predicted value, mixed with the empirical one, plus a PUCT term with the search's prior."""
        if not self.ready:
            return standard_uct_score(child, parent)
        policy = self.policy(parent)
        action = operator_index(child.created_from)
        reward, value = self.network.evaluate_action(program_features(parent.program, self.grammar), action)
        predicted = clip(reward + MUZERO_DISCOUNT * value)
        puct = MUZERO_EXPLORATION * policy[action] * math.sqrt(parent.visits + 1.0) / (child.visits + 1.0)
        empirical = child.value_mean if child.visits else 0.0
        return 0.65 * predicted + 0.35 * empirical + puct + (0.05 if child.visits == 0 else 0.0)

    def candidate_score(self, created_from: dict[str, Any], parent: Node, heuristic: float) -> float:
        if not self.ready:
            return heuristic
        observation = program_features(parent.program, self.grammar)
        reward, value = self.network.evaluate_action(observation, operator_index(created_from))
        return 0.75 * clip(reward + MUZERO_DISCOUNT * value) + 0.25 * heuristic


# ----------------------------------------------------------------------------------------------------------------------
# The tree


def make_node(parent_id: str | None, program: grammar.AtomProgram, created_from: dict[str, Any], status: str) -> Node:
    node_id = "root" if parent_id is None else f"node_{grammar.atom_program_key(program)[:16]}"
    return Node(node_id, parent_id, [], program, program.depth, status=status, created_from=created_from)


def initialize_tree(atom_grammar: grammar.AtomGrammar) -> dict[str, Node]:
    """The empty program, with a child per start symbol: a capability atom (one per execution primitive)."""
    constraints = atom_grammar.global_constraints
    root = make_node(
        None, grammar.AtomProgram([], list(OPEN_SLOTS), list(constraints), 0), {"operator": "root"}, "frontier"
    )
    tree = {"root": root}
    for name in atom_grammar.start_symbols:
        binding = grammar.AtomBinding(name, "capability", grammar.bind_default_arguments(atom_grammar.atoms[name]))
        program = grammar.AtomProgram([binding], OPEN_SLOTS[1:], list(constraints), 1)
        node = make_node("root", program, {"operator": "initialize_start_symbol", "atom": name}, "unvisited")
        tree[node.node_id] = node
        root.children.append(node.node_id)
    return tree


def save_tree(tree: dict[str, Node], atom_stats: dict[str, grammar.AtomStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": {node_id: utils.json_safe(asdict(node)) for node_id, node in sorted(tree.items())},
        "atom_stats": {name: utils.json_safe(asdict(stats)) for name, stats in sorted(atom_stats.items())},
    }
    llm.write_json_atomic(path, payload)  # a failed write must not leave an empty tree behind


def load_tree(path: Path) -> tuple[dict[str, Node], dict[str, grammar.AtomStats]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tree = {
        node_id: Node(**{**node, "program": grammar.atom_program_from_dict(node["program"])})
        for node_id, node in raw["nodes"].items()
    }
    return tree, {name: grammar.AtomStats(**stats) for name, stats in raw["atom_stats"].items()}


def reset_in_flight_nodes(tree: dict[str, Node]) -> None:
    """Return the nodes a stopped run left reserved or training to the frontier."""
    for node in tree.values():
        if node.status in IN_FLIGHT_STATUSES:
            node.status = "frontier"


def reopen_selectable_ancestors(tree: dict[str, Node]) -> None:
    """Unblock the ancestors of selectable nodes, so selection can reach them again."""
    for node in tree.values():
        if node.status not in SELECTABLE_STATUSES:
            continue
        parent_id = node.parent_id
        while parent_id is not None:
            parent = tree[parent_id]
            if parent.status == "blocked":
                parent.status = "frontier"
                parent.created_from.pop("blocked_reason", None)
            parent_id = parent.parent_id


def path_to_node(tree: dict[str, Node], node_id: str) -> list[str]:
    path: list[str] = []
    current_id: str | None = node_id
    while current_id:
        path.append(current_id)
        current_id = tree[current_id].parent_id
    return list(reversed(path))


def trained_program_keys(history: list[dict[str, Any]]) -> set[str]:
    return {
        grammar.atom_program_key(grammar.atom_program_from_dict(row["atom_program"]))
        for row in history
        if row["status"] == "trained"
    }


def is_untrained(node: Node, trained: set[str]) -> bool:
    # one task per program for every method; a trained node can still be widened
    return node.visits == 0 and grammar.atom_program_key(node.program) not in trained


def can_progressively_widen(node: Node) -> bool:
    if node.parent_id is None or node.depth >= common.MAX_DEPTH:
        return False
    if len(node.program.bindings) >= common.MAX_ATOMS_PER_PROGRAM:
        return False
    return len(node.children) < max(1.0, 2.0 * math.sqrt(node.visits + 1))


def best_complete_child(children: list[Node], atom_grammar: grammar.AtomGrammar) -> Node | None:
    complete = [
        child
        for child in children
        if child.status in SELECTABLE_STATUSES
        and grammar.is_program_complete_for_evaluation(child.program, atom_grammar)[0]
    ]
    return max(complete, key=lambda item: item.created_from.get("candidate_score", 0.0)) if complete else None


def standard_uct_score(child: Node, parent: Node) -> float:
    """UCT, plus a bonus for an unvisited child and one for a child whose tasks succeed about half the time."""
    exploit = child.value_mean
    explore = UCT_EXPLORATION * math.sqrt(math.log(parent.visits + 1.0) / (child.visits + 1.0))
    unvisited_bonus = 0.25 if child.visits == 0 else 0.0
    frontier_bonus = 0.0
    if child.policy_success_mean is not None:
        frontier_bonus = 0.15 * objectives.success_uncertainty(child.policy_success_mean)
    return exploit + explore + unvisited_bonus + frontier_bonus


class TreeSearch:
    """Selection, expansion and backup on one run's tree. Also holds what they read: the run's arguments and grammar,
    the per-atom statistics, the history and, for MCTS-Learn, the learned scorer."""

    def __init__(
        self,
        args: argparse.Namespace,
        tree: dict[str, Node],
        atom_stats: dict[str, grammar.AtomStats],
        history: list[dict[str, Any]],
        tree_path: Path,
        muzero: MuZeroScorer | None,
    ) -> None:
        self.args, self.grammar = args, args.atom_grammar
        self.tree, self.atom_stats, self.history = tree, atom_stats, history
        self.tree_path, self.muzero = tree_path, muzero

    def save(self) -> None:
        save_tree(self.tree, self.atom_stats, self.tree_path)

    def is_complete(self, program: grammar.AtomProgram) -> bool:
        return grammar.is_program_complete_for_evaluation(program, self.grammar)[0]

    def uct_score(self, child: Node, parent: Node) -> float:
        return standard_uct_score(child, parent) if self.muzero is None else self.muzero.node_score(child, parent)

    # Selection

    def select_batch(self, batch_size: int) -> list[tuple[list[str], Node, str, float | None]]:
        """Up to batch_size distinct nodes to train, each with its path, how it was selected and its score."""
        if self.muzero is not None:
            self.muzero.prepare(self.tree, self.history)
        selections = []
        for _ in range(batch_size):
            selection = self.select()
            if selection is None:
                break
            node, selected_by, score = selection
            node.status = "reserved"
            selections.append((path_to_node(self.tree, node.node_id), node, selected_by, score))
        return selections

    def select(self) -> tuple[Node, str, float | None] | None:
        """Descend from the root to an untrained complete program, widening nodes on the way and blocking the ones
        that cannot lead to one; None when no such program is left."""
        tree = self.tree
        trained = trained_program_keys(self.history)
        exhausted: set[str] = set()  # nodes with nothing selectable below them during this descent

        def block(node: Node, reason: str | None) -> Node:
            node.status = "blocked"
            node.created_from["blocked_reason"] = reason
            return tree[node.parent_id]

        node = tree["root"]
        while True:
            if node.node_id != "root":
                valid, reason = grammar.is_program_valid(node.program, self.grammar)
                if not valid:
                    node = block(node, reason)
                    continue
                if node.status in {"unvisited", "failed"} and node.visits == 0:
                    complete, reason = grammar.is_program_complete_for_evaluation(node.program, self.grammar)
                    if complete:
                        return node, "unvisited_complete", None
                    if node.depth < common.MAX_DEPTH and can_progressively_widen(node):
                        child = best_complete_child(self.expand(node), self.grammar)
                        if child is not None:
                            return child, "expanded_complete", None
                    if node.depth >= common.MAX_DEPTH:
                        node = block(node, reason)
                        continue

            if can_progressively_widen(node):
                child = best_complete_child(self.expand(node), self.grammar)
                if child is not None:
                    return child, "expanded_complete", None

            children = [tree[child_id] for child_id in node.children]
            selectable = [
                child for child in children if child.node_id not in exhausted and child.status in SELECTABLE_STATUSES
            ]
            if not selectable:
                if node.node_id == "root" or node.status not in SELECTABLE_STATUSES:
                    recovered = self.recover_evaluable_node(trained)
                    return None if recovered is None else (recovered[0], "recovered_evaluable", recovered[1])
                complete, reason = grammar.is_program_complete_for_evaluation(node.program, self.grammar)
                if complete and is_untrained(node, trained):
                    return node, "leaf_complete", None
                in_flight = any(child.node_id in exhausted or child.status in IN_FLIGHT_STATUSES for child in children)
                if complete or in_flight:  # nothing to train here now: look elsewhere
                    exhausted.add(node.node_id)
                    node = tree[node.parent_id]
                else:
                    node = block(node, reason)
                continue

            unvisited = [child for child in selectable if child.visits == 0 and child.status == "unvisited"]
            if unvisited:
                complete_unvisited = [child for child in unvisited if self.is_complete(child.program)]
                if complete_unvisited:  # the best-scored one
                    best = min(
                        complete_unvisited,
                        key=lambda item: (-item.created_from.get("candidate_score", 0.0), item.node_id),
                    )
                    return best, "unvisited_complete", None
                node = min(unvisited, key=lambda item: item.node_id)
                continue

            scored = [(self.uct_score(child, node), child) for child in selectable]
            score, node = max(scored, key=lambda item: item[0])
            if node.children and node.depth < common.MAX_DEPTH:
                continue
            if not is_untrained(node, trained):  # descend: the loop above widens it or climbs back
                continue
            return node, "uct", score

    def recover_evaluable_node(self, trained: set[str]) -> tuple[Node, float] | None:
        """The best-scoring untrained complete node anywhere in the tree, for when the descent finds none."""
        candidates = [
            node
            for node in self.tree.values()
            if node.parent_id is not None
            and node.status in SELECTABLE_STATUSES
            and is_untrained(node, trained)
            and self.is_complete(node.program)
        ]
        if not candidates:
            return None
        scored = [(self.uct_score(node, self.tree[node.parent_id]), node) for node in candidates]
        score, node = max(scored, key=lambda item: (item[0], item[1].node_id))
        return node, score

    # Expansion

    def expand(self, node: Node) -> list[Node]:
        """Add the best new candidate children to node; returns them."""
        children = []
        for program, created_from in self.expansion_candidates(node)[:CHILDREN_PER_EXPANSION]:
            valid, reason = grammar.is_program_valid(program, self.grammar)
            complete, complete_reason = grammar.is_program_complete_for_evaluation(program, self.grammar)
            if valid and complete:
                status = "frontier"
            elif valid and program.depth < common.MAX_DEPTH:
                status = "unvisited"
            else:
                status = "blocked"
            created_from = dict(created_from)
            created_from.setdefault("completeness_score", grammar.program_completeness_score(program))
            if reason:
                created_from["blocked_reason"] = reason
            elif complete_reason:
                created_from["incomplete_reason"] = complete_reason
            child = make_node(node.node_id, program, created_from, status)
            if child.node_id in self.tree:
                continue
            self.tree[child.node_id] = child
            node.children.append(child.node_id)
            children.append(child)
        return children

    def expansion_candidates(self, node: Node) -> list[tuple[grammar.AtomProgram, dict[str, Any]]]:
        """Candidate children, best first: for LLM-MCTS the LLM's proposals, else (or when none survives) grammar
        edits. Invalid, unsupported and duplicate candidates are dropped."""
        programs = [item.program for item in self.tree.values()]
        ranked: list[tuple[float, grammar.AtomProgram, dict[str, Any]]] = []
        if self.args.method == "llm_mcts":
            tokens = set(grammar.backend_capability_tokens(self.args))
            for proposal in self.llm_children(node):
                description = str(proposal.get("description") or "")
                if capability_context.missing_runtime_capability(description, tokens, category="task"):
                    continue
                program, created_from = program_from_llm_proposal(proposal, node.program, self.grammar)
                if not grammar.is_program_valid(program, self.grammar)[0]:
                    continue
                duplicate, duplicate_score, _ = grammar.is_duplicate_program_or_task(
                    program, description, programs, self.history, self.grammar
                )
                created_from["duplicate_score"] = duplicate_score
                if duplicate:
                    continue
                score = self.acquisition_score(program, created_from, node)
                created_from["candidate_score"] = score
                ranked.append((score, program, created_from))

        if not ranked:
            parent_completeness = grammar.program_completeness_score(node.program)
            for program, created_from in grammar_expansion_candidates(node, self.grammar):
                if grammar.program_completeness_score(program) + 1e-9 < parent_completeness:
                    continue
                duplicate, duplicate_score, _ = grammar.is_duplicate_program_or_task(
                    program, created_from.get("proposal_description"), programs, self.history, self.grammar
                )
                if duplicate:
                    continue
                created_from = {
                    "proposal_source": "grammar_expansion",
                    **created_from,
                    "duplicate_score": duplicate_score,
                }
                score = self.acquisition_score(program, created_from, node)
                created_from["candidate_score"] = score
                ranked.append((score, program, created_from))
                if len(ranked) >= CHILDREN_PER_EXPANSION:
                    break

        ranked.sort(key=lambda item: item[0], reverse=True)
        return dedupe_program_candidates([(program, created_from) for _score, program, created_from in ranked])

    def acquisition_score(self, program: grammar.AtomProgram, created_from: dict[str, Any], parent: Node) -> float:
        """How promising a candidate child is: complete, novel, filling the parent's missing slots, keeping its atoms,
        with a measurable success predicate; not a duplicate, vague or unsupported. MCTS-Learn mixes in its model."""
        completeness = grammar.program_completeness_score(program)
        locality = 0.5 if set(grammar.atom_names(parent.program)) - set(grammar.atom_names(program)) else 1.0
        measurable = 1.0 if len(created_from.get("proposal_success_predicate", "").split()) >= 4 else 0.0
        missing_before = set(missing_required_slots(parent.program, self.grammar))
        fills_missing = len(missing_before - set(missing_required_slots(program, self.grammar)))
        pair_novelty = grammar.atom_pair_novelty(program, self.history)
        novelty = (
            atom_combination_novelty(program, self.atom_stats, self.history) * (1.0 - common.NOVELTY_PAIR_WEIGHT)
            + pair_novelty * common.NOVELTY_PAIR_WEIGHT
        )
        description = created_from.get("proposal_description") or ""
        vague_penalty = 0.30 if grammar.is_vague_abstract_description(description) else 0.0
        atoms = self.grammar.atoms
        unsupported = any(b.atom not in atoms or not atoms[b.atom].backend_supported for b in program.bindings)
        unsupported_penalty = 0.25 if unsupported else 0.0
        heuristic = (
            0.35 * completeness
            + 0.20 * novelty
            + 0.15 * min(1.0, fills_missing / 3.0)
            + 0.15 * locality
            + 0.10 * measurable
            - 0.25 * created_from["duplicate_score"]
            - vague_penalty
            - unsupported_penalty
        )
        return heuristic if self.muzero is None else self.muzero.candidate_score(created_from, parent, heuristic)

    def llm_children(self, parent: Node) -> list[dict[str, Any]]:
        """LLM-MCTS: up to LLM_CHILDREN child tasks the LLM proposes under parent."""
        history = self.history
        failed_recent = [
            {
                "description": row.get("description"),
                "status": row.get("status"),
                "atoms": row.get("atom_names"),
                "reward": row.get("mcts_reward"),
            }
            for row in history[-common.MAX_HISTORY_ITEMS :]
            if row.get("status") in FAILED_RUN_STATUSES
        ]
        top_atoms = sorted(
            (
                {
                    "atom": name,
                    "visits": stats.visits,
                    "value_mean": stats.value_mean,
                    "success_mean": stats.success_mean,
                }
                for name, stats in self.atom_stats.items()
            ),
            key=lambda item: (float(item["value_mean"]), int(item["visits"])),
            reverse=True,
        )[:20]
        prompt = {
            "request": "Propose concrete child robot tasks under the current atom-program MCTS node.",
            "num_children": LLM_CHILDREN,
            "robot": self.args.robot,
            "robot_display_name": self.args.robot_display_name,
            "backend": self.args.backend,
            "backend_capabilities": self.args.capability_context["backend_capabilities"],
            "parent_atom_program": grammar.atom_program_to_dict(parent.program),
            "parent_atom_names": grammar.atom_names(parent.program),
            "parent_node": {
                "status": parent.status,
                "visits": parent.visits,
                "value_mean": parent.value_mean,
                "policy_success_mean": parent.policy_success_mean,
                "summary": grammar.program_summary(parent.program),
            },
            "recent_task_descriptions": [
                row["description"] for row in history[-common.RECENT_DUPLICATE_WINDOW :] if row["description"]
            ],
            "high_value_atoms": top_atoms,
            "failed_recent_tasks": failed_recent[:20],
            "grammar_atoms": [
                {
                    "name": atom.name,
                    "category": atom.category,
                    "description": atom.description,
                    "arguments": list(atom.arguments.keys()),
                    "required_context": atom.required_context,
                }
                for atom in self.grammar.atoms.values()
                if atom.backend_supported
            ],
            "instructions": LLM_MCTS_INSTRUCTIONS,
            "schema": {"children": [LLM_MCTS_CHILD_SCHEMA]},
        }
        messages = [
            {"role": "system", "content": LLM_MCTS_SYSTEM_PROMPT},
            {"role": "user", "content": llm.compact_json(prompt)},
        ]
        children = grammar.request_json(messages, self.args, max_tokens=2048).get("children")
        if not isinstance(children, list):
            return []
        return [child for child in children if isinstance(child, dict)][:LLM_CHILDREN]

    # Backup

    def record_result(self, job: common.TrainingJob) -> None:
        """Read a finished PPO run into its record, back its utility up the tree, and log it."""
        record = job.record
        common.finish_training(job, reward_budget=record.reward_budget)
        record.mcts_reward = self.task_utility(record)
        self.backup(record)
        common.append_history(record, self.history, self.args.history)
        self.save()

    def task_utility(self, record: MctsRecord) -> float | None:
        """V (Eq. 7), the same capability-frontier utility as FOSA, clipped to [-1, 1]; FAILED_RUN_REWARD for a run
        that produced nothing; None for a run without a success rate."""
        if record.status in FAILED_RUN_STATUSES and record.raw_reward is None:
            record.policy_score, record.frontier_bonus, record.novelty_bonus, record.cost_penalty = None, 0.0, 0.0, 1.0
            return FAILED_RUN_REWARD
        if record.success_rate is None:
            return None
        program = grammar.atom_program_from_dict(record.atom_program)
        programs = [node.program for node in self.tree.values()]
        record.novelty_bonus = grammar.novelty_bonus_from_program(
            program, self.atom_stats, self.history, programs, self.grammar
        )
        record.cost_penalty = objectives.program_cost(len(record.atom_names), record.status)
        record.policy_score = record.success_rate
        record.frontier_bonus = objectives.frontier_score(record.success_curve)
        return clip(objectives.task_utility(record.success_curve, record.novelty_bonus, record.cost_penalty))

    def backup(self, record: MctsRecord) -> None:
        """Add the record's utility (and success rate) to every node on its path and to its atoms' statistics, and
        set the trained node's status from its success rate."""
        value = record.mcts_reward if record.mcts_reward is not None else 0.0
        success = record.policy_score
        for node_id in record.path:
            node = self.tree[node_id]
            node.visits += 1
            node.value_sum += value
            node.value_mean = node.value_sum / node.visits
            if success is not None:
                previous = node.policy_success_mean if node.policy_success_mean is not None else 0.0
                node.policy_success_mean = (previous * (node.visits - 1) + success) / node.visits

        selected = self.tree[record.node_id]
        selected.last_result = utils.json_safe(asdict(record))
        if record.status in FAILED_RUN_STATUSES:
            selected.status = "failed"
        elif record.status == "trained":
            selected.status = status_after_training(success)
        elif selected.status == "training":
            selected.status = "frontier"

        for name in record.atom_names:
            stats = self.atom_stats.setdefault(name, grammar.AtomStats())
            stats.visits += 1
            stats.value_sum += value
            stats.value_mean = stats.value_sum / stats.visits
            if success is not None:
                stats.success_sum += success
                stats.success_mean = stats.success_sum / stats.visits


def status_after_training(success: float | None) -> str:
    """mastered when the task policy succeeds at least SUCCESS_THRESHOLD of the time, failed when at most
    1 - SUCCESS_THRESHOLD of the time, else (or without a success rate) frontier."""
    if success is None:
        return "frontier"
    if success >= common.SUCCESS_THRESHOLD:
        return "mastered"
    return "frontier" if success > 1.0 - common.SUCCESS_THRESHOLD else "failed"


# ----------------------------------------------------------------------------------------------------------------------
# Program edits


def atom_combination_novelty(
    program: grammar.AtomProgram, atom_stats: dict[str, grammar.AtomStats], history: list[dict[str, Any]]
) -> float:
    single = grammar.atom_visit_novelty(grammar.atom_names(program), atom_stats)
    if not grammar.atom_pair_keys(program):
        return single
    return 0.5 * single + 0.5 * grammar.atom_pair_novelty(program, history)


def grammar_expansion_candidates(
    node: Node, atom_grammar: grammar.AtomGrammar
) -> list[tuple[grammar.AtomProgram, dict[str, Any]]]:
    """Grammar edits of the node's program: the required context it misses, else atoms added to open slots (missing
    required slots first, in a random order seeded by the node), else hardened, simplified or refined programs."""
    program = node.program
    rng = random.Random(utils.stable_int(node.node_id))
    candidates: list[tuple[grammar.AtomProgram, dict[str, Any]]] = []

    slots = missing_required_slots(program, atom_grammar)
    slots.extend(slot for slot in program.open_slots if slot not in slots)
    if node.status == "mastered":
        slots = ["condition", "composition", *slots]
    elif node.status == "failed":
        slots = ["constraint", "entity_or_environment", *slots]

    for requirement in grammar.missing_required_contexts(program, atom_grammar):
        atom = atom_grammar.atoms.get(requirement)
        slot = grammar.required_context_slot(requirement, atom_grammar)
        if atom is None or slot is None or not is_atom_compatible(program, atom, atom_grammar):
            continue
        created_from = {"operator": "add_required_context", "slot": slot, "atom": atom.name}
        candidates.append((add_atom_to_program(program, atom, grammar.role_for_slot(slot, atom)), created_from))
        if len(candidates) >= CHILDREN_PER_EXPANSION:
            return dedupe_program_candidates(candidates)

    for slot in slots:
        atoms = valid_atoms_for_slot(atom_grammar, program, slot)
        rng.shuffle(atoms)
        for atom in atoms:
            created_from = {"operator": "add_atom", "slot": slot, "atom": atom.name}
            candidates.append((add_atom_to_program(program, atom, grammar.role_for_slot(slot, atom)), created_from))
            if len(candidates) >= CHILDREN_PER_EXPANSION:
                return dedupe_program_candidates(candidates)

    if node.status in {"frontier", "mastered"}:
        candidates.extend(harden_program(program, atom_grammar))
    if node.status == "failed":
        candidates.extend(simplify_program(program))
    candidates.extend(refine_arguments(program, atom_grammar))
    return dedupe_program_candidates(candidates)


def dedupe_program_candidates(
    candidates: list[tuple[grammar.AtomProgram, dict[str, Any]]],
) -> list[tuple[grammar.AtomProgram, dict[str, Any]]]:
    """The candidates without repeated programs, each at its first position."""
    unique: dict[str, tuple[grammar.AtomProgram, dict[str, Any]]] = {}
    for program, created_from in candidates:
        unique.setdefault(grammar.atom_program_key(program), (program, created_from))
    return list(unique.values())


def role_filled(bindings: list[grammar.AtomBinding], slot: str) -> bool:
    """Whether a binding fills the slot: its role is one of the slot's categories."""
    return any(binding.role in grammar.SLOT_RULES[slot] for binding in bindings)


def missing_required_slots(program: grammar.AtomProgram, atom_grammar: grammar.AtomGrammar) -> list[str]:
    """The unfilled slots every complete program needs, then the slots of the context its atoms require."""
    slots = [slot for slot in OPEN_SLOTS[:4] if not role_filled(program.bindings, slot)]
    for requirement in grammar.missing_required_contexts(program, atom_grammar):
        slot = grammar.required_context_slot(requirement, atom_grammar)
        if slot is not None and slot not in slots:
            slots.append(slot)
    return slots


def valid_atoms_for_slot(
    atom_grammar: grammar.AtomGrammar, program: grammar.AtomProgram, slot: str
) -> list[grammar.AtomSpec]:
    categories = set(atom_grammar.slot_rules[slot])
    existing = set(grammar.atom_names(program))
    return [
        atom
        for atom in atom_grammar.atoms.values()
        if atom.name not in existing
        and atom.backend_supported
        and atom.category in categories
        and is_atom_compatible(program, atom, atom_grammar)
    ]


def is_atom_compatible(program: grammar.AtomProgram, atom: grammar.AtomSpec, atom_grammar: grammar.AtomGrammar) -> bool:
    names = set(grammar.atom_names(program))
    roles = {binding.role for binding in program.bindings}
    if atom.name in names or names & set(atom.incompatible_with):
        return False
    if any(atom.name in atom_grammar.atoms[binding.atom].incompatible_with for binding in program.bindings):
        return False
    return all(
        grammar.required_context_satisfied(requirement, names, roles, program.open_slots, atom_grammar)
        for requirement in atom.required_context
    )


def add_atom_to_program(program: grammar.AtomProgram, atom: grammar.AtomSpec, role: str) -> grammar.AtomProgram:
    bindings = [*program.bindings, grammar.AtomBinding(atom.name, role, grammar.bind_default_arguments(atom))]
    open_slots = [slot for slot in program.open_slots if slot != role or role == "observation"]
    if role in {"goal", "action"}:
        open_slots = [slot for slot in open_slots if slot != "goal"]
    if role in {"entity", "environment"}:
        open_slots = [slot for slot in open_slots if slot != "entity_or_environment"]
    constraints = list(program.constraints)
    if atom.category == "constraint" and atom.description not in constraints:
        constraints.append(atom.description)
    return grammar.AtomProgram(bindings, open_slots, constraints, program.depth + 1)


def refine_arguments(
    program: grammar.AtomProgram, atom_grammar: grammar.AtomGrammar
) -> list[tuple[grammar.AtomProgram, dict[str, Any]]]:
    """Each argument moved one difficulty level down or up (numeric) or set to another listed value."""
    candidates = []
    for idx, binding in enumerate(program.bindings):
        for name, value in binding.arguments.items():
            spec = atom_grammar.atoms[binding.atom].arguments.get(name, {})
            if spec.get("kind") in {"continuous", "integer"}:
                operator = "refine_numeric_argument"
                alternatives = [common.difficulty_step(spec, value, direction) for direction in ("low", "high")]
            else:
                operator, alternatives = "refine_categorical_argument", spec.get("values", [])
            for alternative in alternatives:
                if alternative is None or alternative == value:
                    continue
                bindings = [
                    grammar.AtomBinding(item.atom, item.role, dict(item.arguments)) for item in program.bindings
                ]
                bindings[idx].arguments[name] = alternative
                refined = grammar.AtomProgram(
                    bindings, list(program.open_slots), list(program.constraints), program.depth + 1
                )
                created_from = {"operator": operator, "atom": binding.atom, "argument": name, "value": alternative}
                candidates.append((refined, created_from))
    return candidates


def harden_program(
    program: grammar.AtomProgram, atom_grammar: grammar.AtomGrammar
) -> list[tuple[grammar.AtomProgram, dict[str, Any]]]:
    """The program with a condition or difficulty atom added (at most CHILDREN_PER_EXPANSION ways)."""
    atoms = [
        atom
        for atom in atom_grammar.atoms.values()
        if atom.category in {"difficulty", "condition"}
        and atom.backend_supported
        and is_atom_compatible(program, atom, atom_grammar)
    ]
    return [
        (
            add_atom_to_program(program, atom, grammar.role_for_slot("condition", atom)),
            {"operator": "harden_task", "atom": atom.name},
        )
        for atom in atoms[:CHILDREN_PER_EXPANSION]
    ]


def simplify_program(program: grammar.AtomProgram) -> list[tuple[grammar.AtomProgram, dict[str, Any]]]:
    """The program without one of its last three bindings (never the first)."""
    candidates = []
    for idx in range(len(program.bindings) - 1, 0, -1)[:3]:
        bindings = [binding for j, binding in enumerate(program.bindings) if j != idx]
        simplified = grammar.AtomProgram(
            bindings, list(program.open_slots), list(program.constraints), program.depth + 1
        )
        candidates.append((simplified, {"operator": "simplify_task", "removed_atom": program.bindings[idx].atom}))
    return candidates


def program_from_llm_proposal(
    proposal: dict[str, Any], parent_program: grammar.AtomProgram, atom_grammar: grammar.AtomGrammar
) -> tuple[grammar.AtomProgram, dict[str, Any]]:
    """An LLM-MCTS proposal as a child program: the parent's bindings plus the proposed atoms, each snapped to the
    nearest grammar atom and completed with default arguments and required context; and its created_from."""
    notes: list[str] = []
    bindings = [grammar.AtomBinding(item.atom, item.role, dict(item.arguments)) for item in parent_program.bindings]
    proposal_atoms = proposal.get("atoms")
    if not isinstance(proposal_atoms, list) or not proposal_atoms:
        proposal_atoms = [asdict(binding) for binding in parent_program.bindings]
        notes.append("proposal omitted atoms; preserved parent atoms only")
    for item in proposal_atoms:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("atom") or item.get("name") or "").strip()
        if not raw_name:
            continue
        atom_name = raw_name if raw_name in atom_grammar.atoms else nearest_atom_name(raw_name, atom_grammar)
        if atom_name is None:
            notes.append(f"unmapped atom {raw_name!r}")
            continue
        atom = atom_grammar.atoms[atom_name]
        if not atom.backend_supported:
            notes.append(f"ignored backend-unsupported atom {atom_name!r}")
            continue
        arguments = dict(item.get("arguments", {}) or {})
        for key, value in grammar.bind_default_arguments(atom).items():
            arguments.setdefault(key, value)
        bindings.append(
            grammar.AtomBinding(atom_name, normalize_role(str(item.get("role") or atom.category), atom), arguments)
        )

    bindings = merge_atom_bindings(satisfy_required_context_bindings(bindings, atom_grammar, notes))
    constraints = list(parent_program.constraints)
    for binding in bindings:
        description = atom_grammar.atoms[binding.atom].description
        if binding.role == "constraint" and description not in constraints:
            constraints.append(description)
    open_slots = [slot for slot in OPEN_SLOTS if not role_filled(bindings, slot)]
    program = grammar.AtomProgram(bindings, open_slots, constraints, max(parent_program.depth + 1, len(bindings)))
    created_from = {
        "operator": str(proposal.get("operator") or "vlm_expansion"),
        "proposal_source": "vlm_expansion",
        "proposal_description": str(proposal.get("description") or "").strip(),
        "proposal_primitive": str(proposal.get("primitive") or "").strip() or None,
        "proposal_success_predicate": str(proposal.get("success_predicate") or "").strip(),
        "proposal_rationale": str(proposal.get("rationale") or "").strip(),
        "proposal_parent_summary": grammar.program_summary(parent_program),
        "canonicalization_notes": "; ".join(notes) if notes else None,
        "completeness_score": grammar.program_completeness_score(program),
    }
    return program, created_from


def nearest_atom_name(name: str, atom_grammar: grammar.AtomGrammar) -> str | None:
    """The grammar atom whose name or description is most similar to name, if similar enough."""
    normalized = normalize_atom_name(name)
    best_name, best_score = None, 0.0
    for atom_name, atom in atom_grammar.atoms.items():
        for candidate in {normalize_atom_name(atom_name), normalize_atom_name(atom.description)}:
            score = difflib.SequenceMatcher(None, normalized, candidate).ratio()
            if normalized and (normalized in candidate or candidate in normalized):
                score = max(score, 0.78)
            if score > best_score:
                best_name, best_score = atom_name, score
    return best_name if best_score >= 0.55 else None


def normalize_atom_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", " ").replace("-", " ")).strip()


def merge_atom_bindings(bindings: list[grammar.AtomBinding]) -> list[grammar.AtomBinding]:
    """One binding per (atom, role), at its first position, with the set arguments of the later ones merged in."""
    merged: dict[tuple[str, str], grammar.AtomBinding] = {}
    for binding in bindings:
        key = (binding.atom, binding.role)
        if key not in merged:
            merged[key] = grammar.AtomBinding(binding.atom, binding.role, dict(binding.arguments))
        else:
            merged[key].arguments.update({k: v for k, v in binding.arguments.items() if v is not None})
    return list(merged.values())


def satisfy_required_context_bindings(
    bindings: list[grammar.AtomBinding], atom_grammar: grammar.AtomGrammar, notes: list[str]
) -> list[grammar.AtomBinding]:
    """The bindings plus a binding of each supported atom that one of them requires and none of them is."""
    names = {binding.atom for binding in bindings}
    categories = {atom_grammar.atoms[binding.atom].category for binding in bindings}
    additions = []
    for binding in bindings:
        for requirement in atom_grammar.atoms[binding.atom].required_context:
            required = atom_grammar.atoms.get(requirement)
            if requirement in names or requirement in categories or required is None or not required.backend_supported:
                continue
            arguments = grammar.bind_default_arguments(required)
            additions.append(grammar.AtomBinding(required.name, normalize_role(required.category, required), arguments))
            names.add(required.name)
            categories.add(required.category)
            notes.append(f"added required context atom {required.name!r}")
    return [*bindings, *additions]


def normalize_role(role: str, atom: grammar.AtomSpec) -> str:
    role = role.strip().lower()
    if role == "difficulty":
        return "condition"
    if role not in {
        "capability",
        "goal",
        "entity",
        "environment",
        "constraint",
        "condition",
        "observation",
        "composition",
    }:
        role = atom.category
    return "goal" if role == "action" else role


# ----------------------------------------------------------------------------------------------------------------------
# Running


def compile_task(
    selection: tuple[list[str], Node, str, float | None],
    iteration: int,
    gpu_id: str,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> MctsRecord | None:
    """Verbalize and compile a selected program into a task to train; None (and the node failed) if the compiler
    fails."""
    path, node, selected_by, score = selection
    program, atom_grammar = node.program, args.atom_grammar
    verbalized = grammar.verbalize_atom_program(program, atom_grammar, args, node.created_from)
    description = str(verbalized["description"]).strip()
    try:
        spec, module_path = generate_env.generate_environment(
            description,
            success_predicate=verbalized.get("success_predicate"),
            model=args.model,
            robot=args.backend_spec.generator_robot_name,
            primitive=verbalized["primitive"] or None,
            module_name=common.unique_module_name(description, iteration, history, prefix=args.method),
            output_dir=common.generated_env_dir(args),
            atom_program=grammar.serialize_atom_program(program, atom_grammar),
            goal_rotation_angle=grammar.goal_rotation_angle(program, atom_grammar),
            goal_rotation_axis=grammar.goal_rotation_axis(program, atom_grammar),
        )
        reward_budget = common.read_reward_budget(module_path)
    except Exception as exc:  # the LLM scene compiler is the one untrusted stage: drop the node, don't die
        print(f"{LOG_PREFIX} scene generation failed for {node.node_id}: {exc!r}; node marked failed", flush=True)
        node.status = "failed"
        return None
    action, entity, condition = grammar.main_action_object_signature(program)
    created_from = node.created_from
    completeness, duplicate = created_from.get("completeness_score"), created_from.get("duplicate_score")
    return MctsRecord(
        iteration=iteration,
        timestamp=time.time(),
        backend=args.backend,
        robot=args.robot,
        node_id=node.node_id,
        parent_id=node.parent_id,
        path=list(path),
        selected_by=selected_by,
        mcts_score=score,
        atom_program=grammar.atom_program_to_dict(program),
        atom_names=grammar.atom_names(program),
        description=description,
        primitive=spec.primitive,
        success_predicate=verbalized.get("success_predicate"),
        module_name=spec.module_name,
        task_id=spec.task_id,
        module_path=str(module_path),
        status="generated",
        reward_budget=reward_budget,
        gpu_id=gpu_id,
        proposal_description=created_from.get("proposal_description"),
        proposal_rationale=created_from.get("proposal_rationale"),
        proposal_operator=created_from.get("operator"),
        completeness_score=None if completeness is None else float(completeness),
        duplicate_score=None if duplicate is None else float(duplicate),
        main_action=action,
        main_entity=entity,
        main_condition=condition,
        atom_pair_keys=grammar.atom_pair_keys(program),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument(
        "--method", choices=METHODS, default="mcts", help="mcts, llm_mcts (LLM-MCTS) or muzero (MCTS-Learn)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.configure_gpus(args)
    common.setup_robot(args)
    log_dir = common.set_run_paths(args, args.method)
    tree_path = log_dir / f"{args.method}_tree.json"
    args.atom_grammar = grammar.build_atom_grammar(args)
    tree, atom_stats = load_tree(tree_path) if tree_path.exists() else (initialize_tree(args.atom_grammar), {})
    reset_in_flight_nodes(tree)
    reopen_selectable_ancestors(tree)
    history = common.load_history(args.history)
    muzero = MuZeroScorer(log_dir / "muzero.pt", args.atom_grammar) if args.method == "muzero" else None
    search = TreeSearch(args, tree, atom_stats, history, tree_path, muzero)
    search.save()

    gpu_ids = common.scheduler_gpu_ids(args.gpu_ids)
    iteration = common.next_iteration(history)
    trained = common.count_trained(history)
    while trained < args.iterations:
        selections = search.select_batch(min(args.parallel_tasks, args.iterations - trained))
        if not selections:  # exit, not break, so slurm.sh does not requeue an exhausted cell
            raise SystemExit(f"{LOG_PREFIX} no untrained evaluable program remains; the search space is exhausted")
        # verbalizing and compiling are I/O-bound LLM calls: run them concurrently, update the tree after
        with ThreadPoolExecutor(max_workers=SCENE_COMPILE_WORKERS) as pool:
            futures = [
                pool.submit(compile_task, selection, iteration + idx, common.batch_gpu_id(gpu_ids, idx), history, args)
                for idx, selection in enumerate(selections)
            ]
            records = [record for future in futures if (record := future.result()) is not None]
        for record in records:
            node = tree[record.node_id]
            node.last_result = utils.json_safe(asdict(record))
            node.status = "training"
        search.save()
        iteration += len(records)

        jobs = [common.launch_training(record, args, args.method) for record in records]
        common.wait_for_training(jobs, search.record_result)
        trained += sum(job.record.status == "trained" and job.record.success_rate is not None for job in jobs)
    return 0


if __name__ == "__main__":
    main()
