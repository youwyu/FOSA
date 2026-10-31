"""FOSA (Sec. 4): a GFlowNet over task skeletons, rewarded by how well a task policy learns each task.

Each round, FOSA proposes 32 skeletons and screens them. It compiles the first new ones (one per PPO slot) into
environments, trains a task policy on each, and scores the learning curves: V (Eq. 7) and R = eps + exp(beta V)
(Eq. 9). Then it fits the GFlowNet to the replayed rewards by trajectory balance (Eq. 5, gflownet.py).

How FOSA proposes a skeleton:
- It draws a goal arm by Thompson sampling. An arm is a goal atom plus values for its categorical arguments; its
  success is how often its trained tasks land on the capability frontier (F > 0, Eq. 8).
- Half of the proposals complete the arm with uniform valid edits.
- The other half improve one of the arm's frontier tasks by local search (Kim et al., 2024): undo 1-3 of its edits with
  P_B and redo them with P_F. If the arm has no frontier task, or local search finds no new task, P_F builds the arm
  from scratch.
- P_F's sampling logits give a bonus to the edits the LLM prefers, and a small share of edits is uniform.

Ablations (--method):
- fosa_llm (FOSA-LLM) samples among the edits the LLM picks, and learns nothing.
- fosa_rnd (FOSA-RND) trains P_F by policy gradient on V plus the RND novelty of the skeleton (Burda et al., 2019).
- `--rnd-bonus` adds that RND novelty to FOSA's own V. The paper does not use it.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fosa import utils
from fosa.compiler import generate_env
from fosa.discovery import common, grammar, objectives
from fosa.discovery.gflownet import GFlowNet, SkeletonEncoder, Trajectory
from fosa.discovery.skeleton_space import SkeletonSpace, State

METHODS = ("fosa", "fosa_rnd", "fosa_llm")
PROPOSALS_PER_ROUND = 32
EXPLORE_SHARE = 0.5  # share of FOSA's proposals that complete the drawn arm with uniform edits
LOCAL_SEARCH_DEPTHS = (1, 2, 3)  # edits local search undoes, fewest first, until the result is a new task
REWARD_TEMPERATURE = 8.0  # beta in R = eps + exp(beta V)
REWARD_EPS = 1e-4  # eps
REJECTED_GFN_REWARD = 0.1  # R of a proposal that is never trained
REJECTED_STATUSES = {"invalid", "incomplete", "duplicate"}
REPLAY_TRAINED, REPLAY_REJECTED = 1024, 3072
MIN_REPLAY = 8  # replay rows needed before the first fit
TRAIN_STEPS = 300  # replay batches per round
BATCH_SIZE = 32
LLM_PRIOR_BONUS = 1.0  # added to P_F's logit of each edit the LLM prefers
LLM_PRIOR_MAX_CALLS = 16  # uncached LLM questions per round (FOSA)
LLM_PRIOR_INSTRUCTION = (
    'Choose at most four valid robot grammar edits. Return only JSON: {"preferred_indices":[integer,...]}'
)
RND_STEPS = 50  # RND predictor steps per round
RND_BONUS_WEIGHT = 0.4  # --rnd-bonus: weight of the RND novelty in V, on the scale of the count novelty
SCENE_COMPILE_WORKERS = 4
MAX_EMPTY_ROUNDS = 100


def history_name(args: argparse.Namespace) -> str:
    return "fosa_rnd_bonus" if args.method == "fosa" and args.rnd_bonus else args.method


# ----------------------------------------------------------------------------------------------------------------------
# Records


@dataclass
class Proposal:
    """A sampled construction of a skeleton, with what the sampler saw on the way."""

    actions: list[dict[str, Any]]  # the edits, STOP last
    terminal_program: dict[str, Any]
    goal_family: str  # the family it was sampled in, whose log Z it trains
    llm_prior_calls: int
    num_valid_actions_per_step: list[int]
    sampled_action_logprobs: list[float]


@dataclass(kw_only=True)
class Record(common.TaskRecord):
    """One history row: a proposal and, if it was trained, its task policy's results."""

    sample_idx: int
    method: str
    atom_program: dict[str, Any]
    atom_names: list[str]
    actions: list[dict[str, Any]]
    success_predicate: str | None
    policy_score: float | None = None
    task_value: float | None = None
    gfn_reward: float = REJECTED_GFN_REWARD
    frontier_bonus: float | None = None
    novelty_bonus: float | None = None
    cost_penalty: float | None = None
    llm_action_prior_used: bool = False
    llm_action_prior_calls: int = 0
    num_valid_actions_per_step: list[int] | None = None
    sampled_action_logprobs: list[float] | None = None
    completeness_score: float | None = None
    duplicate_score: float | None = None
    duplicate_reason: str | None = None
    main_action: str | None = None
    main_entity: str | None = None
    main_condition: str | None = None
    atom_pair_keys: list[str] | None = None


def replay_item(proposal: Proposal, record: Record) -> dict[str, Any]:
    """A proposal labeled with its outcome, as the replay stores it."""
    return {
        "actions": proposal.actions,
        "goal_family": proposal.goal_family,
        "terminal_program": proposal.terminal_program,
        "gfn_reward": record.gfn_reward,
        "record": utils.json_safe(asdict(record)),
    }


class ReplayBuffer:
    """Trained proposals and rejected ones, kept apart so every batch holds both."""

    def __init__(self) -> None:
        self.positives: deque[dict[str, Any]] = deque(maxlen=REPLAY_TRAINED)
        self.negatives: deque[dict[str, Any]] = deque(maxlen=REPLAY_REJECTED)

    def add(self, item: dict[str, Any]) -> None:
        (self.positives if item["record"]["status"] == "trained" else self.negatives).append(item)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        """Half trained, half rejected rows, topped up from the other side when one runs short."""
        size = min(batch_size, len(self))
        trained = min(len(self.positives), (size + 1) // 2)
        rejected = min(len(self.negatives), size - trained)
        trained = min(len(self.positives), size - rejected)
        items = random.sample(list(self.positives), trained) + random.sample(list(self.negatives), rejected)
        random.shuffle(items)
        return items

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")  # write, then rename: a killed job never leaves half a file
        with tmp.open("w", encoding="utf-8") as f:
            for item in (*self.positives, *self.negatives):
                f.write(json.dumps(item, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def load_jsonl(self, path: Path) -> None:
        for item in common.load_history(path):
            if item["record"]["status"] == "trained" and item["record"]["success_rate"] is None:
                continue
            self.add(item)

    def __len__(self) -> int:
        return len(self.positives) + len(self.negatives)


# ----------------------------------------------------------------------------------------------------------------------
# Reward


def gfn_reward(value: float) -> float:
    """R = eps + exp(beta V) (Eq. 9), with the exponent clamped to [-20, 20]."""
    return float(REWARD_EPS + math.exp(max(-20.0, min(20.0, REWARD_TEMPERATURE * value))))


def task_value(
    program: Any, success_curve: list[float], status: str, others: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[float, float]:
    """V = F + lambda_nov N - lambda_cost C (Eq. 7), with the novelty N measured against `others`. Returns (V, N)."""
    novelty = grammar.novelty_bonus_from_program(program, atom_stats(others), others, [], args.atom_grammar)
    cost = objectives.program_cost(len(program.bindings), status)
    return objectives.task_utility(success_curve, novelty, cost), novelty


def atom_stats(history: list[dict[str, Any]]) -> dict[str, grammar.AtomStats]:
    """Visit counts and value statistics of each atom over the trained tasks."""
    stats: dict[str, grammar.AtomStats] = {}
    for row in history:
        if row["status"] != "trained" or row["policy_score"] is None:
            continue
        value, success = row["task_value"], row["policy_score"]
        for name in row["atom_names"]:
            stat = stats.setdefault(name, grammar.AtomStats())
            stat.visits += 1
            stat.value_sum += value
            stat.value_mean = stat.value_sum / stat.visits
            stat.success_sum += success
            stat.success_mean = stat.success_sum / stat.visits
    return stats


def refresh_replay_rewards(
    replay: ReplayBuffer,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    rnd: RND | None,
    encoder: SkeletonEncoder,
) -> None:
    """Re-score every trained replay row against the archive as it is now: novelty falls as similar tasks are added.
    With `rnd`, its normalized novelty is added to V (--rnd-bonus)."""
    for item in replay.positives:
        row = item["record"]
        others = [other for other in history if other["module_name"] != row["module_name"]]
        program = grammar.atom_program_from_dict(row["atom_program"])
        value, novelty = task_value(program, row["success_curve"], row["status"], others, args)
        item["gfn_reward"], row["task_value"], row["novelty_bonus"] = gfn_reward(value), value, novelty
    if rnd is not None and replay.positives:
        programs = [grammar.atom_program_from_dict(item["record"]["atom_program"]) for item in replay.positives]
        novelty = rnd.novelty(encoder(programs).to(rnd.device))
        bonuses = (RND_BONUS_WEIGHT * novelty / novelty.max().clamp_min(1e-12)).tolist()
        for item, bonus in zip(replay.positives, bonuses, strict=True):
            item["gfn_reward"] = gfn_reward(item["record"]["task_value"] + bonus)
            item["record"]["rnd_bonus"] = bonus


class RND:
    """Random network distillation (Burda et al., 2019) on skeleton embeddings: a skeleton's novelty is the error of a
    predictor, trained on every skeleton proposed so far, against a fixed random network."""

    def __init__(self, input_dim: int, device: torch.device, dim: int = 64) -> None:
        self.device = device
        self.target = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, dim)).to(device)
        self.target.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, dim)).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.memory = torch.empty(0, input_dim, device=device)

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (self.predictor(x) - self.target(x)).pow(2).mean(dim=1)

    def fit(self, x: torch.Tensor) -> None:
        self.memory = torch.cat([self.memory, x])
        for _ in range(RND_STEPS):
            batch = self.memory[torch.randint(len(self.memory), (min(64, len(self.memory)),), device=self.device)]
            loss = (self.predictor(batch) - self.target(batch)).pow(2).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "memory": self.memory.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.memory = state["memory"].to(self.device)


# ----------------------------------------------------------------------------------------------------------------------
# Proposals


class LLMEditPrior:
    """The valid edits the LLM prefers at a skeleton, cached. FOSA asks at most LLM_PRIOR_MAX_CALLS uncached questions
    per round; FOSA-LLM is not capped. The LLM only picks among valid edits; it never adds any."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.capped = args.method != "fosa_llm"
        self.cache: dict[str, list[int]] = {}
        self.calls = 0  # uncached questions this round

    def __call__(self, program: Any, actions: list[dict[str, Any]]) -> list[int]:
        """Positions in `actions` that the LLM prefers."""
        key = json.dumps({"program": grammar.atom_program_key(program), "valid_actions": actions}, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if self.capped and self.calls >= LLM_PRIOR_MAX_CALLS:
            return []
        self.calls += 1
        question = {
            "atom_program": grammar.atom_program_to_dict(program),
            "valid_actions": [{"index": i, "action": a} for i, a in enumerate(actions)],
        }
        messages = [
            {"role": "system", "content": LLM_PRIOR_INSTRUCTION},
            {"role": "user", "content": json.dumps(question, separators=(",", ":"))},
        ]
        answer = grammar.request_json(messages, self.args, max_tokens=256, timeout_s=60.0)
        picked = [i for i in answer.get("preferred_indices", []) if isinstance(i, int) and 0 <= i < len(actions)]
        self.cache[key] = list(dict.fromkeys(picked))  # in order, without repeats
        return self.cache[key]


Arm = tuple[str, tuple[tuple[str, str], ...]]  # a goal atom and its categorical argument values, as sorted (name, JSON)


def arm_values(arm: Arm) -> dict[str, Any]:
    return {name: json.loads(value) for name, value in arm[1]}


class Sampler:
    """Proposes skeletons: FOSA's arm-conditioned local search and uniform completion, or an ablation's sampling."""

    def __init__(self, gfn: GFlowNet, prior: LLMEditPrior, args: argparse.Namespace) -> None:
        self.space = gfn.space
        self.gfn = gfn
        self.prior = prior
        self.args = args
        self.device = gfn.device
        self.empty_rounds = 0  # rounds in a row that trained nothing; raises the share of uniform edits

    def propose(self, history: list[dict[str, Any]], replay: ReplayBuffer, seen: set[str]) -> Proposal:
        if self.args.method != "fosa":  # the ablations: a uniform goal family, built from scratch
            family = self.space.goal_families[int(torch.randint(len(self.space.goal_families), (), device=self.device))]
            return self.complete((), [], family, None)
        explore = bool(torch.rand((), device=self.device) < EXPLORE_SHARE)
        arm = self.draw_arm(history)
        if explore:
            return self.complete((), [], arm[0], arm_values(arm), uniform=True)
        seeds = [
            item
            for item in replay.positives
            if self.arm_of(item["record"]["atom_program"]) == arm
            and objectives.frontier_score(item["record"]["success_curve"]) > 0
        ]
        if seeds:
            seed = random.choices(seeds, weights=[item["gfn_reward"] for item in seeds])[0]
            for depth in LOCAL_SEARCH_DEPTHS:
                start = self.back_off(seed["actions"], depth)
                if start is not None:
                    proposal = self.complete(*start, seed["goal_family"], None)
                    if grammar.atom_program_key(grammar.atom_program_from_dict(proposal.terminal_program)) not in seen:
                        return proposal
        return self.complete((), [], arm[0], arm_values(arm))

    # Thompson sampling over goal arms

    def _categorical(self, atom: Any) -> list[str]:
        return [
            name
            for name, spec in atom.arguments.items()
            if spec.get("values") and spec.get("kind") not in {"continuous", "integer"}
        ]

    def arms(self) -> list[Arm]:
        """One arm per goal atom and combination of its categorical argument values, in grammar order."""
        arms = []
        for family in self.space.goal_families:
            atom = self.space.grammar.atoms[family]
            names = self._categorical(atom)
            for values in itertools.product(*[atom.arguments[name]["values"] for name in names]):
                arms.append((family, tuple(sorted(zip(names, (json.dumps(v) for v in values))))))
        return arms

    def arm_of(self, program: dict[str, Any]) -> Arm | None:
        for binding in program["bindings"]:
            atom = self.space.grammar.atoms[binding["atom"]]
            if atom.category == "goal":
                names = self._categorical(atom)
                return atom.name, tuple(sorted((name, json.dumps(binding["arguments"].get(name))) for name in names))
        return None

    def draw_arm(self, history: list[dict[str, Any]]) -> Arm:
        """Thompson sampling on each arm's frontier rate: Beta(1 + frontier tasks, 1 + other trained tasks)."""
        arms = self.arms()
        index = {arm: i for i, arm in enumerate(arms)}
        hits, misses = torch.ones(len(arms)), torch.ones(len(arms))
        for row in history:
            i = index.get(self.arm_of(row["atom_program"])) if row["status"] == "trained" else None
            if i is not None:
                on_frontier = objectives.frontier_score(row.get("success_curve")) > 0
                hits[i] += on_frontier
                misses[i] += not on_frontier
        return arms[int(torch.distributions.Beta(hits, misses).sample().argmax())]

    # Construction

    @torch.no_grad()
    def back_off(self, actions: list[dict[str, Any]], depth: int) -> tuple[State, list[dict[str, Any]]] | None:
        """Local search's first half: from a task's skeleton, undo `depth` edits sampled from P_B. Only parents the
        grammar can rebuild from the empty skeleton are eligible, never the empty skeleton itself. Returns the state
        reached and the edits that rebuild it, or None if there are none."""
        state = self.space.replay(actions)[0][-1]
        for _ in range(depth):
            parents = [
                (parent, edit)
                for parent, edit in self.space.parents(state)
                if parent and self.space.build_order(parent) is not None
            ]
            if not parents:
                break
            logits = self.gfn.backward_logits(state)[[edit for _parent, edit in parents]]
            state = parents[int(torch.distributions.Categorical(logits=logits).sample())][0]
        order = self.space.build_order(state)
        return None if order is None else (state, [self.space.action(edit) for edit in order])

    def epsilon(self) -> float:
        """The share of uniform edits, raised while rounds keep training nothing."""
        return min(0.9, 0.05 + 0.02 * self.empty_rounds)

    @torch.no_grad()
    def complete(
        self,
        state: State,
        actions: list[dict[str, Any]],
        family: str,
        arm: dict[str, Any] | None,
        uniform: bool = False,
    ) -> Proposal:
        """Finish a skeleton inside the goal family (and arm), from `state`, which `actions` reach. Each edit is drawn
        from the sampling logits, or uniformly with probability epsilon (always if `uniform`). Once the family's goal is
        in, the refinements that give it the arm's values come first."""
        num_valid, log_probs = [], []
        calls = self.prior.calls
        while valid := self.space.valid(state, family, arm):
            program = self.space.program(state)
            num_valid.append(len(valid))
            dist = torch.distributions.Categorical(logits=self.logits(state, program, valid))
            choice = self.arm_edit(program, valid, family, arm)
            if choice is None:
                if uniform or torch.rand((), device=self.device) < self.epsilon():
                    choice = int(torch.randint(len(valid), (), device=self.device))
                else:
                    choice = int(dist.sample())
            log_probs.append(float(dist.log_prob(torch.tensor(choice, device=self.device))))
            actions.append(self.space.action(valid[choice]))
            if valid[choice] == self.space.stop:
                break
            state = self.space.step(state, valid[choice])
        return Proposal(
            actions=actions,
            terminal_program=grammar.atom_program_to_dict(self.space.program(state)),
            goal_family=family,
            llm_prior_calls=self.prior.calls - calls,
            num_valid_actions_per_step=num_valid,
            sampled_action_logprobs=log_probs,
        )

    def logits(self, state: State, program: Any, valid: list[int]) -> torch.Tensor:
        """Sampling logits over the valid edits. FOSA: P_F's plus the LLM edit prior. FOSA-RND: P_F's. FOSA-LLM:
        uniform over the LLM's picks, or over every valid edit if it picks none."""
        actions = [self.space.action(edit) for edit in valid]
        if self.args.method == "fosa_llm":
            picked = self.prior(program, actions)
            logits = torch.full((len(valid),), -1e9 if picked else 0.0, dtype=torch.float32, device=self.device)
            logits[picked] = 0.0
            return logits
        logits = self.gfn.forward_logits(state)[valid]
        if self.args.method == "fosa":
            for i in self.prior(program, actions):
                logits[i] += LLM_PRIOR_BONUS
        return logits

    def arm_edit(self, program: Any, valid: list[int], family: str, arm: dict[str, Any] | None) -> int | None:
        """The position in `valid` of a refinement that sets one of the goal's categorical arguments to the arm's
        value."""
        goal = next((binding for binding in program.bindings if binding.atom == family), None)
        if goal is None or not arm:
            return None
        for i, edit_id in enumerate(valid):
            if edit_id == self.space.stop:
                continue
            edit = self.space.edits[edit_id]
            name = edit.arguments.get("argument")
            if (
                edit.action_type == "refine_arg"
                and edit.atom == family
                and name in arm
                and edit.arguments.get("value") == arm[name]
                and goal.arguments.get(name) != arm[name]
            ):
                return i
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Training


class Trainer:
    """Fits the GFlowNet to replayed proposals: trajectory balance for FOSA, policy gradient for FOSA-RND."""

    def __init__(self, gfn: GFlowNet, encoder: SkeletonEncoder) -> None:
        self.gfn = gfn
        self.space = gfn.space
        self.encoder = encoder
        self.trajectories: dict[str, Trajectory | None] = {}  # by recorded actions

    def trajectory(self, item: dict[str, Any]) -> Trajectory | None:
        """The item's construction as the policies see it; None if it never reached STOP."""
        key = json.dumps(item["actions"], sort_keys=True)
        if key not in self.trajectories:
            replayed = self.space.replay(item["actions"])
            if replayed is None:
                self.trajectories[key] = None
            else:
                states, edits, valid = replayed
                backward_valid = [[edit for _parent, edit in self.space.parents(state)] for state in states[1:]]
                self.trajectories[key] = Trajectory(states, edits, valid, backward_valid)
        return self.trajectories[key]

    def train(self, replay: ReplayBuffer, steps: int) -> float:
        """Trajectory balance. Each replay batch is split by goal family; each part takes one step against its log Z."""
        losses = []
        for _ in range(steps):
            by_family: dict[int, list[dict[str, Any]]] = {}
            for item in replay.sample(BATCH_SIZE):
                by_family.setdefault(self.gfn.families.index(item["goal_family"]), []).append(item)
            for family, items in by_family.items():
                kept = [(trajectory, item) for item in items if (trajectory := self.trajectory(item)) is not None]
                if kept:
                    log_rewards = [math.log(max(item["gfn_reward"], REJECTED_GFN_REWARD)) for _t, item in kept]
                    loss = self.gfn.trajectory_balance([t for t, _item in kept], log_rewards, family)
                    losses.append(self.gfn.step(loss))
        return sum(losses) / len(losses) if losses else float("nan")

    def train_rnd(self, replay: ReplayBuffer, rnd: RND, steps: int) -> float:
        """FOSA-RND: policy gradient on V plus RND novelty, each standardized over the batch."""
        losses = []
        for _ in range(steps):
            items = replay.sample(BATCH_SIZE)
            programs = [grammar.atom_program_from_dict(item["terminal_program"]) for item in items]
            novelty = rnd.novelty(self.encoder(programs).to(self.gfn.device))
            value = torch.tensor([item["record"]["task_value"] for item in items], device=self.gfn.device)
            returns = (standardize(value) + standardize(novelty)).tolist()
            kept = [
                (trajectory, r) for item, r in zip(items, returns) if (trajectory := self.trajectory(item)) is not None
            ]
            if kept:
                ret = torch.tensor([r for _t, r in kept], dtype=torch.float32, device=self.gfn.device)
                loss = -((ret - ret.mean()) * self.gfn.log_pf([t for t, _r in kept])).mean()
                losses.append(self.gfn.step(loss))
        return sum(losses) / len(losses) if losses else float("nan")


def standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)


# ----------------------------------------------------------------------------------------------------------------------
# From a proposal to a trained, scored task


def make_record(
    proposal: Proposal,
    program: Any,
    args: argparse.Namespace,
    iteration: int,
    sample_idx: int,
    *,
    status: str,
    description: str = "",
    primitive: str | None = None,
    success_predicate: str | None = None,
    module_name: str | None = None,
    error: str | None = None,
) -> Record:
    action, entity, condition = grammar.main_action_object_signature(program)
    return Record(
        iteration=iteration,
        sample_idx=sample_idx,
        timestamp=time.time(),
        backend=args.backend,
        robot=args.robot,
        method=args.method,
        atom_program=grammar.atom_program_to_dict(program),
        atom_names=grammar.atom_names(program),
        actions=list(proposal.actions),
        description=description,
        primitive=primitive or None,
        success_predicate=success_predicate,
        module_name=module_name,
        module_path=None,
        status=status,
        error=error,
        llm_action_prior_used=proposal.llm_prior_calls > 0,
        llm_action_prior_calls=proposal.llm_prior_calls,
        num_valid_actions_per_step=list(proposal.num_valid_actions_per_step),
        sampled_action_logprobs=list(proposal.sampled_action_logprobs),
        completeness_score=grammar.program_completeness_score(program),
        main_action=action,
        main_entity=entity,
        main_condition=condition,
        atom_pair_keys=grammar.atom_pair_keys(program),
    )


def is_duplicate(program: Any, description: str | None, history: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """A task is a duplicate if its skeleton or its description was already accepted."""
    key = grammar.atom_program_key(program)
    for row in history:
        if row["status"] in REJECTED_STATUSES:
            continue
        if grammar.atom_program_key(grammar.atom_program_from_dict(row["atom_program"])) == key:
            return True, "exact atom program already exists"
        if row["description"].strip() == description:
            return True, "exact description already exists"
    return False, None


def unique_module_name(
    description: str, iteration: int, sample_idx: int, history: list[dict[str, Any]], args: argparse.Namespace
) -> str:
    base = utils.safe_module_name(description)
    name = utils.safe_module_name(f"{history_name(args)}_{iteration:04d}_{sample_idx:02d}_{base}")
    return common.unused_name(name, {row["module_name"] for row in history}, 80)


def screen(
    proposal: Proposal, program: Any, args: argparse.Namespace, history: list[dict[str, Any]], iteration: int, idx: int
) -> Record:
    """Grammar checks and a template description. A proposal that fails them, or repeats an accepted task, is
    rejected."""

    def rejected(status: str, reason: str, value: float) -> Record:
        record = make_record(proposal, program, args, iteration, idx, status=status, error=reason)
        record.task_value, record.gfn_reward = value, REJECTED_GFN_REWARD
        record.duplicate_score, record.duplicate_reason = 0.0, reason
        return record

    valid, reason = grammar.is_program_valid(program, args.atom_grammar)
    if not valid:
        return rejected("invalid", reason, -0.25)
    complete, reason = grammar.is_program_complete_for_evaluation(program, args.atom_grammar)
    if not complete:
        completeness = grammar.program_completeness_score(program)
        return rejected("incomplete", reason, -0.10 + 0.10 * completeness)
    text = grammar.template_task(program, args.atom_grammar, args)
    duplicate, reason = is_duplicate(program, text["description"], history)
    record = make_record(
        proposal,
        program,
        args,
        iteration,
        idx,
        status="duplicate" if duplicate else "candidate",
        description=text["description"],
        primitive=text["primitive"],
        success_predicate=text["success_predicate"],
        module_name=unique_module_name(text["description"], iteration, idx, history, args),
    )
    record.duplicate_score, record.duplicate_reason = float(duplicate), reason
    # a candidate's value and reward are replaced once its task policy is trained
    record.task_value, record.gfn_reward = (-1.0, REJECTED_GFN_REWARD) if duplicate else (0.0, gfn_reward(0.0))
    return record


def canonical_program(program: Any) -> Any:
    """The skeleton with its construction-order fields cleared, so a re-derived skeleton hits the compile cache."""
    bindings = sorted(program.bindings, key=lambda b: (b.role, b.atom, json.dumps(b.arguments, sort_keys=True)))
    return dataclasses.replace(
        program, bindings=type(program.bindings)(bindings), depth=0, open_slots=type(program.open_slots)()
    )


def compile_task(record: Record, args: argparse.Namespace) -> Record:
    """The LLM verbalizes the skeleton (as for the baselines) and compiles it into an environment (Eq. 2)."""
    program = grammar.atom_program_from_dict(record.atom_program)
    verbalized = grammar.verbalize_atom_program(program, args.atom_grammar, args)
    description = str(verbalized.get("description") or record.description).strip()
    duplicate, reason = is_duplicate(program, description, common.load_history(args.history))
    if duplicate and "description" in reason:
        raise RuntimeError(f"duplicate task: {reason}")
    record.description = description
    record.success_predicate = str(verbalized.get("success_predicate") or record.success_predicate or "") or None
    spec, module_path = generate_env.generate_environment(
        record.description,
        success_predicate=record.success_predicate,
        model=args.model,
        robot=args.robot,
        primitive=record.primitive,
        module_name=record.module_name,
        output_dir=common.generated_env_dir(args),
        atom_program=grammar.serialize_atom_program(canonical_program(program), args.atom_grammar),
        goal_rotation_angle=grammar.goal_rotation_angle(program, args.atom_grammar),
        goal_rotation_axis=grammar.goal_rotation_axis(program, args.atom_grammar),
    )
    record.primitive = spec.primitive
    record.description = str(spec.task_description or record.description)  # the text the student trains on
    record.module_path = str(module_path)
    record.reward_budget = common.read_reward_budget(module_path)
    record.status = "generated"
    return record


def score(job: common.TrainingJob, history: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Read the task policy's results and score them: F (Eq. 8), V (Eq. 7) and R (Eq. 9)."""
    record = job.record
    common.finish_training(job, reward_budget=record.reward_budget)
    program = grammar.atom_program_from_dict(record.atom_program)
    record.policy_score = record.success_rate
    record.task_value, record.novelty_bonus = task_value(program, record.success_curve, record.status, history, args)
    record.frontier_bonus = objectives.frontier_score(record.success_curve)
    record.cost_penalty = objectives.program_cost(len(program.bindings), record.status)
    record.gfn_reward = gfn_reward(record.task_value)


# ----------------------------------------------------------------------------------------------------------------------
# The discovery loop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--method", choices=METHODS, default="fosa", help="FOSA, or one of its ablations.")
    parser.add_argument(
        "--rnd-bonus",
        action="store_true",
        help="FOSA only: add each trained skeleton's RND novelty to V (not in the paper).",
    )
    return parser.parse_args(argv)


def setup(args: argparse.Namespace) -> Any:
    common.setup_robot(args)
    name = history_name(args)
    log_dir = common.set_run_paths(args, name)
    args.replay_path = log_dir / f"{name}_replay.jsonl"
    args.model_path = log_dir / f"{name}.pt"
    args.atom_grammar = grammar.build_atom_grammar(args)
    return args.atom_grammar


def torch_device(args: argparse.Namespace) -> tuple[torch.device, list[str]]:
    gpu_ids = common.scheduler_gpu_ids(args.gpu_ids)
    if not torch.cuda.is_available() or args.gpu_ids.lower() in {"", "cpu"}:
        return torch.device("cpu"), gpu_ids
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    requested = gpu_ids[0]
    if requested in visible:
        torch.cuda.set_device(visible.index(requested))
    elif not visible and requested.isdigit() and int(requested) < torch.cuda.device_count():
        torch.cuda.set_device(int(requested))
    else:
        torch.cuda.set_device(0)
    return torch.device("cuda"), gpu_ids


def propose_round(
    sampler: Sampler, replay: ReplayBuffer, history: list[dict[str, Any]], args: argparse.Namespace, iteration: int
) -> list[tuple[Proposal, Any, Record]]:
    """A round of screened proposals. The rejected ones go to the replay, so the sampler learns not to repeat them."""
    sampler.prior.calls = 0
    seen = {
        grammar.atom_program_key(grammar.atom_program_from_dict(row["atom_program"]))
        for row in history
        if row["status"] not in REJECTED_STATUSES
    }
    proposals = []
    for idx in range(PROPOSALS_PER_ROUND):
        proposal = sampler.propose(history, replay, seen)
        program = grammar.atom_program_from_dict(proposal.terminal_program)
        seen.add(grammar.atom_program_key(program))
        proposals.append((proposal, program, screen(proposal, program, args, history, iteration, idx)))
    for proposal, _program, record in proposals:
        if record.status in REJECTED_STATUSES:
            replay.add(replay_item(proposal, record))
    return proposals


def select_batch(
    candidates: list[tuple[Proposal, Any, Record]], history: list[dict[str, Any]], limit: int
) -> list[tuple[Proposal, Any, Record]]:
    """The first candidates, in the order drawn, that duplicate neither the archive nor each other."""
    selected, compared = [], list(history)
    for proposal, program, record in candidates:
        if not is_duplicate(program, record.description, compared)[0]:
            selected.append((proposal, program, record))
            compared.append(utils.json_safe(asdict(record)))
            if len(selected) >= limit:
                break
    return selected


def compile_tasks(
    batch: list[tuple[Proposal, Any, Record]],
    replay: ReplayBuffer,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[Proposal, Record]]:
    """Compile the batch concurrently (LLM calls). A task whose compilation fails is rejected."""
    with ThreadPoolExecutor(max_workers=SCENE_COMPILE_WORKERS) as pool:
        futures = [pool.submit(compile_task, record, args) for _proposal, _program, record in batch]
    compiled = []
    for (proposal, _program, record), future in zip(batch, futures, strict=True):
        try:
            compiled.append((proposal, future.result()))
        except Exception as exc:  # the LLM compiler is the one untrusted stage: reject the task, don't stop the run
            record.status = "duplicate" if "duplicate task" in str(exc) else "invalid"
            record.error = f"scene generation failed: {exc!r}"
            record.task_value, record.gfn_reward = -0.25, REJECTED_GFN_REWARD
            replay.add(replay_item(proposal, record))
            common.append_history(record, history, args.history)
    return compiled


def train_tasks(
    compiled: list[tuple[Proposal, Record]],
    replay: ReplayBuffer,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    """Train a task policy on each compiled task with PPO, in parallel, and log the scored outcomes. Returns the number
    trained."""
    jobs = [(proposal, common.launch_training(record, args, history_name(args))) for proposal, record in compiled]
    common.wait_for_training([job for _proposal, job in jobs], lambda job: score(job, history, args))
    trained = 0
    for proposal, job in jobs:
        record = job.record
        if record.status == "trained" and record.policy_score is not None:
            trained += 1
        else:
            record.task_value, record.gfn_reward = -0.25, REJECTED_GFN_REWARD
        replay.add(replay_item(proposal, record))
        common.append_history(record, history, args.history)
    return trained


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.configure_gpus(args)
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(common.SEED)
    atom_grammar = setup(args)
    history = common.load_history(args.history)
    replay = ReplayBuffer()
    replay.load_jsonl(args.replay_path)
    device, gpu_ids = torch_device(args)
    encoder = SkeletonEncoder(atom_grammar, device)
    gfn = GFlowNet(SkeletonSpace(atom_grammar), encoder, device)
    fosa = args.method == "fosa"
    rnd = RND(encoder.dim, device) if args.method == "fosa_rnd" or (fosa and args.rnd_bonus) else None
    if args.model_path.exists():  # resume
        checkpoint = torch.load(args.model_path, map_location=device)
        gfn.load_state_dict(checkpoint["gflownet"])
        if rnd is not None:
            rnd.load_state_dict(checkpoint["rnd"])
    sampler = Sampler(gfn, LLMEditPrior(args), args)
    trainer = Trainer(gfn, encoder)

    iteration, trained = common.next_iteration(history), common.count_trained(history)
    while trained < args.iterations:
        proposals = propose_round(sampler, replay, history, args, iteration)
        candidates = [p for p in proposals if p[2].status not in REJECTED_STATUSES]
        batch = select_batch(candidates, history, min(args.parallel_tasks, args.iterations - trained))
        for slot, (_proposal, _program, record) in enumerate(batch):
            record.iteration, record.gpu_id = iteration + slot, common.batch_gpu_id(gpu_ids, slot)
            record.module_name = unique_module_name(
                record.description, record.iteration, record.sample_idx, history, args
            )
        iteration += len(batch)
        compiled = compile_tasks(batch, replay, history, args)
        if compiled:
            sampler.empty_rounds = 0
            trained += train_tasks(compiled, replay, history, args)
        else:
            sampler.empty_rounds += 1
            if not candidates and sampler.empty_rounds >= MAX_EMPTY_ROUNDS:
                raise SystemExit(
                    f"[fosa] no new task in {sampler.empty_rounds} rounds: the skeleton space is exhausted"
                )

        if len(replay) >= MIN_REPLAY and args.method != "fosa_llm":  # FOSA-LLM does not learn
            refresh_replay_rewards(replay, history, args, rnd if fosa else None, encoder)
            if args.method == "fosa_rnd":
                trainer.train_rnd(replay, rnd, TRAIN_STEPS)
            else:
                trainer.train(replay, TRAIN_STEPS)
        if rnd is not None:
            rnd.fit(encoder([program for _proposal, program, _record in proposals]).to(device))
        replay.save_jsonl(args.replay_path)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"gflownet": gfn.state_dict(), "rnd": rnd.state_dict() if rnd else None}, args.model_path)
    return 0


if __name__ == "__main__":
    main()
