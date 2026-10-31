"""FOSA's GFlowNet (Bengio et al., 2021) over task skeletons, trained by trajectory balance (Malkin et al., 2022).

P_F and P_B are MLPs that read a frozen text embedding of the serialized partial skeleton. Each outputs one logit per
edit of the skeleton space, STOP last, and invalid edits are masked out. With one log Z per goal family, trajectory
balance makes the sampler, within each family, proportional to the reward R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from fosa.learning.dagger_lib.text import FrozenTextEncoder
from fosa.discovery import grammar
from fosa.discovery.skeleton_space import SkeletonSpace, State


class SkeletonEncoder:
    """Mean-pooled, normalized embedding of a serialized skeleton by a frozen LLM (Qwen2.5-0.5B), cached."""

    def __init__(self, atom_grammar: Any, device: torch.device) -> None:
        self.grammar = atom_grammar
        self.backbone = FrozenTextEncoder(max_length=256, device=device)
        self.dim = self.backbone.text_dim
        self.cache: dict[str, torch.Tensor] = {}

    def __call__(self, programs: list[Any], batch_size: int = 32) -> torch.Tensor:
        texts = [grammar.serialize_atom_program(program, self.grammar) for program in programs]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            hidden, attention = self.backbone.encode_tokens(chunk)
            mask = attention.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            self.cache.update(zip(chunk, nn.functional.normalize(pooled, dim=1).cpu(), strict=True))
        return torch.stack([self.cache[text] for text in texts]) if texts else torch.empty((0, self.dim))


class Policy(nn.Module):
    """MLP (256, 256) from a skeleton embedding to one logit per edit. The last layer starts at zero, so sampling starts
    uniform over the valid edits."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class Trajectory:
    """A complete construction s_0 -> ... -> s_n -> STOP, with what the policies need at each state."""

    states: list[State]  # s_0 (the empty skeleton) to s_n (the task)
    edits: list[int]  # e_k takes s_k to s_k+1; the final STOP is implicit
    forward_valid: list[list[int]]  # the valid edits at s_0..s_n
    backward_valid: list[list[int]]  # the edits that undo into a parent of s_1..s_n


Steps = tuple[list[State], list[int], list[list[int]]]  # states, the edit taken at each, the valid edits at each


def masked_log_prob(logits: torch.Tensor, valid: list[list[int]], chosen: list[int]) -> torch.Tensor:
    """Row-wise log softmax over the valid entries, at the chosen entry."""
    mask = torch.ones_like(logits, dtype=torch.bool)
    for row, entries in enumerate(valid):
        mask[row, entries] = False
    logits = logits.clone()
    logits[mask] = -torch.inf
    rows = torch.arange(len(chosen), device=logits.device)
    return torch.log_softmax(logits, dim=1)[rows, torch.tensor(chosen, dtype=torch.long, device=logits.device)]


class GFlowNet:
    def __init__(self, space: SkeletonSpace, encoder: SkeletonEncoder, device: torch.device) -> None:
        self.space = space
        self.encoder = encoder
        self.device = device
        self.families = list(space.goal_families)
        self.forward_policy = Policy(encoder.dim, space.size).to(device)
        self.backward_policy = Policy(encoder.dim, space.size).to(device)  # skeletons form a DAG, so P_B is learned
        self.log_z = nn.Parameter(torch.zeros(len(self.families), dtype=torch.float32, device=device))
        # log Z learns faster than the policies, since its error biases every trajectory-balance gradient
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.forward_policy.parameters(), "lr": 1e-3},
                {"params": self.backward_policy.parameters(), "lr": 1e-3},
                {"params": [self.log_z], "lr": 1e-1},
            ]
        )

    def features(self, states: list[State]) -> torch.Tensor:
        return self.encoder([self.space.program(state) for state in states]).to(device=self.device, dtype=torch.float32)

    def forward_logits(self, state: State) -> torch.Tensor:
        """P_F's logits over every edit at one state, STOP last."""
        return self.forward_policy(self.features([state]))[0]

    def backward_logits(self, state: State) -> torch.Tensor:
        return self.backward_policy(self.features([state]))[0]

    def log_pf(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """log P_F of each trajectory: the sum of log P_F(e_k | s_k), STOP from s_n included."""
        steps = [(t.states, [*t.edits, self.space.stop], t.forward_valid) for t in trajectories]
        return self._sum_log_probs(self.forward_policy, steps)

    def log_pb(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """log P_B of each trajectory given its task: the sum of log P_B(s_k | s_k+1). STOP's backward probability
        is 1."""
        steps = [(t.states[1:], t.edits, t.backward_valid) for t in trajectories]
        return self._sum_log_probs(self.backward_policy, steps)

    def _sum_log_probs(self, policy: Policy, steps: list[Steps]) -> torch.Tensor:
        """Per trajectory, the summed log-probability of its steps under the policy, in one batched pass."""
        owner = [i for i, (states, _edits, _valid) in enumerate(steps) for _state in states]
        states, edits, valid = ([x for part in steps for x in part[k]] for k in range(3))
        log_p = masked_log_prob(policy(self.features(states)), valid, edits)
        index = torch.tensor(owner, dtype=torch.long, device=self.device)
        return torch.zeros(len(steps), dtype=log_p.dtype, device=self.device).index_add(0, index, log_p)

    def trajectory_balance(self, trajectories: list[Trajectory], log_rewards: list[float], family: int) -> torch.Tensor:
        """(log Z_g + log P_F(tau) - log P_B(tau | x) - log R(x))^2, averaged over trajectories of goal family g."""
        log_r = torch.tensor(log_rewards, dtype=torch.float32, device=self.device)
        return (self.log_z[family] + self.log_pf(trajectories) - self.log_pb(trajectories) - log_r).pow(2).mean()

    def step(self, loss: torch.Tensor) -> float:
        """One Adam step on the loss; P_F's gradient is clipped to norm 1."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def state_dict(self) -> dict[str, Any]:
        return {
            "forward_policy": self.forward_policy.state_dict(),
            "backward_policy": self.backward_policy.state_dict(),
            "log_z": self.log_z.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.forward_policy.load_state_dict(state["forward_policy"])
        self.backward_policy.load_state_dict(state["backward_policy"])
        self.log_z.data.copy_(state["log_z"].to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
