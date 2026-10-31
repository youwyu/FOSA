"""The task utility V of Eq. 7 and its capability-frontier score F (Eq. 8), shared by every task-discovery method."""

from __future__ import annotations

from collections.abc import Sequence

NOVELTY_WEIGHT = 0.4  # lambda_nov
COST_WEIGHT = 0.05  # lambda_cost


def unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def success_uncertainty(success_rate: float) -> float:
    """The normalized Bernoulli variance 4p(1-p): 1 at 50% success, 0 at none or all."""
    return 4.0 * success_rate * (1.0 - success_rate)


def frontier_score(success_curve: Sequence[float] | None) -> float:
    """How close a trained task sits to the embodiment's capability frontier, in [0, 1].

    A task is on the frontier when its outcome is still maximally uncertain after
    training *and* that competence was acquired by training rather than already present:

        F = 4 J(1-J) * (J - J_0)+ / J

    The first factor is the (normalized) Bernoulli variance of the final success rate, so
    it peaks where the robot succeeds about half the time and vanishes for tasks that are
    always or never solved. The second factor is the fraction of final competence gained
    during training, which rejects tasks that merely hover at an intermediate success
    rate without learning. The score has no target or width to tune.
    """

    curve = [unit(value) for value in success_curve or ()]
    if not curve:
        return 0.0
    final, initial = curve[-1], curve[0]
    if final <= 0.0:
        return 0.0
    acquired = max(0.0, final - initial) / final
    return unit(success_uncertainty(final) * acquired)


def program_cost(num_atoms: int, status: str | None = None) -> float:
    """C(x) of Eq. 7: 0.1 per atom beyond the minimal four, plus 1.0 if training failed or 1.5 if it timed out.

    Programs have at most eight atoms, so the paper's min(1, .) cap never binds; the provided grammars complete every
    program at four bindings, so C is 0 for every trained task.
    """
    cost = 0.1 * max(0, num_atoms - 4)
    if status == "train_failed":
        cost += 1.0
    if status == "timeout":
        cost += 1.5
    return cost


def task_utility(success_curve: Sequence[float] | None, novelty: float, cost: float) -> float:
    """V(x) of Eq. 7: the frontier score F, plus NOVELTY_WEIGHT times the novelty N, less COST_WEIGHT times the
    cost C."""
    return frontier_score(success_curve) + NOVELTY_WEIGHT * unit(novelty) - COST_WEIGHT * unit(cost)
