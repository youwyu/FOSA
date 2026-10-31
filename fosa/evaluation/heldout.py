"""Held-out evaluation: distilled students on tasks that no discovery run trained on.

The suite (TASKS below) is shared by every run and lives in logs/heldout:
    prepare   compile each task's scene with the LLM compiler            -> logs/heldout/scenes/<robot>/
    train     train each scene's reference PPO policy, as many as GPUs   -> logs/heldout/teachers/<robot>/
    evaluate  roll out the reference policy and every student, 128 episodes per task
A task is scored only if its reference policy succeeds at least half the time. Success retention is
min(1, SR_student / SR_reference); reward retention is fosa.learning.dagger.reward_retention.

usage: python -m fosa.evaluation.heldout {prepare,train,evaluate} ROBOT [--students DIR | --checkpoint PATH]
"""

# ruff: noqa: E402  (the allocator is chosen before fosa.learning.dagger, which sets its own, is imported)
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")  # evaluation has its GPU alone: reuse allocations

import numpy as np
import torch

from fosa import REPO_ROOT, llm
from fosa.compiler.generate_env import generate_environment
from fosa.discovery.common import NUM_ENVS as PPO_ENVS
from fosa.evaluation import METHODS
from fosa.learning.dagger import CONFIG, evaluate_policy, load_student, reward_retention
from fosa.learning.dagger_lib.runtime import discover_teachers
from fosa.learning.dagger_lib.text import TaskText, TaskTextEncoder

SUITE = REPO_ROOT / "logs" / "heldout"
MIN_REFERENCE_SUCCESS = 0.5  # a task its own reference policy cannot solve scores every student near zero
EPISODES = 128  # parallel evaluation episodes per task
SEED = 2_000_000  # evaluation seed of the first task (plus the task's index), apart from every training seed
TASKS = {  # robot -> held-out task name -> description; the order sets each task's evaluation seed
    "jackal": {
        "mud_track": "Drive across a 4 m patch of low-traction mud with friction coefficient 0.4 while holding a 0.5 m/s forward command.",
        "pothole_field": "Drive over a field of shallow 4 cm potholes at 0.6 m/s while keeping the heading steady.",
        "weak_left_wheels": "Drive straight ahead at 0.6 m/s on flat ground while the left wheels' motors deliver 60% of their normal torque.",
        "uphill_climb": "Climb a 10-degree incline while tracking a 0.6 m/s forward command without stalling.",
        "tight_turn": "Turn in place at 1.5 rad/s on flat ground while keeping the base centred.",
        "sand_patch": "Cross a 3 m patch of high rolling resistance while holding a 0.5 m/s forward command.",
        "curb_climb": "Drive over a 6 cm curb at 0.5 m/s without stalling or flipping.",
        "washboard": "Cross a washboard ripple surface at 1.0 m/s while keeping the heading steady.",
        "trailer_tow": "Tow a 4 kg rigid trailer payload attached behind the base while tracking 0.8 m/s forward and 0.5 rad/s yaw commands on flat ground.",
        "ice_patch_crossing": "Drive across a 3 m long low-friction ice patch while holding a 0.6 m/s forward command without spinning out.",
        "speed_bumps": "Track a 1.2 m/s forward command across a row of 5 cm high speed bumps spaced 1.5 m apart on flat ground.",
        "gravel_reverse": "Reverse at 0.5 m/s across a rough gravel heightfield while keeping the heading steady.",
    },
    "inspire_hand": {
        "sphere_roll_60": "Roll a ball 60 degrees about a horizontal axis with the fingers and hold the new orientation above the palm.",
        "palm_ball_balance": "Keep a ball resting on the open palm for the whole episode without letting it roll off the hand.",
        "sphere_lift_large": "Raise a large ball already within the fingers by about 2 cm and hold it above the palm without dropping it.",
        "cuboid_slide": "Slide a small rectangular block about 2 cm toward the fingertips and hold it above the palm.",
        "cuboid_pitch": "Pitch a small rectangular block 25 degrees about the horizontal axis and hold that orientation above the palm.",
        "cylinder_upright": "Return a tipped short cylinder to upright above the palm and hold it there.",
        "cuboid_turn_90": "Turn a small rectangular block 90 degrees counterclockwise about the vertical axis and hold that orientation above the palm.",
        "sphere_translate": "Move a ball about 3 cm laterally across the palm without dropping it.",
        "sphere_lift": "Raise a ball already within the fingers by about 2 cm and hold it above the palm without dropping it.",
        "cuboid_turn": "Turn a small rectangular block 30 degrees counterclockwise about the vertical axis and hold that orientation above the palm.",
        "cylinder_tip": "Tip a short upright cylinder by 20 degrees toward the wrist and hold it above the palm without dropping it.",
        "cuboid_turn_cw": "Turn a small rectangular block 45 degrees clockwise about the vertical axis and hold that orientation above the palm.",
    },
    "go2": {
        "weak_front_legs": "Walk forward at 0.4 m/s on flat ground while both front legs' motors deliver only half of their normal torque.",
        "narrow_beam_walk": "Walk forward at 0.3 m/s along a 20 cm wide raised walkway without stepping off.",
        "ice_stop": "Walk forward at 0.4 m/s on a floor with friction coefficient 0.2 and come to a controlled stop without falling.",
        "weak_hind_leg": "Walk forward at 0.5 m/s on flat ground while one hind leg's motors deliver only 60% of their normal torque.",
        "shove_walk": "Keep walking forward at 0.4 m/s while being shoved sideways by 60 N impulses every few seconds, without falling.",
        "stair_descent": "Descend four 10 cm steps at 0.4 m/s without falling.",
        "lateral_walk": "Track a 0.4 m/s sideways command on flat ground without drifting forward.",
        "rubble_field": "Walk over a rough 5 cm amplitude heightfield while tracking a 0.5 m/s forward command.",
        "push_recovery": "Hold position on flat ground while repeated 100 N lateral pushes are applied.",
        "block_field": "Walk through a field of 8 cm high square blocks at 0.4 m/s, stepping onto or over the blocks without falling.",
        "downhill_walk": "Walk down a 15-degree slope while tracking a 0.5 m/s forward command without tumbling.",
        "slope_turn": "Turn in place at 1.0 rad/s on a 10-degree slope without drifting downhill.",
    },
    "booster": {
        "single_shove_recovery": "Stand upright and recover, without falling, from a single 60 N lateral shove applied midway through the episode.",
        "crouch_hold": "Lower the base about 10 cm into a crouch and hold it, staying upright, for the rest of the episode.",
        "quarter_turn": "Turn 90 degrees in place and then stand still facing the new direction, staying upright.",
        "reach_to_knees": "Stand upright, lower both hands toward the knees and return them to the sides, without losing balance.",
        "wave_while_standing": "Stand upright and wave one arm side to side for the whole episode without losing balance.",
        "walk_and_stop": "Walk forward at 0.4 m/s, then stop within 0.5 m of the goal marker and hold upright.",
        "push_stand": "Remain standing under repeated 40 N lateral pushes without taking a step.",
        "gentle_ramp_up": "Walk forward at 0.4 m/s up a gentle 5-degree ramp while staying upright.",
        "gentle_ramp_down": "Walk down a 5-degree ramp at 0.3 m/s while staying upright.",
        "low_bar_steps": "Step over a series of 5 cm high bars spaced 1 m apart while walking forward at 0.3 m/s.",
        "turn_and_go": "Turn 180 degrees in place, then walk 2 m to the goal marker and stop upright.",
    },
}


def prepare(robot: str) -> None:
    """Compile the scene of every task that has none yet; a rerun retries a task the compiler gave up on."""
    scenes = SUITE / "scenes" / robot
    for name, description in TASKS[robot].items():
        if not (scenes / f"heldout_{name}.json").exists():
            generate_environment(description, robot=robot, module_name=f"heldout_{name}", output_dir=scenes)


def train(robot: str) -> None:
    """Train the reference policy of every scene that has none, as many at a time as there are GPUs."""
    teachers = SUITE / "teachers" / robot
    teachers.mkdir(parents=True, exist_ok=True)
    trained = {teacher.key for teacher in discover_teachers(teachers, robot=robot)}
    pending = [
        scene for scene in sorted((SUITE / "scenes" / robot).glob("heldout_*.json")) if scene.stem not in trained
    ]
    width = torch.cuda.device_count()
    for start in range(0, len(pending), width):
        wave = [(scene, subprocess.Popen(ppo_command(scene, teachers))) for scene in pending[start : start + width]]
        for scene, process in wave:  # a failed run leaves the task without a reference; evaluate() then skips it
            if process.wait() != 0:
                print(f"[heldout] reference training failed for {scene.stem}", flush=True)


def ppo_command(scene: Path, log_root: Path) -> list[str]:
    task = json.loads(scene.read_text())["task_id"]
    return [
        sys.executable,
        "-m",
        "fosa.learning.ppo",
        "--task",
        task,
        "--gpu-ids",
        "all",
        "--num-envs",
        str(PPO_ENVS),
        "--run-name",
        scene.stem,
        "--log-root",
        str(log_root),
        "--scene-spec",
        str(scene),
    ]


def evaluate(robot: str, checkpoints: dict[str, Path], output: Path) -> None:
    """Evaluate each named student checkpoint on every task that has a reference policy; write the rows to output."""
    device = torch.device("cuda")
    available = {teacher.key: teacher for teacher in discover_teachers(SUITE / "teachers" / robot, robot=robot)}
    teachers = [available[f"heldout_{name}"] for name in TASKS[robot] if f"heldout_{name}" in available]
    students = {name: load_student(path, device)[0].eval() for name, path in checkpoints.items()}
    window = max(int(student.max_text_tokens) for student in students.values())  # the text window used in training
    encoded = TaskTextEncoder(max_length=window).encode([teacher.description for teacher in teachers])
    horizon = next(iter(students.values())).history_horizon
    results = {name: [] for name in students}
    for index, teacher in enumerate(teachers):
        text = TaskText(encoded.features[index : index + 1], encoded.padding[index : index + 1])
        rollout = dict(text=text, device=device, num_envs=EPISODES, seed=SEED + index)
        reference = evaluate_policy(teacher, student=None, history_horizon=horizon, **rollout)
        if reference["success_rate"] < MIN_REFERENCE_SUCCESS:
            print(f"[heldout] skipping {teacher.key}: reference success {reference['success_rate']:.2f}", flush=True)
            continue
        for name, student in students.items():
            result = evaluate_policy(teacher, student=student, history_horizon=student.history_horizon, **rollout)
            results[name].append(
                {
                    "task": teacher.key,
                    "description": teacher.description,
                    "student_reward": result["mean_step_reward"],
                    "teacher_reward": reference["mean_step_reward"],
                    "student_success_rate": result["success_rate"],
                    "teacher_success_rate": reference["success_rate"],
                    "reward": reward_retention(reference["mean_step_reward"], result["mean_step_reward"]),
                }
            )
            print(
                f"{robot}/{teacher.key} {name}: reward={result['mean_step_reward']:.6f} "
                f"success={result['success_rate']}",
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"robot": robot, "methods": results}, indent=2) + "\n")
    for name, rows in results.items():
        success = np.mean([min(1.0, row["student_success_rate"] / row["teacher_success_rate"]) for row in rows])
        reward = np.mean([row["reward"] for row in rows])
        print(f"{robot} {name}: success retention {success:.3f}, reward retention {reward:.3f} over {len(rows)} tasks")
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "train", "evaluate"))
    parser.add_argument("robot", choices=TASKS)
    parser.add_argument(
        "--students",
        type=Path,
        default=llm.model_log_root() / "students" / "seed1",
        help="Students of scripts/slurm_student.sh, <students>/<robot>/<method>/; results go to "
        "<students>/heldout/<robot>.json (default: %(default)s).",
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS), help="Students to evaluate (default: all).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate this one student checkpoint instead; results go to heldout_<robot>.json next to it.",
    )
    args = parser.parse_args(argv)
    if args.step == "prepare":
        prepare(args.robot)
    elif args.step == "train":
        train(args.robot)
    elif args.checkpoint:
        evaluate(args.robot, {"student": args.checkpoint}, args.checkpoint.with_name(f"heldout_{args.robot}.json"))
    else:
        final = f"checkpoints/student_round_{CONFIG.rounds:03d}.pt"
        checkpoints = {method: args.students / args.robot / method / final for method in args.methods}
        evaluate(args.robot, checkpoints, args.students / "heldout" / f"{args.robot}.json")


if __name__ == "__main__":
    main()
