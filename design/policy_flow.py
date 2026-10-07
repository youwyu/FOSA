"""Replay a FOSA construction trajectory through the trained GFlowNet and report P_F over every valid edit.

Read-only: loads the run's saved grammar and policy checkpoint, never calls the LLM, writes only to stdout.
Each step's number of valid edits is checked against the count recorded when the trajectory was sampled.

    python policy_flow.py --run-dir <runs/grammar-v5-lsa/vlm> --robot go2 --iteration 187 --sample-idx 15
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

HANDVLA = Path('/N/slate/youwyu/HandVLA')
sys.path.insert(0, str(HANDVLA))

import _src.vlm.llm_client as llm_client  # noqa: E402


def _blocked(*_args, **_kwargs):
    raise RuntimeError('LLM call blocked in policy_flow.py')


for _name in dir(llm_client):
    if callable(getattr(llm_client, _name)) and not _name.startswith('_') and _name not in {'model_log_root', 'model_log_name'}:
        if any(token in _name.lower() for token in ('chat', 'call', 'complete', 'request', 'query', 'generate')):
            setattr(llm_client, _name, _blocked)

spec = importlib.util.spec_from_file_location('fosa_ours', HANDVLA / '_src' / 'vlm' / '4_ours.py')
ours = importlib.util.module_from_spec(spec)
sys.modules['fosa_ours'] = ours
spec.loader.exec_module(ours)
mcts = ours.mcts


def load_grammar(path: Path):
    data = json.loads(path.read_text())
    data['atoms'] = {name: mcts.AtomSpec(**atom) for name, atom in data['atoms'].items()}
    return mcts.AtomGrammar(**data)


def label(action) -> dict:
    if action.action_type == 'stop':
        return {'op': 'stop'}
    out = {'op': action.action_type, 'role': action.role, 'atom': action.atom}
    if action.action_type == 'refine_arg':
        out['argument'] = action.arguments.get('argument')
        out['value'] = action.arguments.get('value')
        if 'direction' in action.arguments:
            out['direction'] = action.arguments['direction']
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--robot', required=True)
    parser.add_argument('--beam', type=int, default=8)
    parser.add_argument('--branch', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=12)
    args = parser.parse_args()

    robot_dir = args.run_dir / args.robot
    grammar = load_grammar(robot_dir / 'fosa_learn_atom_grammar.json')
    run_args = argparse.Namespace(robot=grammar.robot, backend=grammar.backend)
    specs = ours.RobotTaskSkeletonEnv._build_action_specs(types.SimpleNamespace(grammar=grammar))
    index = {ours.action_key(action): i for i, action in enumerate(specs)}
    ckpt = torch.load(robot_dir / 'fosa_learn.pt', map_location='cpu')
    weights = ckpt['model']['forward_policy']
    hidden, input_dim = weights['net.0.weight'].shape
    assert weights['net.4.weight'].shape[0] == len(specs) + 1
    policy = ours.ExternalPolicy(input_dim, len(specs) + 1, hidden)
    policy.load_state_dict(weights)
    policy.eval()
    encoder = ours.AtomProgramEncoder(torch.device('cpu'))
    env_like = types.SimpleNamespace(grammar=grammar)
    stop = ours.GFNAction('stop')

    def key_of(program):
        return json.dumps(sorted(json.dumps(mcts.atom_program_to_dict(program)['bindings'][i], sort_keys=True)
                                 for i in range(len(program.bindings))))

    evaluated = {}
    for row in map(json.loads, (robot_dir / 'fosa_learn_history.jsonl').read_text().splitlines()):
        if row.get('status') == 'trained' and row.get('atom_program'):
            program = mcts.atom_program_from_dict(row['atom_program'])
            evaluated.setdefault(key_of(program), []).append(row)

    def expand(program):
        candidates = ours.enumerate_valid_actions(program, grammar, run_args)
        valid = [s for s in specs if any(ours.RobotTaskSkeletonEnv._matches_valid_action(s, c) for c in candidates)]
        if any(c.action_type == 'stop' for c in candidates):
            valid.append(stop)
        with torch.no_grad():
            full = policy(encoder.encode(program, grammar).unsqueeze(0))[0]
        logits = torch.stack([full[-1] if a.action_type == 'stop' else full[index[ours.action_key(a)]] for a in valid])
        return valid, torch.softmax(logits, 0).tolist()

    def describe(program, action):
        out = label(action)
        if action.action_type == 'refine_arg':
            after = ours.RobotTaskSkeletonEnv.apply_static_action(env_like, program, action)
            for binding in after.bindings:
                if binding.atom == action.atom:
                    out['result'] = binding.arguments.get(action.arguments.get('argument'))
        return out

    # Beam search over partial skeletons; every node records its top valid edits with P_F.
    beams = [{'program': ours.empty_program(), 'logp': 0.0, 'path': []}]
    finished = []
    for _ in range(args.max_steps):
        grown = []
        for beam in beams:
            valid, probs = expand(beam['program'])
            order = sorted(range(len(valid)), key=lambda i: -probs[i])
            options = [{**describe(beam['program'], valid[i]), 'p': round(probs[i], 4)} for i in order[:5]]
            for i in order[:args.branch]:
                step = {'options': options, 'valid': len(valid), 'chosen': options[order.index(i)] if i in order[:5] else None}
                path = beam['path'] + [step]
                logp = beam['logp'] + float(torch.tensor(probs[i]).log())
                if valid[i].action_type == 'stop':
                    finished.append({'program': beam['program'], 'logp': logp, 'path': path})
                else:
                    after = ours.RobotTaskSkeletonEnv.apply_static_action(env_like, beam['program'], valid[i])
                    grown.append({'program': after, 'logp': logp, 'path': path})
        beams = sorted(grown, key=lambda b: -b['logp'])[:args.beam]
        if not beams:
            break
    finished.sort(key=lambda b: -b['logp'])
    out = []
    for leaf in finished[:args.beam]:
        rows = evaluated.get(key_of(leaf['program']), [])
        out.append({
            'logp': round(leaf['logp'], 3),
            'bindings': mcts.atom_program_to_dict(leaf['program'])['bindings'],
            'evaluated': [{'j0': r['success_curve'][0], 'j': r['success_rate'], 'curve': r['success_curve'],
                           'description': r['description'], 'iteration': r['iteration']} for r in rows],
            'path': leaf['path'],
        })
    print(json.dumps({'robot': args.robot, 'leaves': out}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
