"""Local, retrospective checkpoint probes. No optimizer or environment steps."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sarrl.controllers import ComputedTorqueController
from sarrl.dynamics import PlanarArm
from sarrl.evaluation import planar_safety_config
from sarrl.rl import SACAgent
from sarrl.safety import HOCBFSafetyFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CONDITIONS = ('C0_posthoc_hocbf', 'C1_inloop_hocbf')
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def critic_probe(payload, condition, seed):
    agent = SACAgent.from_state_dict(payload['agent'], seed=731000 + seed)
    replay = payload['replay']
    n = int(replay['size'])
    assert n == 200000
    abort = (replay['dones'][:n, 0] == 1) & (replay['rewards'][:n, 0] == -500)
    if condition == CONDITIONS[0]:
        assert not abort.any()
    assert np.array_equal(replay['obs'][:n][abort], replay['next_obs'][:n][abort])
    groups = {}
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        batch = {k: torch.as_tensor(replay[k][start:stop], device=agent.device)
                 for k in ('obs', 'actions', 'rewards', 'next_obs', 'dones')}
        q1 = agent.q1(batch['obs'], batch['actions'])
        q2 = agent.q2(batch['obs'], batch['actions'])
        targets = []
        for _ in range(4):
            action, logp, _ = agent.actor.sample(batch['next_obs'])
            qnext = torch.minimum(agent.q1_target(batch['next_obs'], action),
                                  agent.q2_target(batch['next_obs'], action))
            targets.append(agent.compute_bellman_target(batch['rewards'], batch['dones'], qnext, logp))
        # Report both expected single-draw loss and residual to the 4-draw mean.
        target = torch.stack(targets)
        loss = (((q1 - target).square() + (q2 - target).square()) / 2).mean(0)
        mean_target = target.mean(0)
        mean_loss = ((q1 - mean_target).square() + (q2 - mean_target).square()) / 2
        signed = (q1 + q2) / 2 - mean_target
        masks = {'all': np.ones(stop-start, dtype=bool), 'abort': abort[start:stop],
                 'success_terminal': (replay['dones'][start:stop, 0] == 1) & ~abort[start:stop],
                 'nonterminal': replay['dones'][start:stop, 0] == 0}
        values = {'sse': loss, 'mean_target_sse': mean_loss, 'signed_sum': signed,
                  'abs_sum': signed.abs(), 'critic_gap_sum': (q1-q2).abs()}
        arrays = {k: v.cpu().numpy().reshape(-1) for k, v in values.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        for group, mask in masks.items():
            stat = groups.setdefault(group, {'n': 0, **{k: 0.0 for k in arrays}})
            stat['n'] += int(mask.sum())
            for key, values in arrays.items():
                stat[key] += float(values[mask].astype(np.float64).sum())
    for stat in groups.values():
        count = stat['n']
        stat['rmse'] = (stat['sse']/count)**0.5 if count else None
        stat['mean_target_rmse'] = (stat['mean_target_sse']/count)**0.5 if count else None
    groups['abort']['loss_share'] = groups['abort']['sse']/groups['all']['sse']
    return {'condition': condition, 'seed': seed, 'alpha': agent.alpha.item(), 'groups': groups}


@torch.inference_mode()
def action_probe(path, observations, seed):
    agent = SACAgent.from_checkpoint(path, seed=732000 + seed)
    tensor = torch.as_tensor(observations, device=agent.device)
    a1, logp, deterministic = agent.actor.sample(tensor)
    a2, _, _ = agent.actor.sample(tensor)
    first, second = a1.cpu().numpy(), a2.cpu().numpy()
    nominal = PlanarArm()
    controller = ComputedTorqueController(nominal)
    safety = HOCBFSafetyFilter(nominal, planar_safety_config())
    corrections, projected_gaps, raw_gaps, outside, infeasible = [], [], [], [], 0
    for obs, u, v in zip(observations, first, second, strict=True):
        state = np.concatenate([obs[:2].astype(float)*np.pi, obs[2:4].astype(float)*8])
        target = obs[4:6].astype(float)*2
        qdes = nominal.inverse_kinematics(target)
        baseline = controller.command(state[:2], state[2:], qdes)
        r1 = safety.filter(state, baseline + 8*u)
        r2 = safety.filter(state, baseline + 8*v)
        if not (r1.success and r2.success):
            infeasible += 1
            continue
        corrections.append(r1.correction_norm)
        raw_gaps.append(float(np.linalg.norm(8*(u-v))))
        projected_gaps.append(float(np.linalg.norm(r1.torque-r2.torque)))
        outside.append(bool(np.any(np.abs((r1.torque-baseline)/8) > 1+1e-6)))
    det = deterministic.cpu().numpy()
    return {'probes': len(observations), 'feasible_pairs': len(corrections),
            'infeasible_pairs': infeasible,
            'sample_abs_mean': float(np.abs(first).mean()),
            'sample_saturation_fraction': float((np.abs(first)>.95).mean()),
            'deterministic_abs_mean': float(np.abs(det).mean()),
            'entropy_mean': float(-logp.mean().item()),
            'correction_mean_feasible': float(np.mean(corrections)),
            'intervention_fraction_feasible': float(np.mean(np.asarray(corrections)>1e-6)),
            'mean_raw_pair_gap': float(np.mean(raw_gaps)),
            'mean_projected_pair_gap': float(np.mean(projected_gaps)),
            'projected_residual_outside_actor_box_fraction': float(np.mean(outside))}


def main():
    report = {'scope': 'retrospective exploratory, not a formal gate',
              'critic_scope': 'final 200k agents on their own complete training replay; not heldout value calibration',
              'target_draws': 4,
              'projection_scope': 'selected best agents, common mixture of replay states; nominal reconstruction from float32, no plant rollout',
              'critic': [], 'actions': [], 'inputs': []}
    hashes = {}
    for seed in range(20,25):
        pools = []
        rng = np.random.default_rng(733000+seed)
        for condition in CONDITIONS:
            directory = ROOT/'training'/condition/f'seed_{seed}'
            path = directory/'training_final.pt'
            before = sha(path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            assert payload['loop_state']['step'] == 200000
            result = critic_probe(payload, condition, seed)
            report['critic'].append(result)
            obs = payload['replay']['obs'][:int(payload['replay']['size'])]
            # Clipped q/qd cannot be inverted, so exclude them from nominal probes.
            valid = np.flatnonzero((np.abs(obs[:,:4]) < .99999).all(1))
            pools.append(obs[rng.choice(valid,128,replace=False)].copy())
            hashes[str(path)] = before
            report['inputs'].append({'path': str(path.relative_to(ROOT)), 'sha256': before})
            print(json.dumps({'critic': result}), flush=True)
            del payload
        observations = np.concatenate(pools)
        for condition in CONDITIONS:
            path = ROOT/'training'/condition/f'seed_{seed}'/'best.pt'
            before = sha(path)
            result = {'condition': condition, 'seed': seed, **action_probe(path, observations, seed)}
            report['actions'].append(result)
            hashes[str(path)] = before
            report['inputs'].append({'path': str(path.relative_to(ROOT)), 'sha256': before})
            print(json.dumps({'actions': result}), flush=True)
    assert all(sha(Path(path)) == digest for path,digest in hashes.items())
    report['checkpoint_hashes_unchanged'] = True
    report['script_sha256'] = sha(Path(__file__))
    (OUT/'checkpoint_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Audit complete. All 20 checkpoint hashes unchanged.', flush=True)


if __name__ == '__main__':
    main()
