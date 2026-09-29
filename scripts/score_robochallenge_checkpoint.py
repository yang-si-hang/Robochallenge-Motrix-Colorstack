"""Score a socket checkpoint on the yellow-on-orange MotrixSim task.

Scores are local proxies for RoboChallenge Table30 task 239v, not official results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from eval_openpi_socket import (StackColorBlocksEnv, IKSolver, json_safe,
                                RolloutDatasetRecorder, output_dir, run_episode, validate_metadata,
                                wait_for_health, IMAGE_WRITER_THREADS, VIDEO_ENCODER_THREADS)
from lighting_randomization import sample_lighting
from build_scene import build
from run_scene import INITIAL_EE_POSITION_RELATIVE_TO_TABLE, INITIAL_EE_QUATERNION


# Match collect_color_stack.py: it calls run_scene.sample_initial_pose(), whose
# TCP target is relative to the table center (XY) and table top (Z). Convert
# that target to world coordinates for the held-out scoring CLI defaults.
_SCENE_TABLE = json.loads((ROOT / 'configs/scene.yaml').read_text())['table']
INITIAL_POSITION = tuple(
    _SCENE_TABLE['center'][axis] + INITIAL_EE_POSITION_RELATIVE_TO_TABLE[axis]
    + (_SCENE_TABLE['half_size'][2] if axis == 2 else 0.0)
    for axis in range(3)
)
# TCP orientation uses world-frame xyzw quaternion order.
INITIAL_QUATERNION = INITIAL_EE_QUATERNION


class AttemptObserver:
    """Keep a reviewable log of command-defined attempts and lifted grasps."""

    def __init__(self, resting_z):
        self.resting_z = resting_z
        self.mode = 'open'
        self.close_attempts = 0
        self.failed_closes = 0
        self.grasp_seen = False
        self.release_attempts = 0
        self.release_open_seen = False
        self.retry_count = 0
        self.consecutive_failed_retries = 0
        self.events = []

    def __call__(self, action):
        t = float(action['sim_time_s'])
        command = float(action['gripper_command']) / 255.0
        qpos = np.asarray(action['actual_gripper_qpos'], dtype=float)
        z = float(action['blocks']['yellow'][2])
        # Hysteresis avoids turning small command fluctuations into attempts.
        next_mode = 'closed' if command >= .7 else 'open' if command <= .3 else self.mode
        if next_mode == 'closed' and self.mode != 'closed':
            self.close_attempts += 1
            self.events.append({'time_s': t, 'kind': 'close_attempt', 'number': self.close_attempts,
                                'command': command})
            if self.failed_closes:
                self.retry_count += 1
                self.consecutive_failed_retries += 1
                self.events.append({'time_s': t, 'kind': 'grasp_retry', 'deduction': .5,
                                    'after_failed_attempts': self.failed_closes})
                self.failed_closes = 0
        if next_mode == 'open' and self.mode == 'closed':
            if not self.grasp_seen:
                self.failed_closes += 1
                self.events.append({'time_s': t, 'kind': 'failed_close_cycle',
                                    'pick_z_m': z, 'gripper_qpos': qpos.tolist()})
            else:
                self.release_attempts += 1
                self.events.append({'time_s': t, 'kind': 'release_attempt',
                                    'number': self.release_attempts, 'pick_z_m': z})
                if self.release_attempts > 1:
                    self.retry_count += 1
                    self.consecutive_failed_retries += 1
                    self.events.append({'time_s': t, 'kind': 'place_retry', 'deduction': .5})
        if not self.grasp_seen and z >= self.resting_z + .02 and np.min(qpos) > .05:
            self.grasp_seen = True
            self.consecutive_failed_retries = 0
            self.events.append({'time_s': t, 'kind': 'grasp_awarded',
                                'pick_z_m': z, 'resting_z_m': self.resting_z,
                                'gripper_qpos': qpos.tolist()})
        if self.grasp_seen and next_mode == 'open' and np.max(qpos) < .12:
            if not self.release_open_seen:
                self.events.append({'time_s': t, 'kind': 'gripper_open_observed',
                                    'gripper_qpos': qpos.tolist()})
            self.release_open_seen = True
        self.mode = next_mode
        if self.consecutive_failed_retries > 4:
            self.events.append({'time_s': t, 'kind': 'early_termination',
                                'reason': 'consecutive_failed_retries_exceeded_4'})
            return 'consecutive_failed_retries_exceeded_4'
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--initial-position', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   default=INITIAL_POSITION,
                   help='world-frame TCP position for runs without --training-dataset; defaults to the collection target')
    p.add_argument('--initial-quaternion', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                   default=INITIAL_QUATERNION,
                   help='world-frame xyzw TCP quaternion for runs without --training-dataset; defaults to the collection target')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=150)
    p.add_argument('--warmup-frames', type=int, default=10)
    p.add_argument('--output', type=Path, default=Path('artifacts/robochallenge_checkpoint_score'))
    p.add_argument('--checkpoint', default='24999', help='label recorded in report; server identity is not verified')
    p.add_argument('--training-dataset', type=Path,
                   help='LeRobot training dataset directory; reads its sibling <name>_episodes.jsonl')
    p.add_argument('--selection-seed', type=int, default=20260928)
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--video-gop', type=int, default=1)
    p.add_argument('--video-bframes', type=int, default=None)
    p.add_argument('--video-qp', type=int, default=21)
    p.add_argument('--video-preset', choices=tuple(f'p{i}' for i in range(1, 8)), default='p6')
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.warmup_frames < 0 or not 1 <= args.port <= 65535:
        p.error('invalid episodes, max-steps, warmup-frames, or port')
    if args.video_gop < 1 or (args.video_bframes is not None and args.video_bframes < 0) or not 0 <= args.video_qp <= 51:
        p.error('invalid video encoder settings')
    if args.video_gop > 1 and args.video_bframes is not None and args.video_gop <= args.video_bframes + 1:
        p.error('--video-gop must exceed --video-bframes + 1')
    args.pick_color, args.place_color, args.dry_run = 'yellow', 'orange', False
    return args


def select_episodes(args):
    """Select either episodes from the specified dataset or held-out seeds."""
    if args.training_dataset is None:
        return list(range(args.seed, args.seed + args.episodes)), [None] * args.episodes, None
    dataset = args.training_dataset.resolve()
    info_path = dataset / 'meta' / 'info.json'
    log_path = dataset.parent / f'{dataset.name}_episodes.jsonl'
    if not info_path.is_file():
        raise FileNotFoundError(f'LeRobot dataset metadata missing: {info_path}')
    if not log_path.is_file():
        raise FileNotFoundError(f'episode seed/lighting log missing: {log_path}')
    info = json.loads(info_path.read_text())
    total = int(info['total_episodes'])
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    candidates = [row for row in rows if row.get('result') == 'success' and
                  row.get('pick_color') == 'yellow' and row.get('place_color') == 'orange' and
                  isinstance(row.get('episode_index'), int) and 0 <= row['episode_index'] < total]
    if len(candidates) < args.episodes:
        raise ValueError(f'{dataset}: only {len(candidates)} matching yellow/orange episodes for {args.episodes} requested')
    if len({row['episode_index'] for row in candidates}) != len(candidates):
        raise ValueError(f'duplicate training episode indices in {log_path}')
    selection = np.random.default_rng(args.selection_seed).choice(len(candidates), size=args.episodes, replace=False)
    selected = [candidates[int(i)] for i in selection]
    for row in selected:
        if ('seed' not in row or 'attempt' not in row.get('lighting', {}) or
                len(row.get('initial_end_effector_pose', [])) != 7):
            raise ValueError(f'training episode {row["episode_index"]} lacks seed, lighting attempt or initial TCP pose in {log_path}')
        pose = np.asarray(row['initial_end_effector_pose'], dtype=float)
        if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError(f'training episode {row["episode_index"]} has invalid initial TCP pose in {log_path}')
    return [row['seed'] for row in selected], selected, (dataset, log_path)


def main():
    args = parse_args()
    args.seeds, selected_rows, training_source = select_episodes(args)
    wait_for_health(args.host, args.port)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    contract = validate_metadata(metadata)
    out = output_dir(args.output)
    (out / 'selection.json').write_text(json.dumps({
        'training_dataset': str(training_source[0]) if training_source else None,
        'episode_log': str(training_source[1]) if training_source else None,
        'selection_seed': args.selection_seed if training_source else None,
        'seeds': args.seeds, 'training_episode_indices': [r['episode_index'] if r else None for r in selected_rows],
        'initial_tcp_poses': [r['initial_end_effector_pose'] if r else [*args.initial_position, *args.initial_quaternion]
                              for r in selected_rows]}, indent=2))
    config = json.loads((ROOT / 'configs/scene.yaml').read_text())
    ik = IKSolver()
    results = []
    recorder = None
    try:
        for index in range(args.episodes):
            episode_dir = out / f'episode_{index:03d}'
            episode_dir.mkdir()
            episode_config = config
            if training_source:
                row = selected_rows[index]
                args.initial_position = row['initial_end_effector_pose'][:3]
                args.initial_quaternion = row['initial_end_effector_pose'][3:]
                episode_config = sample_lighting(config, args.seeds[index], row['lighting']['attempt'])
                build(episode_config)
            env = StackColorBlocksEnv(render=True, headless=True, width=224, height=224,
                                      render_hz=20, config=episode_config, ssgi_enabled=True)
            if args.save_video and recorder is None:
                recorder = RolloutDatasetRecorder(out / 'lerobot_dataset', env.capture(),
                    save_video=True, video_options=(args.video_gop, args.video_bframes,
                                                    args.video_qp, args.video_preset),
                    image_writer_threads=IMAGE_WRITER_THREADS,
                    video_encoder_threads=VIDEO_ENCODER_THREADS)
            table = config['table']
            resting_z = float(table['center'][2] + table['half_size'][2] + config['block']['half_size'])
            observer = AttemptObserver(resting_z)
            try:
                # The shared rollout stops policy inference on a completed stack,
                # then performs its settle check before this script scores it.
                result = run_episode(env, client, contract, args, ik, index, episode_dir,
                                     dataset_recorder=recorder, action_observer=observer)
            except Exception as exc:
                if recorder is not None:
                    try:
                        recorder.commit_episode()
                    except Exception:
                        pass
                result = {'episode': index, 'seed': args.seeds[index],
                          'failure': f'{type(exc).__name__}: {exc}',
                          'traceback': traceback.format_exc(), 'duration_sim_s': 0.0,
                          'grasp_stage': False, 'place_stage': False, 'success': False}
            # A scored placement requires an observed release and a preceding grasp.
            grasp = bool(observer.grasp_seen)
            place = bool(grasp and observer.release_attempts and observer.release_open_seen
                         and result.get('place_stage'))
            retry = observer.retry_count
            stage_points = 5 * int(grasp) + 5 * int(place)
            score = stage_points - .5 * retry
            result.update({'grasp_stage': grasp, 'place_stage': place,
                           'success': bool(place and not result.get('failure')),
                           'retry_count': retry, 'retry_deduction': .5 * retry,
                           'stage_points': stage_points, 'progress_score': score,
                           'score_proxy': score, 'events': observer.events,
                           'close_attempts': observer.close_attempts,
                           'release_attempts': observer.release_attempts,
                           'scoring_notes': ['Retry events use command hysteresis; inspect events and steps.json.',
                                             'Negative scores are preserved; official display clipping is unpublished.',
                                             'Official grasp, stack and retry thresholds are unpublished.']})
            (episode_dir / 'result.json').write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
            results.append(result)
            env.close()
    finally:
        if recorder is not None:
            recorder.finalize()
    n = len(results)
    grasped = sum(r['grasp_stage'] for r in results)
    placed = sum(r['place_stage'] for r in results)
    total = sum(r['progress_score'] for r in results)
    summary = {'episodes': n, 'checkpoint_label': args.checkpoint,
               'score_identity': 'official 5+5 stage weights and 0.5 retry deductions with MotrixSim proxy decisions',
               'total_progress_score': total, 'maximum_progress_score': 10 * n,
               'mean_progress_score': total / n,
               'success_rate_pct': 100 * sum(r['success'] for r in results) / n,
               'grasp_stage_success_rate_pct': 100 * grasped / n,
               'place_stage_success_rate_all_pct': 100 * placed / n,
               'place_stage_success_rate_given_grasp_pct': 100 * placed / grasped if grasped else None,
               'mean_duration_sim_s': sum(r.get('duration_sim_s', 0) for r in results) / n,
               'official_10_rollout_protocol': n == 10}
    report = {'summary': summary, 'episodes': results, 'server_metadata': json_safe(metadata),
              'seeds': args.seeds,
              'training_dataset': str(training_source[0]) if training_source else None,
              'episode_log': str(training_source[1]) if training_source else None,
              'video_options': {'gop': args.video_gop, 'bframes': args.video_bframes,
                                'qp': args.video_qp, 'preset': args.video_preset} if args.save_video else None,
              'checkpoint_identity_verified': False,
              'limitations': ['Simulation is not the official real robot evaluation.',
                              'Initial scenes use seeded simulation, not official visual task reproduction.',
                              'Task criticality, official physical thresholds, and negative-score display are unpublished.']}
    (out / 'summary.json').write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
