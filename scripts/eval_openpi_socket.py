"""Evaluate an OpenPI WebSocket policy in MotrixSim."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from stack_color_blocks import StackColorBlocksEnv
from stack_color_blocks.control import ARM_JOINTS
from stack_color_blocks.ik import IKSolver
from stack_color_blocks.openpi_adapter import (extract_actions, make_observation,
                                                pose_to_rot6d, relative_chunk_to_world,
                                                validate_metadata)
from stack_color_blocks.policy_control import PolicyController
from stack_color_blocks.task_metrics import evaluate_stack, summarize
from run_scene import forward_kinematic_pose, sample_initial_pose


COLORS = ('yellow', 'orange', 'sky_blue')
CAMERA_FEATURE_NAMES = {'global_rgb': 'base_0_rgb', 'wrist_rgb': 'left_wrist_0_rgb'}
IMAGE_WRITER_THREADS = 8
VIDEO_ENCODER_THREADS = 8
STATE_FEATURE_NAMES = ('x', 'y', 'z', 'rot6d_0', 'rot6d_1', 'rot6d_2',
                       'rot6d_3', 'rot6d_4', 'rot6d_5', 'gripper_position')
ACTION_FEATURE_NAMES = ('x', 'y', 'z', 'rot6d_0', 'rot6d_1', 'rot6d_2',
                        'rot6d_3', 'rot6d_4', 'rot6d_5', 'gripper_command')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--pick-color', choices=COLORS, default='yellow')
    parser.add_argument('--place-color', choices=COLORS, default='orange')
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=900,
                        help='maximum policy actions per episode (each action spans 0.05 sim seconds)')
    parser.add_argument('--dry-run', action='store_true',
                        help='request one chunk and solve its first execution horizon with IK, without moving the scene')
    parser.add_argument('--warmup-frames', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('data/deploy/openpi_24999'))
    args = parser.parse_args()
    if args.pick_color == args.place_color:
        parser.error('pick and place colors must differ')
    if args.port < 1 or args.port > 65535 or args.episodes < 1 or args.max_steps < 1 or args.warmup_frames < 0:
        parser.error('port, episodes, and max-steps must be positive; warmup-frames must be nonnegative')
    return args


def output_dir(path):
    path = path if path.is_absolute() else ROOT / path
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'{path} already contains files; choose a new --output directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_images(directory, images):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        for name, image in images.items():
            Image.fromarray(image, mode='RGB').save(directory / f'{name}.jpg', quality=90)
    except ImportError:
        np.savez_compressed(directory / 'images.npz', **images)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def wait_for_health(host, port, timeout_s=120):
    url = f'http://{host}:{port}/healthz'
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = response.read().decode('utf-8').strip()
                if response.status == 200 and body == 'OK':
                    return
                last_error = f'health endpoint returned {response.status}: {body!r}'
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f'OpenPI server did not become healthy within {timeout_s}s: {last_error}')


def state_snapshot(tcp_pose, gripper_qpos):
    gripper = float(np.clip(np.mean(gripper_qpos) / 0.8, 0.0, 1.0))
    state = np.r_[pose_to_rot6d(tcp_pose), gripper].astype(np.float32)
    return state


class RolloutDatasetRecorder:
    """Write executed 20 Hz transitions to a LeRobot image dataset."""

    def __init__(self, dataset_path, initial_images, save_video=False, video_options=None,
                 image_writer_threads=IMAGE_WRITER_THREADS,
                 video_encoder_threads=VIDEO_ENCODER_THREADS):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError('Install requirements-sim.txt to save LeRobot rollouts') from exc
        if dataset_path.exists() and any(dataset_path.iterdir()):
            raise FileExistsError(f'{dataset_path} already contains files; refusing to overwrite a dataset')
        expected_shapes = {name: tuple(np.asarray(image).shape)
                           for name, image in initial_images.items()}
        if set(expected_shapes) != set(CAMERA_FEATURE_NAMES):
            raise ValueError(f'expected camera keys {tuple(CAMERA_FEATURE_NAMES)}, got {tuple(expected_shapes)}')
        for name, image in initial_images.items():
            if np.asarray(image).dtype != np.uint8 or len(expected_shapes[name]) != 3 or expected_shapes[name][-1] != 3:
                raise ValueError(f'{name} must be uint8 HWC RGB, got {expected_shapes[name]}/{np.asarray(image).dtype}')
        features = {
            'observation.state': {'dtype': 'float32', 'shape': (10,), 'names': list(STATE_FEATURE_NAMES)},
            'action': {'dtype': 'float32', 'shape': (10,), 'names': list(ACTION_FEATURE_NAMES)},
            **{
                f'observation.images.{CAMERA_FEATURE_NAMES[name]}': {
                    'dtype': 'video' if save_video else 'image', 'shape': shape,
                    'names': ['height', 'width', 'channels'],
                }
                for name, shape in expected_shapes.items()
            },
        }
        self.dataset_path = dataset_path
        if save_video:
            from lerobot_video_encoder import ConfigurableVideoDataset
            dataset_type = ConfigurableVideoDataset
        else:
            dataset_type = LeRobotDataset
        create_kwargs = dict(
            repo_id='local/openpi_24999', root=dataset_path, fps=20,
            robot_type='ur5e', features=features, use_videos=save_video,
            image_writer_processes=0, image_writer_threads=image_writer_threads,
            encoder_threads=video_encoder_threads,
        )
        if save_video:
            create_kwargs['vcodec'] = 'h264_nvenc'
        self.dataset = dataset_type.create(**create_kwargs)
        if video_options is not None:
            (self.dataset.video_gop, self.dataset.video_bframes,
             self.dataset.video_qp, self.dataset.video_preset) = video_options
        self.expected_shapes = expected_shapes
        self.frame_count = 0
        self.episode_frame_count = 0
        self.current_task = None

    def begin_episode(self, task):
        if self.current_task is not None:
            raise RuntimeError('previous LeRobot episode has not been committed')
        self.current_task = task
        self.episode_frame_count = 0

    def add_transition(self, state, action, images):
        if self.current_task is None:
            raise RuntimeError('begin_episode must be called before adding LeRobot frames')
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (10,) or action.shape != (10,):
            raise ValueError(f'LeRobot state/action must have shape (10,), got {state.shape}/{action.shape}')
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('LeRobot state/action contains non-finite values')
        if set(images) != set(self.expected_shapes):
            raise ValueError(f'camera keys changed: {tuple(images)}')
        frame = {'observation.state': state, 'action': action, 'task': self.current_task}
        for name, image in images.items():
            image = np.asarray(image)
            if image.shape != self.expected_shapes[name] or image.dtype != np.uint8:
                raise ValueError(f'{name} must remain uint8 {self.expected_shapes[name]}, got {image.shape}/{image.dtype}')
            frame[f'observation.images.{CAMERA_FEATURE_NAMES[name]}'] = image
        self.dataset.add_frame(frame)
        self.frame_count += 1
        self.episode_frame_count += 1

    def commit_episode(self):
        if self.current_task is None:
            return 0
        saved = self.episode_frame_count
        if saved:
            self.dataset.save_episode(parallel_encoding=False)
        self.current_task = None
        self.episode_frame_count = 0
        return saved

    def finalize(self):
        self.dataset.finalize()


def record_executed_transition(env, chain, ik_solver, obs, recorder):
    actual_tcp = np.asarray(chain.get_end_effector_pose(env.data), dtype=float).reshape(7)
    command_tcp = forward_kinematic_pose(ik_solver, obs['target'])
    state = state_snapshot(actual_tcp, obs['gripper_qpos'])
    action = np.r_[pose_to_rot6d(command_tcp),
                   np.clip(float(obs['gripper_target']) / 255.0, 0.0, 1.0)].astype(np.float32)
    recorder.add_transition(state, action, env.capture())


def run_episode(env, client, metadata_contract, args, ik_solver, episode_index, episode_dir,
                dataset_recorder=None, action_observer=None):
    seed = args.seeds[episode_index] if hasattr(args, 'seeds') else args.seed + episode_index
    initial = env.reset(seed, randomize_blocks=True)
    chain, _pose, initial_qpos, yaw = sample_initial_pose(env, seed)
    if hasattr(args, 'initial_position') and hasattr(args, 'initial_quaternion'):
        # The checkpoint scorer supplies either its collection-matched default
        # TCP pose or the exact pose recorded in a selected training episode.
        target_pose = [*args.initial_position, *args.initial_quaternion]
        initial_qpos = np.asarray(ik_solver.solve(target_pose, initial_qpos).joint_angles, dtype=float)
    for name, value in zip(ARM_JOINTS, initial_qpos):
        env.model.get_joint(name).set_dof_pos(env.data, np.asarray([value], dtype=float))
    env.model.forward_kinematic(env.data)
    env.controller.command = np.asarray(initial_qpos, dtype=float).copy()
    for _ in range(args.warmup_frames * 25):
        env.step({'arm_qpos': initial_qpos, 'gripper_target': 0.0})

    blocks_initial = {k: v.copy() for k, v in env.get_observation()['blocks'].items()}
    table = env.config['table']
    table_top = table['center'][2] + table['half_size'][2]
    prompt = f"stack the {args.pick_color.replace('_', ' ')} block on top of the {args.place_color.replace('_', ' ')} block"
    if dataset_recorder is not None:
        dataset_recorder.begin_episode(prompt)
    controller = PolicyController(env, ik_solver, metadata_contract['control_frequency_hz'])
    rows = []
    max_pick_z = float(blocks_initial[args.pick_color][2])
    rest_positions = []
    last_arm_command = np.asarray(initial_qpos, dtype=float).copy()
    last_gripper_command = 0.0
    start_sim = float(env.time)
    wall_start = time.monotonic()
    failure = None
    success_candidate = False
    for policy_step in range(args.max_steps):
        obs = env.get_observation()
        tcp = np.asarray(chain.get_end_effector_pose(env.data), dtype=float).reshape(7).copy()
        images = env.capture()
        state = state_snapshot(tcp, obs['gripper_qpos'])
        request = make_observation(images, state, prompt)
        req_start = time.monotonic()
        try:
            response = client.infer(request)
            inference_s = time.monotonic() - req_start
            raw = extract_actions(response)
            # Every target is computed from this same request state snapshot.
            absolute = relative_chunk_to_world(raw, state)
        except Exception as exc:
            failure = f'policy_request_failed: {type(exc).__name__}: {exc}'
            rows.append({'policy_step': policy_step, 'sim_time_s': env.time,
                         'request_state': state.tolist(), 'inference_s': time.monotonic()-req_start,
                         'failure': failure})
            break
        if args.dry_run:
            save_images(episode_dir / 'images' / f'{policy_step:05d}', images)
        step_row = {'policy_step': policy_step, 'sim_time_s': float(env.time),
                    'request_tcp_pose': tcp.tolist(), 'request_state': state.tolist(),
                    'inference_s': inference_s, 'raw_actions': raw.tolist(),
                    'world_targets': absolute.tolist(), 'executed': []}
        if args.dry_run:
            seed_qpos = np.array([env.model.get_joint(name).get_dof_pos(env.data)[0]
                                  for name in ARM_JOINTS], dtype=float)
            try:
                for action_index in range(metadata_contract['execution_horizon']):
                    ik = ik_solver.solve(absolute[action_index, :7], seed_qpos)
                    seed_qpos = np.asarray(ik.joint_angles, dtype=float)
                    step_row['executed'].append({
                        'target_tcp_pose': absolute[action_index, :7].tolist(),
                        'gripper_normalized': float(absolute[action_index, 7]),
                        'arm_target_rad': seed_qpos.tolist(),
                        'ik_position_error_m': ik.position_error,
                        'ik_orientation_error_rad': ik.orientation_error,
                        'ik_iterations': ik.iterations,
                    })
            except Exception as exc:
                failure = f'ik_preflight_failed: {type(exc).__name__}: {exc}'
                step_row['failure'] = failure
            rows.append(step_row)
            break
        try:
            for action_index in range(metadata_contract['execution_horizon']):
                result, execution = controller.execute(absolute[action_index])
                max_pick_z = max(max_pick_z, float(result['blocks'][args.pick_color][2]))
                last_arm_command = np.asarray(execution['arm_target_rad'], dtype=float)
                last_gripper_command = float(execution['gripper_command'])
                execution.update({'sim_time_s': float(result['time']),
                                  'actual_tcp_pose': np.asarray(chain.get_end_effector_pose(env.data), dtype=float).reshape(7).tolist(),
                                  'actual_arm_qpos': result['arm_qpos'].tolist(),
                                  'actual_gripper_qpos': result['gripper_qpos'].tolist(),
                                  'blocks': {k: v.tolist() for k, v in result['blocks'].items()}})
                if dataset_recorder is not None:
                    record_executed_transition(env, chain, ik_solver, result, dataset_recorder)
                step_row['executed'].append(execution)
                if action_observer is not None:
                    stop_reason = action_observer(execution)
                    if stop_reason:
                        failure = stop_reason
                        break
                # Once a released block forms a valid stack, stop asking the policy
                # for actions. The existing settle window below still verifies it.
                blocks = result['blocks']
                pick = np.asarray(blocks[args.pick_color], dtype=float)
                support = np.asarray(blocks[args.place_color], dtype=float)
                table_top = env.config['table']['center'][2] + env.config['table']['half_size'][2]
                resting_center_z = table_top + env.config['block']['half_size']
                xy_error = float(np.linalg.norm(pick[:2] - support[:2]))
                height_delta = float(pick[2] - support[2])
                expected_delta = 2.0 * float(env.config['block']['half_size'])
                stack_contact = (xy_error <= 0.01 and abs(height_delta - expected_delta) <= 0.008
                                 and np.arccos(np.clip(1.0 - 2.0 * np.sum(pick[3:5] ** 2), -1.0, 1.0)) <= 0.15
                                 and np.arccos(np.clip(1.0 - 2.0 * np.sum(support[3:5] ** 2), -1.0, 1.0)) <= 0.15)
                gripper_open = float(np.max(result['gripper_qpos'])) < 0.12
                if max_pick_z >= resting_center_z + 0.02 and stack_contact and gripper_open:
                    success_candidate = True
                    break
            if failure:
                rows.append(step_row)
                break
            step_row['result_tcp_pose'] = np.asarray(chain.get_end_effector_pose(env.data), dtype=float).reshape(7).tolist()
            step_row['result_blocks'] = {k: v.tolist() for k, v in result['blocks'].items()}
            rows.append(step_row)
            if success_candidate:
                break
        except Exception as exc:
            failure = f'action_execution_failed: {type(exc).__name__}: {exc}'
            step_row['failure'] = failure
            rows.append(step_row)
            break
    if not args.dry_run and failure is None and not success_candidate and len(rows) >= args.max_steps:
        failure = 'max_policy_steps_reached'
    # Match the task's 0.5 s settle observation while holding the last policy command.
    if not args.dry_run:
        for settle_step in range(250):
            settled = env.step({'arm_qpos': last_arm_command,
                                'gripper_target': last_gripper_command})
            rest_positions.append(settled['blocks'][args.pick_color][:3].copy())
            if dataset_recorder is not None and (settle_step + 1) % 25 == 0:
                record_executed_transition(env, chain, ik_solver, settled, dataset_recorder)
    duration_sim = float(env.time - start_sim)
    final_blocks = {k: v.copy() for k, v in env.get_observation()['blocks'].items()}
    if args.dry_run:
        metrics = {'dry_run': True, 'success': None, 'grasp_stage': None,
                   'place_stage': None, 'progress_score_no_retry': None,
                   'inference_requests': len(rows)}
    else:
        metrics = evaluate_stack(blocks_initial, final_blocks, rest_positions,
                                 args.pick_color, args.place_color, table_top,
                                 env.config['block']['half_size'], max_pick_z=max_pick_z)
    if failure:
        metrics['success'] = False
        metrics['failure'] = failure
    metrics.update({'episode': episode_index, 'seed': seed,
                    'pick_color': args.pick_color, 'place_color': args.place_color,
                    'duration_sim_s': duration_sim,
                    'duration_wall_s': time.monotonic() - wall_start,
                    'steps_used': len(rows), 'initial_ee_yaw_deg': yaw,
                    'initial_blocks': {k: v.tolist() for k, v in blocks_initial.items()},
                    'final_blocks': {k: v.tolist() for k, v in final_blocks.items()},
                    'policy_metadata': metadata_contract})
    (episode_dir / 'steps.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (episode_dir / 'result.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dataset_recorder is not None:
        metrics['lerobot_frames'] = dataset_recorder.commit_episode()
        metrics['lerobot_dataset'] = str(dataset_recorder.dataset_path)
        (episode_dir / 'result.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    args = parse_args()
    training_log = ROOT / 'data/color_stack_v1.1_episodes.jsonl'
    if training_log.exists():
        trained_seeds = {json.loads(line)['seed'] for line in training_log.read_text().splitlines() if line.strip()}
        overlap = sorted(set(range(args.seed, args.seed + args.episodes)) & trained_seeds)
        if overlap:
            raise ValueError(f'evaluation seed overlaps color_stack_v1.1 training data: {overlap}')
    out = output_dir(args.output)
    wait_for_health(args.host, args.port)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as exc:
        raise RuntimeError('Install requirements-inference.txt before running socket evaluation') from exc

    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    contract = validate_metadata(metadata)
    base_config = json.loads((ROOT / 'configs/scene.yaml').read_text())
    env = StackColorBlocksEnv(render=True, headless=True, width=224, height=224,
                              render_hz=20, config=base_config, ssgi_enabled=True)
    ik_solver = IKSolver()
    dataset_recorder = None
    if not args.dry_run:
        dataset_recorder = RolloutDatasetRecorder(
            out / 'lerobot_dataset', env.capture(),
            image_writer_threads=IMAGE_WRITER_THREADS,
            video_encoder_threads=VIDEO_ENCODER_THREADS,
        )
    results = []
    try:
        for i in range(args.episodes):
            episode_dir = out / f'episode_{i:03d}'
            episode_dir.mkdir(parents=True, exist_ok=False)
            try:
                result = run_episode(env, client, contract, args, ik_solver, i, episode_dir,
                                     dataset_recorder=dataset_recorder)
            except Exception as exc:
                partial_frames = 0
                if dataset_recorder is not None:
                    try:
                        partial_frames = dataset_recorder.commit_episode()
                    except Exception as save_exc:
                        exc = RuntimeError(f'{exc}; LeRobot partial episode save failed: {save_exc}')
                result = {'episode': i, 'seed': args.seed + i,
                          'success': False, 'grasp_stage': False, 'place_stage': False,
                          'progress_score_no_retry': 0.0, 'duration_sim_s': 0.0,
                          'lerobot_frames': partial_frames,
                          'failure': f'{type(exc).__name__}: {exc}',
                          'traceback': traceback.format_exc()}
                (episode_dir / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
            results.append(result)
    finally:
        env.close()
        if dataset_recorder is not None:
            dataset_recorder.finalize()
    report = {'checkpoint_contract': contract, 'server_metadata': json_safe(metadata),
              'episodes': results,
              'lerobot_dataset': None if dataset_recorder is None else str(dataset_recorder.dataset_path),
              'summary': None if args.dry_run else summarize(results)}
    (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report['summary'] if report['summary'] is not None else {
        'dry_run': True, 'episodes': len(results),
        'all_ik_preflight_passed': all(not r.get('failure') for r in results)},
        indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
