"""Collect successful randomized color stacking episodes into one LeRobot dataset.

.venv/bin/python scripts/collect_color_stack.py \
  --seed 20260926 --pick-color yellow --place-color orange --episodes 70 \
  --max-attempts 1000 --output data/color_stack_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import run_scene
from build_scene import build
from lighting_randomization import capture_ready, image_quality, sample_lighting
from stack_color_blocks import StackColorBlocksEnv
from stack_color_blocks.control import ARM_JOINTS

COLORS = ('yellow', 'orange', 'sky_blue')
STATE_SIZE = len(run_scene.POSE_FEATURE_NAMES) + 1
ACTION_SIZE = len(run_scene.POSE_FEATURE_NAMES) + 1
CAMERA_FEATURE_NAMES = {'global_rgb': 'base_0_rgb', 'wrist_rgb': 'left_wrist_0_rgb'}
EE_FEATURE_NAMES = [f'ee.{name}' for name in run_scene.POSE_FEATURE_NAMES] + ['ee.gripper_pos']
IMAGE_WRITER_THREADS = 8
VIDEO_ENCODER_THREADS = 8


class RetryEpisode(Exception):
    """A stochastic camera/trajectory issue that should be resampled."""


class RecordingError(RuntimeError):
    deterministic_recording_error = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1, help='master seed for unique per-attempt seeds')
    parser.add_argument('--pick-color', choices=COLORS, default='yellow')
    parser.add_argument('--place-color', choices=COLORS, default='orange')
    parser.add_argument('--episodes', type=int, default=100,
                        help='number of successful episodes for the selected task')
    parser.add_argument('--max-attempts', type=int, default=1000,
                        help='maximum attempts for the selected task')
    parser.add_argument('--output', type=Path, default=Path('data/color_stack_v2'))
    parser.add_argument('--max-steps', type=int, default=90000)
    parser.add_argument('--warmup-frames', type=int, default=10,
                        help='hold each episode at its initial pose for this many dataset frames before recording')
    parser.add_argument('--randomize-background', action='store_true')
    parser.add_argument('--save-video', action='store_true',
                        help='store camera streams as LeRobot video features instead of individual images')
    parser.add_argument('--video-gop', type=int, default=1,
                        help='NVENC GOP length; 1 means all-intra (default: 1)')
    parser.add_argument('--video-bframes', type=int, default=None,
                        help='NVENC B frames; default: 0 for GOP 1, otherwise min(3, GOP-2)')
    parser.add_argument('--video-qp', type=int, default=21,
                        help='NVENC constant QP (0-51); default: 21')
    parser.add_argument('--video-preset', choices=tuple(f'p{i}' for i in range(1, 8)), default='p6',
                        help='NVENC preset; default: p6')
    parser.add_argument('--lighting-attempts', type=int, default=None,
                        help='override the configured number of lighting samples per seed')
    args = parser.parse_args()
    if args.pick_color == args.place_color:
        parser.error('--pick-color and --place-color must differ')
    if (args.episodes < 1 or args.max_attempts < 1 or args.max_steps < 1
            or args.warmup_frames < 0):
        parser.error('episode, attempt, and step limits must be positive; warm-up frames must be nonnegative')
    if args.lighting_attempts is not None and args.lighting_attempts < 1:
        parser.error('--lighting-attempts must be positive')
    if args.video_gop is not None and args.video_gop < 1:
        parser.error('--video-gop must be positive')
    if args.video_bframes is not None and args.video_bframes < 0:
        parser.error('--video-bframes must be nonnegative')
    if args.video_qp is not None and not 0 <= args.video_qp <= 51:
        parser.error('--video-qp must be between 0 and 51')
    if args.video_preset is not None and args.video_preset not in tuple(f'p{i}' for i in range(1, 8)):
        parser.error('--video-preset must be one of p1, p2, p3, p4, p5, p6, p7')
    if args.save_video and run_scene.DATASET_VIDEO_CODEC != 'h264_nvenc':
        parser.error('video encoder options currently require h264_nvenc')
    if args.save_video and args.video_gop > 1 and args.video_bframes is not None:
        if args.video_gop <= args.video_bframes + 1:
            parser.error('--video-gop must be greater than --video-bframes + 1')
    return args


def absolute_path(path):
    return path if path.is_absolute() else ROOT / path


def rgb_valid(image, name, expected_shape=None):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise RecordingError(f'{name} must be a HxWx3 numpy array')
    if image.dtype != np.uint8:
        raise RecordingError(f'{name} must have uint8 dtype, got {image.dtype}')
    if expected_shape is not None and image.shape != expected_shape:
        raise RecordingError(f'{name} image shape changed from {expected_shape} to {image.shape}')
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise RetryEpisode(f'{name} image resolution too small: {image.shape}')
    if not np.isfinite(image).all() or float(image.std()) < 3 or float(image.mean()) < 2:
        raise RetryEpisode(f'{name} image appears blank or unready')


class CollectionRecorder:
    """Persistent image-only LeRobot writer with an uncommitted per-episode buffer."""

    def __init__(self, output_path, initial_images, save_video=False, video_options=None):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError('Install requirements-sim.txt to collect LeRobot data') from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and any(output_path.iterdir()):
            raise FileExistsError(
                f'{output_path} already contains data; refusing to overwrite it. '
                'Choose a new --output or move the existing dataset first.'
            )
        self.dataset_path = output_path
        self.expected_images = {name: tuple(image.shape) for name, image in initial_images.items()}
        if set(self.expected_images) != set(run_scene.CAMERA_NAMES):
            raise RecordingError(f'expected cameras {run_scene.CAMERA_NAMES}, got {tuple(self.expected_images)}')
        image_features = {
            f'observation.images.{CAMERA_FEATURE_NAMES[name]}': {'dtype': 'video' if save_video else 'image', 'shape': shape,
                                           'names': ['height', 'width', 'channels']}
            for name, shape in self.expected_images.items()
        }
        features = {
            'observation.state': {'dtype': 'float32', 'shape': (STATE_SIZE,), 'names': EE_FEATURE_NAMES},
            'action': {'dtype': 'float32', 'shape': (ACTION_SIZE,), 'names': EE_FEATURE_NAMES},
            **image_features,
        }
        try:
            if video_options is not None:
                from lerobot_video_encoder import ConfigurableVideoDataset
                dataset_type = ConfigurableVideoDataset
            else:
                dataset_type = LeRobotDataset
            create_kwargs = dict(
                repo_id=f'local/{output_path.name}', root=output_path, fps=run_scene.DATASET_FPS,
                robot_type='ur5e', features=features, use_videos=save_video,
                image_writer_processes=0, image_writer_threads=IMAGE_WRITER_THREADS,
                encoder_threads=VIDEO_ENCODER_THREADS,
            )
            if save_video:
                create_kwargs['vcodec'] = run_scene.DATASET_VIDEO_CODEC
            self.dataset = dataset_type.create(**create_kwargs)
            if video_options is not None:
                (self.dataset.video_gop, self.dataset.video_bframes,
                 self.dataset.video_qp, self.dataset.video_preset) = video_options
        except Exception as exc:
            raise RecordingError(f'could not create LeRobot dataset at {output_path}: {exc}') from exc
        self.frame_count = 0
        self.save_video = save_video
        self.tasks = []
        self.states = []
        self.actions = []
        self.image_shapes = {key: [] for key in self.expected_images}

    def add_frame(self, obs, action, images, task, actual_tcp_pose):
        expected_task = self.current_task
        if task != expected_task:
            raise RecordingError(f'task changed within episode: {task!r} != {expected_task!r}')
        state = np.r_[run_scene.pose_to_xyz_rot6d(actual_tcp_pose),
                      np.clip(np.mean(obs['gripper_qpos']) / run_scene.GRIPPER_DRIVER_POSITION_MAX, 0, 1)]
        action_pose = action.get('tcp_pose', actual_tcp_pose)
        action_vec = np.r_[run_scene.pose_to_xyz_rot6d(action_pose),
                           np.clip(action['gripper_target'] / 255.0, 0, 1)]
        if state.shape != (STATE_SIZE,) or action_vec.shape != (ACTION_SIZE,):
            raise RecordingError(f'invalid state/action shape: {state.shape}, {action_vec.shape}')
        if not np.isfinite(state).all() or not np.isfinite(action_vec).all():
            raise RecordingError('state or action contains NaN/Inf')
        if set(images) != set(self.expected_images):
            raise RecordingError(f'camera keys changed: {tuple(images)}')
        frame = {'observation.state': state.astype(np.float32), 'action': action_vec.astype(np.float32),
                 'task': task}
        for name, image in images.items():
            rgb_valid(image, name, self.expected_images[name])
            frame[f'observation.images.{CAMERA_FEATURE_NAMES[name]}'] = image
            self.image_shapes[name].append(tuple(image.shape))
        try:
            self.dataset.add_frame(frame)
        except Exception as exc:
            raise RecordingError(f'LeRobot rejected frame {self.frame_count}: {exc}') from exc
        self.states.append(state)
        self.actions.append(action_vec)
        self.tasks.append(task)
        self.frame_count += 1

    def begin_episode(self, task):
        self.current_task = task
        self.tasks.clear()
        self.states.clear()
        self.actions.clear()
        self.image_shapes = {key: [] for key in self.expected_images}
        self.frame_count = 0

    def validate(self, expected_task, stack_success):
        if not stack_success:
            raise RetryEpisode('final stack verification did not pass')
        if self.frame_count < 1:
            raise RetryEpisode('episode contains no frames')
        lengths = [self.frame_count, len(self.tasks), len(self.states), len(self.actions)]
        lengths.extend(len(values) for values in self.image_shapes.values())
        if len(set(lengths)) != 1:
            raise RetryEpisode(f'episode field lengths differ: {lengths}')
        states, actions = np.asarray(self.states), np.asarray(self.actions)
        if states.shape != (self.frame_count, STATE_SIZE) or actions.shape != (self.frame_count, ACTION_SIZE):
            raise RetryEpisode(f'invalid state/action arrays: {states.shape}, {actions.shape}')
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise RetryEpisode('state/action contains non-finite values')
        if self.tasks != [expected_task] * self.frame_count:
            raise RetryEpisode('task text is inconsistent within the episode')
        if any(len(set(shapes)) != 1 for shapes in self.image_shapes.values()):
            raise RetryEpisode('camera image dimensions changed during the episode')

    def commit(self):
        try:
            self.dataset.save_episode(parallel_encoding=False)
        except Exception as exc:
            raise RecordingError(f'LeRobot save_episode failed: {exc}') from exc
        count = self.frame_count
        self.begin_episode(self.current_task)
        return count

    def discard(self):
        try:
            if (self.dataset.episode_buffer is not None
                    and 'episode_index' in self.dataset.episode_buffer
                    and (self.dataset.episode_buffer.get('size', 0) or self.frame_count)):
                self.dataset.clear_episode_buffer(delete_images=True)
        except Exception as exc:
            raise RecordingError(f'could not discard failed episode buffer: {exc}') from exc
        self.begin_episode(self.current_task)

    def finalize(self):
        try:
            self.dataset.finalize()
        except Exception as exc:
            raise RecordingError(f'LeRobot finalize failed: {exc}') from exc


def task_text(pick, place):
    return f"stack the {pick.replace('_', ' ')} block on top of the {place.replace('_', ' ')} block"


def validate_trajectory_completion(metadata):
    """Require successful collected episodes to finish with the post-place lift."""
    stages = metadata.get('stages') or []
    if not stages or stages[-1].get('phase') != 'lift_after_place':
        raise RetryEpisode('trajectory did not finish with lift_after_place')
    if not metadata.get('success_criteria', {}).get('arm_lifted', False):
        raise RetryEpisode('final robot posture was not lifted')


def select_lighting_env(seed, args, config, lighting_attempts):
    spec = config['lighting_randomization']
    rejected = []
    for lighting_index in range(lighting_attempts):
        sampled_config = sample_lighting(config, seed, lighting_index)
        build(sampled_config)
        env = StackColorBlocksEnv(render=True, headless=True, render_hz=0,
                                  config=sampled_config, ssgi_enabled=True)
        try:
            if not env.ssgi_enabled:
                raise RuntimeError('runtime SSGI failed to enable')
            env.reset(seed, randomize_background=args.randomize_background, randomize_blocks=True)
            home_images = capture_ready(env)
            home_pass, home_metrics = image_quality(home_images, spec['quality'])
            _, _, initial_qpos, _ = run_scene.sample_initial_pose(env, seed)
            for name, value in zip(ARM_JOINTS, initial_qpos):
                env.model.get_joint(name).set_dof_pos(env.data, np.asarray([value], dtype=float))
            env.model.forward_kinematic(env.data)
            initial_images = capture_ready(env)
            initial_pass, initial_metrics = image_quality(initial_images, spec['quality'])
            if home_pass and initial_pass:
                env.lighting_validation = {'attempt': lighting_index, 'rejected_attempts': rejected,
                                           'camera_metrics': {'home': home_metrics,
                                                              'sampled_initial': initial_metrics}}
                return env, sampled_config
            rejected.append({'attempt': lighting_index,
                             'camera_metrics': {'home': home_metrics, 'sampled_initial': initial_metrics}})
        except Exception:
            env.close()
            raise
        env.close()
    return None, {'rejected_lighting_samples': rejected}


def append_jsonl(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def verify_dataset(path, expected_counts, expect_video=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image
    import io
    dataset = LeRobotDataset(repo_id=f'local/{path.name}', root=path)
    if bool(dataset.meta.video_keys) != expect_video:
        raise RuntimeError(f'expected video features={expect_video}, got {dataset.meta.video_keys}')
    for camera in run_scene.CAMERA_NAMES:
        feature = dataset.meta.features.get(f'observation.images.{CAMERA_FEATURE_NAMES[camera]}')
        expected_dtype = 'video' if expect_video else 'image'
        if feature is None or feature['dtype'] != expected_dtype:
            raise RuntimeError(f'{camera} expected LeRobot {expected_dtype} feature: {feature}')
    if dataset.meta.total_episodes != sum(expected_counts.values()):
        raise RuntimeError(f'expected {sum(expected_counts.values())} episodes, got {dataset.meta.total_episodes}')
    expected_pairs = tuple(expected_counts)
    counts = {pair: 0 for pair in expected_pairs}
    tasks = set()
    frame_task_indices = dataset.hf_dataset['task_index']
    for index in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[index]
        frame_start = int(ep['dataset_from_index'])
        frame_end = int(ep['dataset_to_index'])
        if frame_end <= frame_start:
            raise RuntimeError(f'empty episode {index}')
        row = dataset[frame_start]
        task_text_value = row['task']
        tasks.add(task_text_value)
        pair = next((pair for pair in expected_pairs if task_text(*pair) == task_text_value), None)
        if pair is None:
            raise RuntimeError(f'unknown task text in episode {index}: {task_text_value!r}')
        expected_task_index = int(dataset.meta.tasks.loc[task_text_value, 'task_index'])
        episode_task_indices = {int(frame_task_indices[frame].item())
                                for frame in range(frame_start, frame_end)}
        if episode_task_indices != {expected_task_index}:
            raise RuntimeError(f'task mapping changed inside episode {index}: {episode_task_indices}')
        counts[pair] += 1
        for frame_index in (frame_start, frame_end - 1):
            sample = dataset[frame_index]
            for camera in run_scene.CAMERA_NAMES:
                key = f'observation.images.{CAMERA_FEATURE_NAMES[camera]}'
                image = sample[key]
                array = image.detach().cpu().numpy() if hasattr(image, 'detach') else np.asarray(image)
                if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                    array = np.moveaxis(array, 0, -1)
                if array.ndim != 3 or array.shape[-1] != 3:
                    raise RuntimeError(f'image decode shape invalid for {key}: {array.shape}')
                if np.issubdtype(array.dtype, np.floating):
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                elif array.dtype != np.uint8:
                    raise RuntimeError(f'image decode dtype invalid for {key}: {array.dtype}')
                Image.fromarray(array).save(io.BytesIO(), format='PNG')
        if tuple(np.asarray(row['observation.state']).shape) != (STATE_SIZE,):
            raise RuntimeError(f'bad state shape in episode {index}')
    if counts != expected_counts:
        raise RuntimeError(f'episode counts mismatch: {counts}')
    expected_tasks = {task_text(*pair) for pair in expected_pairs}
    if tasks != expected_tasks:
        raise RuntimeError(f'task mapping mismatch: {tasks}')
    forbidden = list(path.rglob('*.mp4')) + list(path.rglob('*.mkv')) + list(path.rglob('*.webm'))
    if forbidden and not expect_video:
        raise RuntimeError(f'video files found: {forbidden[:5]}')
    return {'episodes': dataset.meta.total_episodes,
            'counts': {f'{pick}->{place}': count for (pick, place), count in counts.items()},
            'tasks': sorted(tasks),
            'video_features': list(dataset.meta.video_keys), 'video_files': len(forbidden)}


def main():
    args = parse_args()
    output_path = absolute_path(args.output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f'{output_path} is nonempty; refusing to overwrite valid or partial data')
    attempt_log = output_path.parent / f'{output_path.name}_attempts.jsonl'
    episode_log = output_path.parent / f'{output_path.name}_episodes.jsonl'
    base_config = json.loads((ROOT / 'configs/scene.yaml').read_text())
    base_config['ssgi']['enabled'] = True
    lighting_attempts = args.lighting_attempts or base_config['lighting_randomization']['max_attempts']
    pair = (args.pick_color, args.place_color)
    counts = {pair: 0}
    used_seeds = set()
    attempts = successes = failures = 0
    recorder = None
    try:
        pick, place = pair
        while counts[pair] < args.episodes:
            if attempts >= args.max_attempts:
                raise RuntimeError(f'max attempts reached: {attempts}; counts={counts}')
            seed_rng = np.random.default_rng(np.random.SeedSequence([args.seed, attempts, 0xC010]))
            seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))
            while seed in used_seeds:
                seed = (seed + 1) % np.iinfo(np.int32).max
            used_seeds.add(seed)
            attempts += 1
            prompt = task_text(pick, place)
            log = {'attempt': attempts, 'pick_color': pick, 'place_color': place,
                   'task': prompt, 'seed': seed, 'result': 'pending'}
            try:
                env, lighting_result = select_lighting_env(seed, args, base_config, lighting_attempts)
            except Exception as exc:
                log.update(result='error', failure=f'{type(exc).__name__}: {exc}')
                append_jsonl(attempt_log, log)
                raise
            if env is None:
                log.update(result='retry', failure='lighting camera checks failed',
                           lighting_validation=lighting_result)
                append_jsonl(attempt_log, log)
                failures += 1
                continue
            try:
                xml_text = (ROOT / 'assets/stack_color_blocks.xml').read_text()
                if '<ssgi active="true"' not in xml_text or not env.ssgi_enabled:
                    raise RecordingError('SSGI must be active in generated MJCF and runtime')
                initial_images = env.capture()
                if recorder is None:
                    video_options = ((args.video_gop, args.video_bframes,
                                      args.video_qp, args.video_preset)
                                     if args.save_video else None)
                    recorder = CollectionRecorder(output_path, initial_images,
                                                  save_video=args.save_video,
                                                  video_options=video_options)
                recorder.begin_episode(prompt)
                run_args = SimpleNamespace(
                    headless=True, task='color_block_stack', pick_color=pick, place_color=place,
                    output=output_path, seed=seed, max_steps=args.max_steps,
                    warmup_frames=args.warmup_frames,
                    randomize_background=args.randomize_background, randomize_lighting=True,
                    ssgi=True, fixed_blocks=False,
                )
                metadata = run_scene.run_stack(env, run_args, output_path, recorder=recorder,
                                               write_diagnostics=False, raise_on_failure=False)
                try:
                    if not metadata['success']:
                        raise RetryEpisode(metadata.get('failure') or 'trajectory failed')
                    validate_trajectory_completion(metadata)
                    recorder.validate(prompt, True)
                except RetryEpisode as exc:
                    recorder.discard()
                    log.update(result='retry', failure=str(exc), steps=metadata.get('steps_used'),
                               lighting=env.config.get('lighting_sample'),
                               initial_blocks=metadata.get('initial_blocks'),
                               initial_end_effector_pose=metadata.get('initial_end_effector_pose'))
                    append_jsonl(attempt_log, log)
                    failures += 1
                    continue
                saved_frames = recorder.commit()
                counts[pair] += 1
                successes += 1
                log.update(result='success', episode_index=recorder.dataset.meta.total_episodes - 1,
                           frames=saved_frames, steps=metadata.get('steps_used'),
                           lighting=env.config.get('lighting_sample'),
                           lighting_validation=env.lighting_validation,
                           ssgi={'enabled': env.ssgi_enabled, 'mjcf_active': True,
                                 'parameters': env.config['ssgi']},
                           initial_blocks=metadata.get('initial_blocks'),
                           initial_end_effector_pose=metadata.get('initial_end_effector_pose'),
                           initial_arm_qpos=metadata.get('initial_arm_qpos'))
                append_jsonl(attempt_log, log)
                append_jsonl(episode_log, log)
                print(f"SAVED {counts[pair]}/{args.episodes} {pick}->{place} "
                      f"seed={seed} frames={saved_frames}", flush=True)
            except RecordingError as exc:
                log.update(result='error', failure=f'{type(exc).__name__}: {exc}',
                           lighting=env.config.get('lighting_sample'),
                           lighting_validation=env.lighting_validation)
                append_jsonl(attempt_log, log)
                if recorder is not None:
                    recorder.discard()
                raise
            finally:
                env.close()
        if recorder is None:
            raise RuntimeError('no episode was recorded')
    finally:
        if recorder is not None:
            recorder.finalize()
    expected = {pair: args.episodes}
    result = verify_dataset(output_path, expected, expect_video=args.save_video)
    result.update(attempts=attempts, successes=successes, failures=failures,
                  output=str(output_path), seed=args.seed)
    (output_path.parent / f'{output_path.name}_verification.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print('VERIFIED', json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
