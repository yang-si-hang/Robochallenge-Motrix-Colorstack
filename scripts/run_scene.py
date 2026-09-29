"""Run a fixed-start color-block stacking trajectory and save recordings."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from stack_color_blocks import StackColorBlocksEnv
from stack_color_blocks.control import ARM_JOINTS, HOME_QPOS, MAX_SPEED
from stack_color_blocks.ik import END_EFFECTOR_OFFSET, IKSolver, solve_ik
from motrixsim.ik import IkChain
from build_scene import build
from lighting_randomization import capture_ready, image_quality, sample_lighting

CAMERA_NAMES = ('global_rgb', 'wrist_rgb')
DATASET_FPS = 20
DATASET_VIDEO_CODEC = 'h264_nvenc'
DATASET_VIDEO_GOP = 1
DATASET_VIDEO_BFRAMES = None
DATASET_VIDEO_QP = 21
DATASET_VIDEO_PRESET = 'p6'
DATASET_VIDEO_ENCODER_THREADS = 8
DATASET_IMAGE_WRITER_THREADS = 8
GUI_RENDER_HZ = 30
TRACE_INTERVAL = 10
POSITION_REACHED_TOLERANCE = 0.015
ORIENTATION_REACHED_TOLERANCE = 0.035
TRAVEL_HEIGHT_ABOVE_TABLE_M = 0.2
APPROACH_HEIGHT_ABOVE_BLOCK_M = 0.05
STACK_RELEASE_GAP_M = 0.003
GRASP_TCP_OFFSET_ABOVE_BLOCK_CENTER_M = 0.01  # Avoid driving the TCP/finger assembly through the tabletop.
GRIPPER_DRIVER_POSITION_MAX = 0.8
GRIPPER_STABLE_SECONDS = 0.35
STAGE_TIMEOUT_SECONDS = 16.0
TRAJECTORY_SAMPLE_SECONDS = 0.02
TCP_TRANSLATION_SPEED_MPS = 0.12
TCP_ROTATION_SPEED_RADPS = 0.65
# Fixed TCP pose relative to the tabletop frame: XY origin at table center, Z origin at its top.
INITIAL_EE_POSITION_RELATIVE_TO_TABLE = (-0.1, 0.1, 0.2)
# Quaternion order is xyzw: 180 degrees about the global X axis.
INITIAL_EE_QUATERNION = (0.7071, -0.7071, 0.0, 0.0)
COLORS = ('yellow', 'orange', 'sky_blue')
POSE_FEATURE_NAMES = ('x', 'y', 'z', 'rot6d_0', 'rot6d_1', 'rot6d_2',
                      'rot6d_3', 'rot6d_4', 'rot6d_5')


def json_default(value):
    """Convert NumPy values to standard JSON types for run diagnostics."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def pose_to_xyz_rot6d(pose):
    """Convert [xyz, xyzw] into xyz plus the first two rotation-matrix columns."""
    pose = np.asarray(pose, dtype=float)
    x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    rot6d = rotation[:, :2].T.reshape(6)
    return np.r_[pose[:3], rot6d]


def forward_kinematic_pose(ik_solver, arm_qpos):
    """Get the nominal TCP pose for a joint command using the IK model."""
    ik_solver.data.reset(ik_solver.model)
    for name, value in zip(ARM_JOINTS, arm_qpos):
        ik_solver.model.get_joint(name).set_dof_pos(ik_solver.data, np.asarray([value], dtype=float))
    ik_solver.model.forward_kinematic(ik_solver.data)
    return np.asarray(ik_solver.chain.get_end_effector_pose(ik_solver.data), dtype=float).copy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    display_mode = parser.add_mutually_exclusive_group(required=True)
    display_mode.add_argument('--headless', action='store_true')
    display_mode.add_argument('--gui', action='store_true')
    parser.add_argument('--task', choices=('color_block_stack',), default='color_block_stack')
    parser.add_argument('--pick-color', choices=COLORS, required=True)
    parser.add_argument('--place-color', choices=COLORS, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/stack_color_blocks'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=90000,
                        help='total simulation step limit for the full task')
    parser.add_argument('--warmup-frames', type=int, default=10,
                        help='simulation-rate dataset frames to hold at the initial pose before recording')
    parser.add_argument('--randomize-background', action='store_true')
    parser.add_argument('--randomize-lighting', action='store_true',
                        help='sample one lighting setup per episode and validate both RGB cameras')
    ssgi_group = parser.add_mutually_exclusive_group()
    ssgi_group.add_argument('--ssgi', dest='ssgi', action='store_true', help='enable screen-space global illumination')
    ssgi_group.add_argument('--no-ssgi', dest='ssgi', action='store_false', help='disable screen-space global illumination')
    parser.set_defaults(ssgi=None)
    parser.add_argument('--fixed-blocks', action='store_true',
                        help='use configured fixed block positions and zero yaw')
    args = parser.parse_args()
    if args.pick_color == args.place_color:
        parser.error('--pick-color and --place-color must be different')
    if args.max_steps < 1 or args.warmup_frames < 0:
        parser.error('--max-steps must be positive and --warmup-frames nonnegative')
    return args


def prepare_output_dir(output_arg):
    output_dir = output_arg if output_arg.is_absolute() else ROOT / output_arg
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class LeRobotRecorder:
    def __init__(self, output_dir, initial_images):
        try:
            from lerobot_video_encoder import ConfigurableVideoDataset
        except ImportError as exc:
            raise RuntimeError('Headless recording requires LeRobot; install requirements-sim.txt') from exc
        image_features = {
            f'observation.images.{name}': {
                'dtype': 'video', 'shape': tuple(image.shape),
                'names': ['height', 'width', 'channels'],
            }
            for name, image in initial_images.items()
        }
        state_names = [*POSE_FEATURE_NAMES, 'gripper_position']
        action_names = [*POSE_FEATURE_NAMES, 'gripper_command']
        features = {
            'observation.state': {'dtype': 'float32', 'shape': (len(state_names),),
                                  'names': state_names},
            'action': {'dtype': 'float32', 'shape': (len(action_names),),
                       'names': action_names},
            **image_features,
        }
        self.dataset_path = output_dir / 'lerobot_dataset'
        suffix = 1
        while self.dataset_path.exists() and any(self.dataset_path.iterdir()):
            self.dataset_path = output_dir / f'lerobot_dataset_{suffix:03d}'
            suffix += 1
        self.dataset = ConfigurableVideoDataset.create(
            repo_id='local/stack_color_blocks', root=self.dataset_path,
            fps=DATASET_FPS, robot_type='ur5e', features=features, use_videos=True,
            vcodec=DATASET_VIDEO_CODEC, image_writer_processes=0,
            image_writer_threads=DATASET_IMAGE_WRITER_THREADS,
            encoder_threads=DATASET_VIDEO_ENCODER_THREADS,
        )
        (self.dataset.video_gop, self.dataset.video_bframes,
         self.dataset.video_qp, self.dataset.video_preset) = (
            DATASET_VIDEO_GOP, DATASET_VIDEO_BFRAMES,
            DATASET_VIDEO_QP, DATASET_VIDEO_PRESET)
        self.frame_count = 0

    def add_frame(self, obs, action, images, task, actual_tcp_pose):
        gripper_position = np.clip(np.mean(obs['gripper_qpos']) / GRIPPER_DRIVER_POSITION_MAX, 0.0, 1.0)
        state = np.r_[pose_to_xyz_rot6d(actual_tcp_pose), gripper_position].astype(np.float32)
        action_pose = action.get('tcp_pose', actual_tcp_pose)
        gripper_command = np.clip(action['gripper_target'] / 255.0, 0.0, 1.0)
        action_vector = np.r_[pose_to_xyz_rot6d(action_pose), gripper_command].astype(np.float32)
        frame = {'observation.state': state, 'action': action_vector, 'task': task}
        frame.update({f'observation.images.{name}': image for name, image in images.items()})
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def close_episode(self):
        if self.frame_count:
            # A forked encoder worker cannot reliably initialize CUDA after rendering.
            self.dataset.save_episode(parallel_encoding=False)
            self.dataset.finalize()


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])


def quat_error(a, b):
    dot = float(np.clip(abs(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))), 0, 1))
    return 2.0 * float(np.arccos(dot))


def slerp_quaternion(a, b, amount):
    a = np.asarray(a, dtype=float); a /= np.linalg.norm(a)
    b = np.asarray(b, dtype=float); b /= np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b; dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = a + amount*(b-a)
        return q / np.linalg.norm(q)
    angle = np.arccos(dot)
    return (np.sin((1.0-amount)*angle)*a + np.sin(amount*angle)*b) / np.sin(angle)


def smoothstep5(amount):
    amount = float(np.clip(amount, 0.0, 1.0))
    return amount**3 * (10.0 + amount*(-15.0 + 6.0*amount))


def sample_path(values, progress):
    """Linearly sample precomputed points whose time parameter is quintic-smoothed."""
    values = np.asarray(values)
    coordinate = float(np.clip(progress, 0.0, 1.0)) * (len(values)-1)
    lo = min(int(coordinate), len(values)-1)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (coordinate-lo)*(values[hi]-values[lo])


def plan_cartesian_trajectory(start_pose, target_pose, initial_qpos, ik_solver):
    """Quintic Cartesian translation/SLERP orientation with a continuous IK branch."""
    start_pose = np.asarray(start_pose, dtype=float).copy()
    target_pose = np.asarray(target_pose, dtype=float).copy()
    position_distance = float(np.linalg.norm(target_pose[:3]-start_pose[:3]))
    rotation_distance = quat_error(start_pose[3:], target_pose[3:])
    duration = max(1.875*position_distance/TCP_TRANSLATION_SPEED_MPS,
                   1.875*rotation_distance/TCP_ROTATION_SPEED_RADPS, 0.5)
    sample_count = max(1, int(np.ceil(duration/TRAJECTORY_SAMPLE_SECONDS)))
    poses = [start_pose]
    joints = [np.asarray(initial_qpos, dtype=float).copy()]
    seed = joints[0]
    final_result = None
    for index in range(1, sample_count+1):
        progress = smoothstep5(index/sample_count)
        pose = np.r_[start_pose[:3] + progress*(target_pose[:3]-start_pose[:3]),
                     slerp_quaternion(start_pose[3:], target_pose[3:], progress)]
        result = ik_solver.solve(pose, seed)
        final_result = result
        seed = np.asarray(result.joint_angles, dtype=float)
        poses.append(pose)
        joints.append(seed.copy())
    poses = np.asarray(poses)
    joints = np.asarray(joints)
    joint_path_length = np.sum(np.abs(np.diff(joints, axis=0)), axis=0)
    joint_limited_duration = 1.875 * float(np.max(joint_path_length)) / MAX_SPEED
    duration = max(duration, joint_limited_duration)
    return poses, joints, duration, final_result


def yaw_from_pose(pose):
    q = pose[3:] / np.linalg.norm(pose[3:])
    return 2.0 * float(np.arctan2(q[2], q[3]))


def tilt_from_pose(pose):
    q = pose[3:] / np.linalg.norm(pose[3:])
    return float(np.arccos(np.clip(1.0 - 2.0*(q[0]**2 + q[1]**2), -1.0, 1.0)))


def aligned_down_quaternion(block_pose, reference_quaternion):
    """Downward baseline composed with the block's world yaw; square symmetry allows 90° aliases."""
    yaw = yaw_from_pose(block_pose)
    down = np.asarray(INITIAL_EE_QUATERNION, dtype=float)
    candidates = []
    for quarter_turn in range(4):
        angle = yaw + quarter_turn * np.pi / 2
        world_z = np.array([0.0, 0.0, np.sin(angle/2), np.cos(angle/2)])
        q = quat_multiply(world_z, down)
        candidates.append(q / np.linalg.norm(q))
    return min(candidates, key=lambda q: quat_error(q, reference_quaternion))


def sample_initial_pose(env, seed=None):
    """Solve the fixed tabletop-relative TCP pose into a reachable arm state."""
    table = env.config['table']
    center = np.asarray(table['center'], dtype=float)
    half_size = np.asarray(table['half_size'], dtype=float)
    x, y, z = INITIAL_EE_POSITION_RELATIVE_TO_TABLE
    position = np.array([center[0] + x, center[1] + y, center[2] + half_size[2] + z])
    quaternion = np.asarray(INITIAL_EE_QUATERNION, dtype=float)
    pose = np.r_[position, quaternion / np.linalg.norm(quaternion)]
    chain = IkChain(env.model, end_link='grip_base', start_link='base',
                    end_effector_offset=END_EFFECTOR_OFFSET)
    result = solve_ik(pose, HOME_QPOS)
    yaw_deg = float(np.rad2deg(yaw_from_pose(pose)))
    return chain, pose, np.asarray(result.joint_angles, dtype=float), yaw_deg


def build_stages(env, initial_pose, initial_qpos, pick_color, place_color, seed_value, ik_solver):
    blocks = env.get_observation()['blocks']
    pick = blocks[pick_color].copy()
    place = blocks[place_color].copy()
    cfg = env.config
    table_top = cfg['table']['center'][2] + cfg['table']['half_size'][2]
    half = cfg['block']['half_size']
    travel_z = table_top + TRAVEL_HEIGHT_ABOVE_TABLE_M
    approach_pick_z = pick[2] + half + APPROACH_HEIGHT_ABOVE_BLOCK_M
    approach_place_z = place[2] + half + APPROACH_HEIGHT_ABOVE_BLOCK_M
    pick_q = aligned_down_quaternion(pick, initial_pose[3:])
    place_q = aligned_down_quaternion(place, pick_q)
    pick_tcp_z = pick[2] + GRASP_TCP_OFFSET_ABOVE_BLOCK_CENTER_M
    release_tcp_z = place[2] + 2*half + STACK_RELEASE_GAP_M + GRASP_TCP_OFFSET_ABOVE_BLOCK_CENTER_M

    def pose(x, y, z, q): return np.r_[x, y, z, q]
    stages = [
        ('move_above_pick', pose(pick[0], pick[1], initial_pose[2], initial_pose[3:]), 0.0, 'pose'),
        ('approach_pick', pose(pick[0], pick[1], approach_pick_z, initial_pose[3:]), 0.0, 'pose'),
        ('align_pick', pose(pick[0], pick[1], approach_pick_z, pick_q), 0.0, 'pose'),
        ('descend_pick', pose(pick[0], pick[1], pick_tcp_z, pick_q), 0.0, 'pose'),
        ('close_gripper', None, 255.0, 'hold'),
        ('lift_pick', pose(pick[0], pick[1], travel_z, pick_q), 255.0, 'pose'),
        ('move_above_place', pose(place[0], place[1], travel_z, pick_q), 255.0, 'pose'),
        ('approach_place', pose(place[0], place[1], approach_place_z, pick_q), 255.0, 'pose'),
        ('align_place', pose(place[0], place[1], approach_place_z, place_q), 255.0, 'pose'),
        ('descend_place', pose(place[0], place[1], release_tcp_z, place_q), 255.0, 'pose'),
        ('open_gripper', None, 0.0, 'hold'),
        ('lift_after_place', pose(place[0], place[1], travel_z, place_q), 0.0, 'pose'),
    ]
    # Solve the complete nominal route before actuating the robot.
    solved = []
    seed = np.asarray(initial_qpos, dtype=float)
    arm_goal = seed.copy()
    for name, target_pose, grip, kind in stages:
        if kind == 'pose':
            try:
                result = ik_solver.solve(target_pose, seed)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(f'IK preflight failed at stage {name}, seed {seed_value}, '
                                   f'{pick_color}->{place_color}: {exc}') from exc
            arm_goal = np.asarray(result.joint_angles, dtype=float)
            seed = arm_goal
            ik = {'joint_angles_rad': arm_goal.tolist(),
                  'position_error_m': result.position_error,
                  'orientation_error_rad': result.orientation_error,
                  'iterations': result.iterations}
        else:
            ik = None
        solved.append({'name': name, 'target_pose': None if target_pose is None else target_pose.copy(),
                       'gripper_target': grip, 'arm_target': arm_goal.copy(), 'kind': kind, 'ik': ik})
    return solved, pick, place


def metadata_base(env, args, initial):
    return {
        'motrixsim': importlib.metadata.version('motrixsim'),
        'python': platform.python_version(),
        'asset_commit': 'c96a32d28fb5da84da38c1da4d749e7a13212855',
        'config': env.config,
        'task': args.task,
        'pick_color': args.pick_color,
        'place_color': args.place_color,
        'seed': args.seed,
        'randomize_background': args.randomize_background,
        'randomize_lighting': args.randomize_lighting,
        'ssgi': {'enabled': env.ssgi_enabled,
                 'mjcf_active': True,
                 'parameters': {k: env.config['ssgi'][k] for k in (
                     'resolutionscale', 'raycount', 'stepcount', 'thickness', 'intensity',
                     'gidenoiseoffset', 'rraycount', 'rdenoiseoffset')}},
        'lighting_validation': getattr(env, 'lighting_validation', None),
        'randomize_blocks': not args.fixed_blocks,
        'fixed_blocks': args.fixed_blocks,
        'travel_height_above_table_m': TRAVEL_HEIGHT_ABOVE_TABLE_M,
        'approach_height_above_block_m': APPROACH_HEIGHT_ABOVE_BLOCK_M,
        'stack_release_gap_m': STACK_RELEASE_GAP_M,
        'grasp_tcp_offset_above_block_center_m': GRASP_TCP_OFFSET_ABOVE_BLOCK_CENTER_M,
        'position_reached_tolerance_m': POSITION_REACHED_TOLERANCE,
        'descent_position_tolerance_m': 0.01,
        'descend_place_position_tolerance_m': POSITION_REACHED_TOLERANCE,
        'orientation_reached_tolerance_rad': ORIENTATION_REACHED_TOLERANCE,
        'stage_timeout_seconds': STAGE_TIMEOUT_SECONDS,
        'gripper_stable_seconds': GRIPPER_STABLE_SECONDS,
        'home_qpos': list(HOME_QPOS),
        'robot_base_pose': env.model.get_body('base').get_pose(env.data).tolist(),
        'initial_end_effector_pose': initial['initial_end_effector_pose'],
        'initial_requested_end_effector_pose': initial['initial_requested_end_effector_pose'],
        'initial_ee_yaw_deg': initial['initial_ee_yaw_deg'],
        'initial_pose_source': 'script_global_constants',
        'initial_ee_position_relative_to_table_m': list(INITIAL_EE_POSITION_RELATIVE_TO_TABLE),
        'initial_ee_quaternion_xyzw': list(INITIAL_EE_QUATERNION),
        'initial_arm_qpos': initial['initial_arm_qpos'],
        'initial_blocks': {n: p.tolist() for n, p in initial['blocks'].items()},
        'initial_background': {n: p.tolist() for n, p in initial['background'].items()},
        'max_steps': args.max_steps,
        'warmup_frames': getattr(args, 'warmup_frames', 0),
        'cameras': list(CAMERA_NAMES),
        'trace': 'trace.json',
    }


def run_stack(env, args, output_dir, recorder=None, write_diagnostics=True, raise_on_failure=True):
    initial = env.reset(args.seed, randomize_background=args.randomize_background,
                        randomize_blocks=not args.fixed_blocks)
    chain, initial_pose, initial_qpos, initial_ee_yaw_deg = sample_initial_pose(env, args.seed)
    ik_solver = IKSolver()
    for name, value in zip(ARM_JOINTS, initial_qpos):
        env.model.get_joint(name).set_dof_pos(env.data, np.asarray([value], dtype=float))
    env.model.forward_kinematic(env.data)
    env.controller.command = initial_qpos.copy()
    initial['initial_end_effector_pose'] = chain.get_end_effector_pose(env.data).tolist()
    initial['initial_requested_end_effector_pose'] = initial_pose.tolist()
    initial['initial_arm_qpos'] = initial_qpos.tolist()
    initial['initial_ee_yaw_deg'] = initial_ee_yaw_deg
    initial['blocks'] = env.get_observation()['blocks']
    try:
        stages, pick_start, place_start = build_stages(env, initial['initial_end_effector_pose'],
                                                      initial_qpos, args.pick_color, args.place_color,
                                                      args.seed, ik_solver)
    except Exception as exc:
        trace = [{'step': 0, 'time_s': env.time, 'phase': 'ik_preflight', 'failure': str(exc),
                  'seed': args.seed, 'pick_color': args.pick_color, 'place_color': args.place_color}]
        (output_dir / 'run.json').write_text(json.dumps({
            'task': args.task, 'seed': args.seed, 'pick_color': args.pick_color,
            'place_color': args.place_color, 'success': False, 'failure': str(exc),
            'ssgi': {'enabled': env.ssgi_enabled, 'mjcf_active': True,
                     'parameters': {k: env.config['ssgi'][k] for k in (
                         'resolutionscale', 'raycount', 'stepcount', 'thickness', 'intensity',
                         'gidenoiseoffset', 'rraycount', 'rdenoiseoffset')}},
            'initial_end_effector_pose': initial['initial_end_effector_pose'],
            'initial_requested_end_effector_pose': initial_pose.tolist(),
            'initial_arm_qpos': initial_qpos.tolist(),
            'initial_blocks': {n: p.tolist() for n, p in initial['blocks'].items()},
        }, indent=2, ensure_ascii=False, default=json_default))
        (output_dir / 'trace.json').write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, default=json_default))
        raise
    sample_interval_steps = int(round(1.0 / (DATASET_FPS * env.dt)))
    if abs(sample_interval_steps * env.dt - 1.0 / DATASET_FPS) > 1e-9:
        raise ValueError(f'{DATASET_FPS} Hz dataset rate must be an integer multiple of simulation dt={env.dt}')
    owns_recorder = recorder is None
    task_prompt = (f"stack the {args.pick_color.replace('_', ' ')} block on top of the "
                   f"{args.place_color.replace('_', ' ')} block")
    trace = []
    stage_records = []
    step_index = 0
    failure = None
    last_obs = env.get_observation()
    carried_offset = None
    carried_xy_offset = None

    def record_dataset_sample(obs, actual_tcp_pose):
        if not args.headless:
            return
        # obs['target'] is the joint command after Controller.apply's speed limit.
        action = {'tcp_pose': forward_kinematic_pose(ik_solver, obs['target']),
                  'gripper_target': obs['gripper_target']}
        images = env.capture()
        recorder.add_frame(obs, action, images, task_prompt, actual_tcp_pose)

    def record(phase, target_pose, actual_pose, obs, pos_error=None, ori_error=None,
               transition_reason=None):
        trace.append({'step': step_index, 'time_s': round(obs['time'], 4), 'phase': phase,
                      'target_pose': None if target_pose is None else np.asarray(target_pose).tolist(),
                      'actual_tcp_pose': np.asarray(actual_pose).tolist(),
                      'position_error_m': pos_error, 'orientation_error_rad': ori_error,
                      'arm_qpos': obs['arm_qpos'].tolist(), 'gripper_qpos': obs['gripper_qpos'].tolist(),
                      'blocks': {n: p.tolist() for n, p in obs['blocks'].items()},
                      'transition_reason': transition_reason})

    try:
        # Let the scene and cameras settle while the arm remains at its sampled
        # initial pose. These steps intentionally do not enter the dataset.
        warmup_frames = int(getattr(args, 'warmup_frames', 0))
        if warmup_frames < 0:
            raise ValueError('warmup_frames must be nonnegative')
        for _ in range(warmup_frames * sample_interval_steps):
            if step_index >= args.max_steps:
                raise RuntimeError(f'total step limit {args.max_steps} reached during warm-up')
            last_obs = env.step({'arm_qpos': initial_qpos, 'gripper_target': 0.0})
            step_index += 1
        if args.headless:
            initial_images = env.capture()
            if owns_recorder:
                recorder = LeRobotRecorder(output_dir, initial_images)
        for stage_index, stage in enumerate(stages):
            name = stage['name']
            start_step = step_index
            start_time = env.time
            actual_start_pose = np.asarray(chain.get_end_effector_pose(env.data), dtype=float)
            actual_start_qpos = np.asarray(last_obs['arm_qpos'], dtype=float).copy()
            stage_target = None if stage['target_pose'] is None else stage['target_pose'].copy()
            trajectory_poses = trajectory_joints = None
            trajectory_duration = 0.0
            trajectory_final_ik = None
            if stage['kind'] == 'pose':
                blocks_now = env.get_observation()['blocks']
                pick_pose = blocks_now[args.pick_color]
                place_pose = blocks_now[args.place_color]
                travel_z = (env.config['table']['center'][2] + env.config['table']['half_size'][2]
                            + TRAVEL_HEIGHT_ABOVE_TABLE_M)
                if name == 'move_above_pick':
                    stage_target[:2] = pick_pose[:2]
                    stage_target[2] = initial_pose[2]
                    stage_target[3:] = initial_pose[3:]
                elif name == 'approach_pick':
                    stage_target[:2] = pick_pose[:2]
                    stage_target[2] = pick_pose[2] + env.config['block']['half_size'] + APPROACH_HEIGHT_ABOVE_BLOCK_M
                    stage_target[3:] = actual_start_pose[3:]
                elif name == 'align_pick':
                    stage_target[:3] = actual_start_pose[:3]
                elif name == 'descend_pick':
                    stage_target[:2] = pick_pose[:2]
                    stage_target[2] = pick_pose[2] + GRASP_TCP_OFFSET_ABOVE_BLOCK_CENTER_M
                    stage_target[3:] = actual_start_pose[3:]
                elif name == 'lift_pick':
                    stage_target[:2] = actual_start_pose[:2]
                    stage_target[2] = travel_z
                    stage_target[3:] = actual_start_pose[3:]
                elif name == 'move_above_place':
                    offset = np.zeros(2) if carried_xy_offset is None else carried_xy_offset
                    stage_target[:2] = place_pose[:2] + offset
                    stage_target[2] = actual_start_pose[2]
                    stage_target[3:] = actual_start_pose[3:]
                elif name == 'approach_place':
                    stage_target[:2] = actual_start_pose[:2]
                    stage_target[2] = place_pose[2] + env.config['block']['half_size'] + APPROACH_HEIGHT_ABOVE_BLOCK_M
                    stage_target[3:] = actual_start_pose[3:]
                elif name == 'align_place':
                    stage_target[:3] = actual_start_pose[:3]
                elif name == 'descend_place':
                    measured_offset = float(actual_start_pose[2] - pick_pose[2])
                    measured_xy_offset = actual_start_pose[:2] - pick_pose[:2]
                    carried_offset = measured_offset
                    stage_target[:2] = place_pose[:2] + measured_xy_offset
                    stage_target[2] = (place_pose[2] + 2*env.config['block']['half_size']
                                       + STACK_RELEASE_GAP_M + measured_offset)
                    stage_target[3:] = actual_start_pose[3:]
                elif name == 'lift_after_place':
                    stage_target[:2] = actual_start_pose[:2]
                    stage_target[2] = travel_z
                    stage_target[3:] = actual_start_pose[3:]
                try:
                    trajectory_poses, trajectory_joints, trajectory_duration, trajectory_final_ik = \
                        plan_cartesian_trajectory(actual_start_pose, stage_target, actual_start_qpos, ik_solver)
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(f'Cartesian trajectory IK failed at stage {name}, seed {args.seed}, '
                                       f'{args.pick_color}->{args.place_color}: {exc}') from exc
                stage['arm_target'] = trajectory_joints[-1].copy()
                stage['ik'] = {'joint_angles_rad': stage['arm_target'].tolist(),
                               'position_error_m': trajectory_final_ik.position_error,
                               'orientation_error_rad': trajectory_final_ik.orientation_error,
                               'iterations': trajectory_final_ik.iterations,
                               'interpolation': 'quintic_xyz_slerp',
                               'trajectory_samples': len(trajectory_joints),
                               'trajectory_duration_s': trajectory_duration}
            stable_since = None
            stable_qpos = None
            reached_since = None
            stage_reason = 'target_reached'
            timeout = STAGE_TIMEOUT_SECONDS
            if name == 'open_gripper': timeout = 5.0
            timeout = max(timeout, trajectory_duration + 4.0)
            while step_index < args.max_steps:
                progress = 1.0 if trajectory_duration <= 0 else (env.time-start_time)/trajectory_duration
                arm_command = (sample_path(trajectory_joints, progress) if trajectory_joints is not None
                               else actual_start_qpos)
                action = {'arm_qpos': arm_command, 'gripper_target': stage['gripper_target']}
                last_obs = env.step(action)
                step_index += 1
                actual_pose = np.asarray(chain.get_end_effector_pose(env.data), dtype=float)
                if name in ('move_above_place', 'approach_place', 'align_place', 'descend_place'):
                    carried_pose = last_obs['blocks'][args.pick_color]
                    table_top = env.config['table']['center'][2] + env.config['table']['half_size'][2]
                    if carried_pose[2] < table_top + env.config['block']['half_size'] + 0.015:
                        raise RuntimeError(f'carried block lost during stage {name}, seed {args.seed}, '
                                           f'{args.pick_color}->{args.place_color}; block z={carried_pose[2]:.4f}')
                sample_progress = 1.0 if trajectory_duration <= 0 else (env.time-start_time)/trajectory_duration
                target_pose = (None if trajectory_poses is None else sample_path(trajectory_poses, sample_progress))
                if stage['kind'] == 'pose':
                    position_error = float(np.linalg.norm(actual_pose[:3] - target_pose[:3]))
                    orientation_error = quat_error(actual_pose[3:], target_pose[3:])
                    position_tolerance = (0.01 if name == 'descend_pick' else
                                          POSITION_REACHED_TOLERANCE)
                    at_target = (sample_progress >= 1.0 and position_error <= position_tolerance and
                                 orientation_error <= ORIENTATION_REACHED_TOLERANCE)
                else:
                    position_error = orientation_error = None
                    at_target = True

                if name in ('close_gripper', 'open_gripper'):
                    qpos = last_obs['gripper_qpos']
                    desired_stable = (np.min(qpos) > 0.05) if name == 'close_gripper' else (np.max(qpos) < 0.12)
                    if stable_qpos is not None and np.max(np.abs(qpos - stable_qpos)) < 0.003 and desired_stable:
                        if stable_since is None: stable_since = env.time
                    else:
                        stable_since = env.time if desired_stable else None
                    stable_qpos = qpos.copy()
                    at_target = (stable_since is not None and
                                 env.time - stable_since >= GRIPPER_STABLE_SECONDS)

                if step_index % sample_interval_steps == 0:
                    record_dataset_sample(last_obs, actual_pose)
                if at_target:
                    if reached_since is None: reached_since = env.time
                    if env.time - reached_since >= 0.15 or name in ('close_gripper', 'open_gripper'):
                        break
                else:
                    reached_since = None

                if step_index % TRACE_INTERVAL == 0:
                    record(name, target_pose, actual_pose, last_obs, position_error, orientation_error)
                if env.time - start_time > timeout:
                    raise RuntimeError(f'stage timeout: {name}, seed {args.seed}, '
                                       f'{args.pick_color}->{args.place_color}, position error {position_error}, '
                                       f'orientation error {orientation_error}')
            else:
                raise RuntimeError(f'total step limit {args.max_steps} reached during {name}')

            actual_pose = np.asarray(chain.get_end_effector_pose(env.data), dtype=float)
            if name == 'lift_pick':
                block_pose = last_obs['blocks'][args.pick_color]
                table_top = env.config['table']['center'][2] + env.config['table']['half_size'][2]
                min_lift_z = table_top + env.config['block']['half_size'] + 0.02
                if block_pose[2] < min_lift_z:
                    raise RuntimeError(f'grasp verification failed at stage {name}, seed {args.seed}, '
                                       f'{args.pick_color}->{args.place_color}: carried block z={block_pose[2]:.4f}, '
                                       f'required >= {min_lift_z:.4f}')
                carried_offset = float(actual_pose[2] - block_pose[2])
                carried_xy_offset = actual_pose[:2] - block_pose[:2]
                # Correct the carry path for the measured TCP-to-block XY offset so
                # the object center, rather than the TCP, reaches the support center.
                seed = stage['arm_target'].copy()
                for future in stages[stage_index+1:10]:
                    if future['kind'] != 'pose':
                        continue
                    future['target_pose'][:2] = place_start[:2] + carried_xy_offset
                    result = ik_solver.solve(future['target_pose'], seed)
                    future['arm_target'] = np.asarray(result.joint_angles, dtype=float)
                    future['ik'] = {'joint_angles_rad': list(result.joint_angles),
                                    'position_error_m': result.position_error,
                                    'orientation_error_rad': result.orientation_error,
                                    'iterations': result.iterations,
                                    'recomputed_with_measured_carried_xy_offset_m': carried_xy_offset.tolist()}
                    seed = future['arm_target']
            record(name, target_pose, actual_pose, last_obs, position_error, orientation_error,
                   'stage target and stability condition met')
            stage_records.append({'phase': name, 'start_step': start_step, 'end_step': step_index,
                                 'start_time_s': start_time, 'end_time_s': env.time,
                                 'target_pose': None if target_pose is None else np.asarray(target_pose).tolist(),
                                 'commanded_ik_pose': None if stage_target is None else np.asarray(stage_target).tolist(),
                                 'target_arm_qpos': stage['arm_target'].tolist(),
                                 'ik': stage['ik'], 'final_actual_pose': actual_pose.tolist(),
                                 'final_position_error': position_error,
                                 'final_orientation_error_rad': orientation_error,
                                 'transition_reason': stage_reason})
        # Continue the shared loop briefly after the last phase and verify true support contact and rest.
        rest_start = env.time
        rest_positions = []
        hold_qpos = np.asarray(last_obs['arm_qpos'], dtype=float).copy()
        while env.time - rest_start < 0.5 and step_index < args.max_steps:
            last_obs = env.step({'arm_qpos': hold_qpos, 'gripper_target': 0.0})
            step_index += 1
            rest_positions.append(last_obs['blocks'][args.pick_color][:3].copy())
            if step_index % TRACE_INTERVAL == 0:
                actual_pose = chain.get_end_effector_pose(env.data)
                record('verify_stack', None, actual_pose, last_obs)
            if step_index % sample_interval_steps == 0:
                actual_tcp_pose = np.asarray(chain.get_end_effector_pose(env.data), dtype=float)
                record_dataset_sample(last_obs, actual_tcp_pose)
        final_blocks = last_obs['blocks']
        picked = final_blocks[args.pick_color]
        support = final_blocks[args.place_color]
        xy_error = float(np.linalg.norm(picked[:2] - support[:2]))
        height_delta = float(picked[2] - support[2])
        expected_delta = 2*env.config['block']['half_size']
        picked_tilt = tilt_from_pose(picked)
        support_tilt = tilt_from_pose(support)
        stack_contact = (xy_error <= 0.01 and abs(height_delta - expected_delta) <= 0.008
                         and picked_tilt <= 0.15 and support_tilt <= 0.15)
        final_tcp_pose = np.asarray(chain.get_end_effector_pose(env.data), dtype=float)
        table_top = env.config['table']['center'][2] + env.config['table']['half_size'][2]
        arm_lifted = bool(final_tcp_pose[2] >=
                          table_top + TRAVEL_HEIGHT_ABOVE_TABLE_M - POSITION_REACHED_TOLERANCE)
        gripper_open = bool(np.max(last_obs['gripper_qpos']) < 0.12)
        if not stack_contact:
            raise RuntimeError(f'stack verification failed at stage verify_stack: XY error {xy_error:.5f} m, '
                               f'center height delta {height_delta:.5f} m')
        if not arm_lifted or not gripper_open:
            raise RuntimeError(f'final posture verification failed: arm lifted={arm_lifted}, '
                               f'gripper qpos {last_obs["gripper_qpos"].tolist()}')
        if len(rest_positions) >= 2 and np.linalg.norm(rest_positions[-1] - rest_positions[0]) > 0.006:
            raise RuntimeError('stack verification failed: picked block did not settle')
    except Exception as exc:
        if not owns_recorder and getattr(exc, 'deterministic_recording_error', False):
            raise
        failure = str(exc)
        trace.append({'step': step_index, 'time_s': round(env.time, 4), 'phase': 'failed',
                      'failure': failure, 'seed': args.seed,
                      'pick_color': args.pick_color, 'place_color': args.place_color})
    finally:
        if recorder is not None and owns_recorder:
            recorder.close_episode()
        metadata = metadata_base(env, args, initial)
        metadata.update({'success': failure is None, 'failure': failure, 'steps_used': step_index,
                         'dataset_fps': DATASET_FPS if recorder is not None else None,
                         'dataset_video_codec': DATASET_VIDEO_CODEC if recorder is not None else None,
                         'dataset_video_encoder': ({'gop': DATASET_VIDEO_GOP,
                                                    'bframes': 0 if DATASET_VIDEO_GOP == 1 else DATASET_VIDEO_BFRAMES,
                                                    'qp': DATASET_VIDEO_QP,
                                                    'preset': DATASET_VIDEO_PRESET,
                                                    'threads': DATASET_VIDEO_ENCODER_THREADS}
                                                   if recorder is not None else None),
                         'dataset_frames': recorder.frame_count if recorder is not None else 0,
                         'dataset_path': str(recorder.dataset_path) if recorder is not None else None,
                         'dataset_state_features': ([*POSE_FEATURE_NAMES, 'gripper_position']
                                                    if recorder is not None else None),
                         'dataset_action_features': ([*POSE_FEATURE_NAMES, 'gripper_command']
                                                     if recorder is not None else None),
                         'dataset_gripper_range': [0.0, 1.0] if recorder is not None else None,
                         'stages': stage_records, 'final_actual_blocks': {
                             n: p.tolist() for n, p in last_obs['blocks'].items()},
                         'final_arm_qpos': last_obs['arm_qpos'].tolist(),
                         'final_gripper_qpos': last_obs['gripper_qpos'].tolist(),
                         'carried_tcp_offset_m': carried_offset,
                         'carried_tcp_xy_offset_m': None if carried_xy_offset is None else carried_xy_offset.tolist(),
                         'success_criteria': {'stack_xy_error_m': xy_error if failure is None else None,
                                              'center_height_delta_m': height_delta if failure is None else None,
                                              'picked_block_tilt_rad': picked_tilt if failure is None else None,
                                              'support_block_tilt_rad': support_tilt if failure is None else None,
                                              'stack_contact': stack_contact if failure is None else False,
                                              'arm_lifted': arm_lifted if failure is None else False,
                                              'gripper_open': gripper_open if failure is None else False}})
        if write_diagnostics:
            (output_dir / 'run.json').write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, default=json_default))
            (output_dir / 'trace.json').write_text(
                json.dumps(trace, indent=2, ensure_ascii=False, default=json_default))
    if failure and raise_on_failure:
        raise RuntimeError(failure)
    if failure is None:
        print('PASS color_block_stack', output_dir, args.pick_color, 'on', args.place_color,
              'steps', step_index, 'stack_xy_error_m', xy_error, 'center_height_delta_m', height_delta)
    return metadata


def main():
    args = parse_args()
    output_dir = prepare_output_dir(args.output)
    base_config = json.loads((ROOT / 'configs/scene.yaml').read_text())
    ssgi_enabled = base_config['ssgi']['enabled'] if args.ssgi is None else args.ssgi
    if args.randomize_lighting:
        spec = base_config['lighting_randomization']
        base_config['ssgi']['enabled'] = ssgi_enabled
        rejected = []
        env = None
        for attempt in range(spec['max_attempts']):
            config = sample_lighting(base_config, args.seed, attempt)
            build(config)
            candidate = StackColorBlocksEnv(render=True, headless=args.headless,
                                            render_hz=0 if args.headless else GUI_RENDER_HZ,
                                            config=config, ssgi_enabled=ssgi_enabled)
            try:
                candidate.reset(args.seed, randomize_background=args.randomize_background,
                                randomize_blocks=not args.fixed_blocks)
                home_images = capture_ready(candidate)
                home_passed, home_metrics = image_quality(home_images, spec['quality'])
                _, _, initial_qpos, _ = sample_initial_pose(candidate, args.seed)
                for name, value in zip(ARM_JOINTS, initial_qpos):
                    candidate.model.get_joint(name).set_dof_pos(
                        candidate.data, np.asarray([value], dtype=float))
                candidate.model.forward_kinematic(candidate.data)
                initial_images = capture_ready(candidate)
                initial_passed, initial_metrics = image_quality(initial_images, spec['quality'])
                metrics = {'home': home_metrics, 'sampled_initial': initial_metrics}
                if home_passed and initial_passed:
                    candidate.lighting_validation = {'attempt': attempt,
                                                     'rejected_attempts': rejected,
                                                     'camera_metrics': metrics}
                    env = candidate
                    break
                rejected.append({'attempt': attempt, 'camera_metrics': metrics})
            except Exception:
                candidate.close()
                raise
            candidate.close()
        if env is None:
            raise RuntimeError('No lighting sample passed camera validation: '
                               + json.dumps(rejected, ensure_ascii=False, default=json_default))
    else:
        base_config['ssgi']['enabled'] = ssgi_enabled
        build(base_config)
        env = StackColorBlocksEnv(render=True, headless=args.headless,
                                  render_hz=0 if args.headless else GUI_RENDER_HZ,
                                  config=base_config, ssgi_enabled=ssgi_enabled)
    try:
        run_stack(env, args, output_dir)
    finally:
        env.close()


if __name__ == '__main__':
    main()
