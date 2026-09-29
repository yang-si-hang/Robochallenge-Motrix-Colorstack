"""Observation and action conversions for the OpenPI socket policy."""
from __future__ import annotations

import numpy as np


CAMERA_KEYS = {
    'global_rgb': 'observation.images.base_0_rgb',
    'wrist_rgb': 'observation.images.left_wrist_0_rgb',
}
STATE_KEY = 'observation.state'


def decode_rot6d(value):
    """Decode the first two rotation-matrix columns using Gram-Schmidt."""
    six = np.asarray(value, dtype=np.float64)
    if six.shape != (6,) or not np.all(np.isfinite(six)):
        raise ValueError('rot6d must contain six finite values')
    columns = six.reshape(2, 3).T
    first = columns[:, 0]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError('rot6d first column has zero length')
    first /= first_norm
    second = columns[:, 1] - first * np.dot(first, columns[:, 1])
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError('rot6d columns are collinear')
    second /= second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def matrix_to_quaternion_xyzw(matrix):
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError('rotation matrix must be finite and 3x3')
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-5) or np.linalg.det(r) < 0.999:
        raise ValueError('rotation matrix is not a proper rotation')
    trace = float(np.trace(r))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
                      (r[1, 0]-r[0, 1])/s, 0.25*s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1+r[0, 0]-r[1, 1]-r[2, 2])*2
            q = np.array([0.25*s, (r[0, 1]+r[1, 0])/s,
                          (r[0, 2]+r[2, 0])/s, (r[2, 1]-r[1, 2])/s])
        elif i == 1:
            s = np.sqrt(1+r[1, 1]-r[0, 0]-r[2, 2])*2
            q = np.array([(r[0, 1]+r[1, 0])/s, 0.25*s,
                          (r[1, 2]+r[2, 1])/s, (r[0, 2]-r[2, 0])/s])
        else:
            s = np.sqrt(1+r[2, 2]-r[0, 0]-r[1, 1])*2
            q = np.array([(r[0, 2]+r[2, 0])/s, (r[1, 2]+r[2, 1])/s,
                          0.25*s, (r[1, 0]-r[0, 1])/s])
    return q / np.linalg.norm(q)


def pose_to_rot6d(pose):
    """Encode [xyz, xyzw] as xyz plus first two rotation columns."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError('TCP pose must have seven finite values')
    q = pose[3:]
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError('TCP quaternion has zero length')
    x, y, z, w = q / norm
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return np.r_[pose[:3], r[:, :2].T.reshape(6)]


def make_observation(images, state, prompt):
    """Build and validate the policy observation in OpenPI feature format."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be a non-empty string')
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (10,) or not np.all(np.isfinite(state)):
        raise ValueError('observation.state must contain ten finite float32 values')
    result = {STATE_KEY: state, 'prompt': prompt}
    for source, target in CAMERA_KEYS.items():
        image = np.asarray(images[source])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f'{source} must be uint8 HWC RGB, got {image.shape}/{image.dtype}')
        if image.shape[:2] != (224, 224):
            raise ValueError(f'{source} must be 224x224, got {image.shape[:2]}')
        result[target] = image
    return result


def relative_chunk_to_world(actions, request_state):
    """Convert a relative policy chunk to absolute TCP poses using one state snapshot.

    The first nine action values are a translation and rotation relative to the
    request pose. The final value is an absolute normalized gripper command.
    """
    actions = np.asarray(actions, dtype=np.float64)
    state = np.asarray(request_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10 or actions.shape[0] == 0:
        raise ValueError(f'actions must have shape (H, 10), got {actions.shape}')
    if not np.all(np.isfinite(actions)):
        raise ValueError('actions contain non-finite values')
    if state.shape != (10,) or not np.all(np.isfinite(state)):
        raise ValueError('request_state must contain ten finite values')
    p0 = state[:3]
    r0 = decode_rot6d(state[3:9])
    output = np.empty((len(actions), 8), dtype=np.float64)
    for i, action in enumerate(actions):
        delta_r = decode_rot6d(action[3:9])
        target_r = r0 @ delta_r
        output[i, :3] = p0 + r0 @ action[:3]
        output[i, 3:7] = matrix_to_quaternion_xyzw(target_r)
        output[i, 7] = np.clip(action[9], 0.0, 1.0)
    return output


def validate_metadata(metadata):
    """Validate the model contract exposed by the socket handshake."""
    if not isinstance(metadata, dict):
        raise ValueError('policy handshake metadata must be a mapping')
    candidates = [metadata]
    for key in ('info', 'metadata', 'policy_info'):
        if isinstance(metadata.get(key), dict):
            candidates.append(metadata[key])
    flat = {}
    for candidate in candidates:
        flat.update(candidate)
    def find(names):
        for name in names:
            if name in flat:
                return flat[name]
        return None
    horizon = find(('prediction_horizon', 'action_horizon'))
    execution = find(('execution_horizon',))
    frequency = find(('control_frequency_hz', 'frequency_hz'))
    action_dim = find(('action_dim', 'action_size'))
    representation = find(('action_representation',))
    if horizon is not None and int(horizon) != 20:
        raise ValueError(f'expected prediction_horizon=20, server reported {horizon}')
    if execution is not None and int(execution) != 10:
        raise ValueError(f'expected execution_horizon=10, server reported {execution}')
    if frequency is not None and float(frequency) != 20.0:
        raise ValueError(f'expected control_frequency_hz=20, server reported {frequency}')
    if action_dim is not None and int(action_dim) != 10:
        raise ValueError(f'expected action_dim=10, server reported {action_dim}')
    expected_representation = 'tcp_relative_xyz_rot6d_absolute_gripper'
    if representation is not None and representation != expected_representation:
        raise ValueError(f'expected action_representation={expected_representation!r}, server reported {representation!r}')
    return {'prediction_horizon': 20 if horizon is None else int(horizon),
            'execution_horizon': 10 if execution is None else int(execution),
            'control_frequency_hz': 20 if frequency is None else float(frequency),
            'action_dim': 10 if action_dim is None else int(action_dim),
            'action_representation': expected_representation if representation is None else representation}


def extract_actions(response):
    if not isinstance(response, dict) or 'actions' not in response:
        raise ValueError('policy response must contain actions')
    actions = np.asarray(response['actions'], dtype=np.float32)
    if actions.shape != (20, 10):
        raise ValueError(f'expected actions with shape (20, 10), got {actions.shape}')
    if not np.all(np.isfinite(actions)):
        raise ValueError('policy returned non-finite actions')
    return actions
