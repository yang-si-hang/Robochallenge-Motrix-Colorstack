"""Execute absolute TCP targets with the scene's rate-limited controller."""
from __future__ import annotations

import numpy as np

from .control import ARM_JOINTS
from .ik import IKSolver


class PolicyExecutionError(RuntimeError):
    """A policy action could not be converted into a valid simulator command."""


class PolicyController:
    def __init__(self, env, ik_solver=None, policy_frequency_hz=20):
        self.env = env
        self.ik = ik_solver or IKSolver()
        self.policy_frequency_hz = float(policy_frequency_hz)
        self.substeps = int(round(1.0 / (self.policy_frequency_hz * env.dt)))
        if self.substeps < 1 or abs(self.substeps * env.dt - 1.0 / self.policy_frequency_hz) > 1e-8:
            raise ValueError('policy frequency must divide the simulation timestep exactly')

    def execute(self, target):
        """Execute [x,y,z,qx,qy,qz,qw,gripper] for one policy period."""
        target = np.asarray(target, dtype=float)
        if target.shape != (8,) or not np.all(np.isfinite(target)):
            raise PolicyExecutionError('absolute target must contain eight finite values')
        if not 0.0 <= target[7] <= 1.0:
            raise PolicyExecutionError(f'normalized gripper target outside [0,1]: {target[7]}')
        qnorm = np.linalg.norm(target[3:7])
        if qnorm < 1e-8:
            raise PolicyExecutionError('target quaternion has zero length')
        pose = target[:7].copy()
        pose[3:] /= qnorm
        try:
            initial = np.array([self.env.model.get_joint(name).get_dof_pos(self.env.data)[0]
                                for name in ARM_JOINTS], dtype=float)
            ik = self.ik.solve(pose, initial)
        except Exception as exc:
            raise PolicyExecutionError(f'IK failed for policy target: {exc}') from exc
        arm_target = np.asarray(ik.joint_angles, dtype=float)
        grip = float(np.clip(target[7], 0.0, 1.0) * 255.0)
        last = None
        for _ in range(self.substeps):
            last = self.env.step({'arm_qpos': arm_target, 'gripper_target': grip})
        return last, {'target_tcp_pose': pose.tolist(), 'arm_target_rad': arm_target.tolist(),
                      'gripper_command': grip, 'ik_position_error_m': ik.position_error,
                      'ik_orientation_error_rad': ik.orientation_error,
                      'ik_iterations': ik.iterations, 'substeps': self.substeps}
