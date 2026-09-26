"""Non-actuating UR5e inverse kinematics for the stack color blocks scene."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from motrixsim import SceneData, load_model
from motrixsim.ik import DlsSolver, IkChain

from .control import ARM_JOINTS, HOME_QPOS

ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = ROOT / "assets" / "stack_color_blocks.xml"
END_EFFECTOR_OFFSET = (0.0, 0.0, 0.145, 0.0, 0.0, 0.0, 1.0)
POSITION_TOLERANCE = 1e-3  # metres
ORIENTATION_TOLERANCE = 1e-3  # radians


@dataclass(frozen=True)
class IKResult:
    joint_angles: tuple[float, ...]
    position_error: float
    orientation_error: float
    iterations: int


class IKError(RuntimeError):
    """The target could not be reached and independently verified."""


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    # Quaternion order is xyzw. q and -q encode the same orientation.
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _load_kinematics():
    model = load_model(str(SCENE_PATH))
    chain = IkChain(model, end_link="grip_base", start_link="base",
                    end_effector_offset=END_EFFECTOR_OFFSET)
    return model, chain


class IKSolver:
    """Reusable IK model/solver for resolving a sequence of nearby poses."""

    def __init__(self):
        self.model, self.chain = _load_kinematics()
        self.data = SceneData(self.model)
        self.check_data = SceneData(self.model)
        self.solver = DlsSolver(max_iter=300, tolerance=1e-7, damping=1e-3,
                                adaptive_damping=True)
        limits = np.asarray(self.model.joint_limits, dtype=float)
        lo, hi = [], []
        for name in ARM_JOINTS:
            index = self.model.joint_dof_pos_indices[self.model.get_joint_index(name)]
            lo.append(limits[0, index]); hi.append(limits[1, index])
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)

    def solve(self, target_pose: Sequence[float], initial_qpos: Sequence[float] | None = None) -> IKResult:
        """Solve world-frame ``[x,y,z,qx,qy,qz,qw]`` without touching caller state."""
        pose = np.asarray(target_pose, dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("target_pose must contain exactly seven finite values [x,y,z,qx,qy,qz,qw]")
        qnorm = float(np.linalg.norm(pose[3:]))
        if qnorm < 1e-12:
            raise ValueError("target quaternion must have non-zero norm")
        pose = pose.copy()
        pose[3:] /= qnorm

        seed = np.asarray(HOME_QPOS if initial_qpos is None else initial_qpos, dtype=float)
        if seed.shape != (6,) or not np.all(np.isfinite(seed)):
            raise ValueError("initial_qpos must contain exactly six finite joint angles")
        if np.any(seed < self.lo - 1e-8) or np.any(seed > self.hi + 1e-8):
            raise ValueError("initial_qpos violates UR5e joint limits")

        self.data.reset(self.model)
        for name, value in zip(ARM_JOINTS, seed):
            self.model.get_joint(name).set_dof_pos(self.data, np.asarray([value], dtype=float))
        self.model.forward_kinematic(self.data)
        raw = np.asarray(self.solver.solve(self.chain, self.data, pose), dtype=float).reshape(-1)
        if raw.size != 8 or not np.all(np.isfinite(raw)):
            raise IKError("IK solver failed to produce a finite six-joint solution")
        iterations, _residual, q = int(raw[0]), float(raw[1]), raw[2:]
        if q.shape != (6,) or np.any(q < self.lo - 1e-6) or np.any(q > self.hi + 1e-6):
            raise IKError("IK solver returned joint angles outside the UR5e limits")

        self.check_data.reset(self.model)
        for name, value in zip(ARM_JOINTS, q):
            self.model.get_joint(name).set_dof_pos(self.check_data, np.asarray([value], dtype=float))
        self.model.forward_kinematic(self.check_data)
        actual = np.asarray(self.chain.get_end_effector_pose(self.check_data), dtype=float).reshape(7)
        pos_error = float(np.linalg.norm(actual[:3] - pose[:3]))
        actual_q = actual[3:] / np.linalg.norm(actual[3:])
        ori_error = _rotation_error(actual_q, pose[3:])
        if pos_error > POSITION_TOLERANCE or ori_error > ORIENTATION_TOLERANCE:
            raise IKError(f"target unreachable or not converged: position error {pos_error:.6g} m, "
                          f"orientation error {ori_error:.6g} rad (solver residual {_residual:.6g})")
        return IKResult(tuple(float(x) for x in q), pos_error, ori_error, iterations)


def solve_ik(target_pose: Sequence[float], initial_qpos: Sequence[float] | None = None) -> IKResult:
    """Solve one world-frame pose without touching caller state."""
    return IKSolver().solve(target_pose, initial_qpos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve UR5e IK without moving a simulated robot")
    parser.add_argument("pose", nargs=7, type=float, metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                        help="world-frame pose in metres and xyzw quaternion order")
    parser.add_argument("--initial-qpos", nargs=6, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="optional initial joint angles in radians; defaults to HOME_QPOS")
    args = parser.parse_args()
    result = solve_ik(args.pose, args.initial_qpos)
    print("joint_angles_rad:", list(result.joint_angles))
    print(f"position_error_m: {result.position_error:.9g}")
    print(f"orientation_error_rad: {result.orientation_error:.9g}")
    print("iterations:", result.iterations)


if __name__ == "__main__":
    main()
