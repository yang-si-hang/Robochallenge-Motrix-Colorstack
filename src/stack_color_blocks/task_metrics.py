"""MotrixSim stack task metrics, kept independent from expert trajectories."""
from __future__ import annotations

import numpy as np


def evaluate_stack(initial_blocks, final_blocks, rest_positions, pick_color, place_color,
                   table_top_z, block_half_size, max_pick_z=None, max_pick_height=None):
    """Calculate stage proxies using the published local geometric thresholds.

    Grasp detection is approximated by a lifted block exceeding 2 cm above its
    resting center. Callers may supply a measured maximum height from the trace.
    """
    picked_initial = np.asarray(initial_blocks[pick_color], dtype=float)
    picked_final = np.asarray(final_blocks[pick_color], dtype=float)
    support = np.asarray(final_blocks[place_color], dtype=float)
    resting_center_z = float(table_top_z + block_half_size)
    measured_max_z = float(picked_final[2] if max_pick_z is None else max_pick_z)
    grasp = measured_max_z >= resting_center_z + 0.02
    xy_error = float(np.linalg.norm(picked_final[:2] - support[:2]))
    height_delta = float(picked_final[2] - support[2])
    expected_delta = 2.0 * float(block_half_size)
    picked_tilt = _tilt(picked_final[3:7])
    support_tilt = _tilt(support[3:7])
    rest = np.asarray(rest_positions, dtype=float)
    drift = (float(np.linalg.norm(rest[-1, :2] - rest[0, :2])) if rest.ndim == 2 and len(rest) >= 2
             else None)
    placement = (xy_error <= 0.01 and abs(height_delta - expected_delta) <= 0.008
                 and picked_tilt <= 0.15 and support_tilt <= 0.15
                 and drift is not None and drift <= 0.006)
    return {
        'grasp_stage': bool(grasp), 'place_stage': bool(placement),
        'success': bool(grasp and placement),
        'retry_count': None,
        'progress_score_no_retry': float(5 * int(grasp) + 5 * int(placement)),
        'retry_adjusted_score': None,
        'metrics': {'resting_pick_center_z_m': resting_center_z,
                    'max_pick_z_m': measured_max_z,
                    'pick_height_above_rest_m': measured_max_z - resting_center_z,
                    'stack_xy_error_m': xy_error,
                    'center_height_delta_m': height_delta,
                    'expected_center_height_delta_m': expected_delta,
                    'picked_block_tilt_rad': picked_tilt,
                    'support_block_tilt_rad': support_tilt,
                    'rest_drift_m': drift,
                    'initial_pick_pose': picked_initial.tolist()},
    }


def _tilt(quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    rzz = 1.0 - 2.0 * (x*x + y*y)
    return float(np.arccos(np.clip(rzz, -1.0, 1.0)))


def summarize(results):
    total = len(results)
    if not total:
        raise ValueError('cannot summarize zero evaluation episodes')
    grasped = sum(bool(r.get('grasp_stage')) for r in results)
    placed = sum(bool(r.get('place_stage')) for r in results)
    succeeded = sum(bool(r.get('success')) for r in results)
    scores = [float(r.get('progress_score_no_retry', 0.0)) for r in results]
    durations = [float(r.get('duration_sim_s', 0.0)) for r in results]
    return {'episodes': total, 'success_count': succeeded,
            'success_rate': succeeded / total,
            'mean_progress_score_no_retry': float(np.mean(scores)),
            'total_progress_score_no_retry': float(np.sum(scores)),
            'mean_duration_sim_s': float(np.mean(durations)),
            'grasp_stage_success_rate': grasped / total,
            'place_stage_success_rate_all': placed / total,
            'place_stage_success_rate_given_grasp': (placed / grasped if grasped else None),
            'score_definition': '5 points per local stage proxy; retries not detected'}
