"""Seeded episode lighting and RGB quality checks."""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np


def sample_lighting(base_config, seed, attempt):
    """Vary key direction broadly while preserving the diffuse light budget."""
    config = deepcopy(base_config)
    spec = config['lighting_randomization']
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5, attempt]))
    lights = {light['name']: light for light in config['lighting']}
    key, ceiling, rear = (lights[name] for name in ('window_key', 'ceiling_fill', 'rear_fill'))

    base_dir = np.asarray(key['dir'], dtype=float)
    azimuth = math.atan2(base_dir[1], base_dir[0])
    elevation = math.atan2(-base_dir[2], np.linalg.norm(base_dir[:2]))
    azimuth += math.radians(rng.uniform(-spec['key_azimuth_jitter_deg'],
                                         spec['key_azimuth_jitter_deg']))
    elevation += math.radians(rng.uniform(-spec['key_elevation_jitter_deg'],
                                           spec['key_elevation_jitter_deg']))
    elevation = float(np.clip(elevation, math.radians(spec['key_min_elevation_deg']),
                      math.radians(spec['key_max_elevation_deg'])))
    key['dir'] = [math.cos(elevation)*math.cos(azimuth),
                  math.cos(elevation)*math.sin(azimuth), -math.sin(elevation)]

    brightness = rng.uniform(*spec['brightness_scale_range'])
    # Transfer a small amount between key and ceiling fill. Their sum, plus
    # the rear fill, changes only by the shared brightness multiplier.
    transfer = rng.uniform(-spec['key_fill_transfer'], spec['key_fill_transfer'])
    for index in range(3):
        key['diffuse'][index] = brightness*(key['diffuse'][index] + transfer)
        ceiling['diffuse'][index] = brightness*(ceiling['diffuse'][index] - transfer)
        rear['diffuse'][index] *= brightness
    if min(ceiling['diffuse']) <= 0:
        raise ValueError('lighting randomization makes ceiling fill nonpositive')
    config['lighting_sample'] = {
        'seed': seed, 'attempt': attempt,
        'key_azimuth_deg': math.degrees(azimuth),
        'key_elevation_deg': math.degrees(elevation),
        'brightness_scale': brightness, 'key_fill_transfer': transfer,
    }
    return config


def image_quality(images, thresholds):
    """Check the central task area in both cameras; report measurable reasons."""
    results = {}
    passed = True
    for name, rgb in images.items():
        h, w = rgb.shape[:2]
        region = rgb[h//5:4*h//5, w//5:4*w//5, :3].astype(np.float32)
        luminance = region @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        p5, median, p95 = np.percentile(luminance, [5, 50, 95])
        bright = float(np.mean(luminance >= 245))
        dark = float(np.mean(luminance <= 15))
        contrast = float(luminance.std())
        reasons = []
        if not thresholds['median_range'][0] <= median <= thresholds['median_range'][1]:
            reasons.append('median_brightness')
        if p95 > thresholds['max_p95'] or bright > thresholds['max_bright_fraction']:
            reasons.append('clipped_highlights')
        if dark > thresholds['max_dark_fraction']:
            reasons.append('dark_area')
        if contrast < thresholds['min_std']:
            reasons.append('low_contrast_or_unready')
        results[name] = {'p5': float(p5), 'median': float(median), 'p95': float(p95),
                         'bright_fraction': bright, 'dark_fraction': dark,
                         'std': contrast, 'passed': not reasons, 'reasons': reasons}
        passed &= not reasons
    return passed, results


def capture_ready(env, timeout_seconds=10):
    """Wait for asynchronous renderer assets before judging lighting."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        env.render.sync(env.data, wait=True)
        images = env.capture()
        if (all(image.std() > 15 for image in images.values()) and
                np.abs(images['global_rgb'].astype(np.float32) -
                       images['wrist_rgb'].astype(np.float32)).mean() > 1):
            return images
        time.sleep(0.1)
    raise RuntimeError('camera images did not become ready before lighting validation')
