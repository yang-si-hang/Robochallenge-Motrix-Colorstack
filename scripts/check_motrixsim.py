"""Verify MotrixSim physics and two headless RGB cameras on this machine."""

from __future__ import annotations

import argparse
import importlib.metadata
import tempfile
from pathlib import Path

import numpy as np
from motrixsim import SceneData, load_model, step
from motrixsim.render import RenderApp


SCENE = """\
<mujoco model="motrixsim_smoke">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 2"/>
    <camera name="global" pos="0 0 1" quat="1 0 0 0"/>
    <camera name="wrist_proxy" pos="0.05 0 0.7" quat="1 0 0 0"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.4 0.4 0.4 1"/>
    <body name="cube" pos="0 0 0.5">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.03 0.03 0.03" mass="0.1" rgba="1 0.6 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        scene_path = Path(tmp) / "scene.xml"
        scene_path.write_text(SCENE)
        model = load_model(str(scene_path))
        for camera in model.cameras:
            camera.set_render_target("image", 128, 128)
        data = SceneData(model)

        rgb_frames = []
        with RenderApp(headless=True) as render:
            render.launch(model)
            for index in range(len(model.cameras)):
                render.get_camera(index).active = True
            # Asset loading is asynchronous; short warmups can produce blank frames.
            for _ in range(120):
                step(model, data)
                render.sync(data, wait=True)
            captures = [render.get_camera(i).capture() for i in range(2)]
            render.sync(data, wait=True)
            for index, capture in enumerate(captures):
                if capture.state != "done":
                    raise RuntimeError(f"camera {index} capture: {capture.state}")
                image = capture.take_image()
                if image is None:
                    raise RuntimeError(f"camera {index} returned no image")
                rgb = image.pixels[:, :, :3].copy()
                image.save_to_disk(str(args.output_dir / f"camera_{index}.png"))
                rgb_frames.append(rgb)
                del image
            del captures

    if any(frame.std() < 5.0 for frame in rgb_frames):
        raise RuntimeError("camera output is nearly uniform; scene may be invisible")
    difference = np.abs(rgb_frames[0].astype(np.int16) - rgb_frames[1].astype(np.int16)).mean()
    if difference < 0.2:
        raise RuntimeError("two camera outputs are nearly identical")
    print(
        f"PASS motrixsim={importlib.metadata.version('motrixsim')} "
        f"physics_steps=120 rgb_shape={rgb_frames[0].shape} "
        f"mean_camera_difference={difference:.2f} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
