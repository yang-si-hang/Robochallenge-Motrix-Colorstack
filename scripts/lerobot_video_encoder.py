"""LeRobot 0.4.4 episode video encoder with configurable NVENC options.

The worker and frame encoding flow below follow LeRobot's
``_encode_video_worker`` and ``encode_video_frames``. Only the codec options
needed by this collection script are exposed here. LeRobot still handles
episode metadata and concatenation.
"""

import glob
import shutil
import tempfile
from pathlib import Path

import av
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH


class ConfigurableVideoDataset(LeRobotDataset):
    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        return encode_video_worker(
            video_key, episode_index, self.root, self.fps, self.vcodec,
            self.video_gop, self.video_bframes, self.video_qp, self.video_preset,
            self._encoder_threads,
        )


def encode_video_worker(video_key, episode_index, root, fps, vcodec, gop, bframes, qp, preset,
                        encoder_threads=None):
    """Local copy of LeRobot's worker, with encoder settings forwarded."""
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f'{video_key}_{episode_index:03d}.mp4'
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    img_dir = (root / fpath).parent
    try:
        encode_video_frames(img_dir, temp_path, fps, vcodec, gop, bframes, qp, preset,
                            encoder_threads=encoder_threads)
    except Exception:
        shutil.rmtree(temp_path.parent)
        raise
    shutil.rmtree(img_dir)
    return temp_path


def encode_video_frames(imgs_dir, video_path, fps, vcodec, gop, bframes, qp, preset,
                        encoder_threads=None):
    """Local copy of LeRobot's PNG-to-MP4 flow with NVENC codec options."""
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    template = 'frame-' + '[0-9]' * 6 + '.png'
    input_list = sorted(glob.glob(str(imgs_dir / template)),
                        key=lambda x: int(x.split('-')[-1].split('.')[0]))
    if not input_list:
        raise FileNotFoundError(f'No images found in {imgs_dir}.')
    with Image.open(input_list[0]) as image:
        width, height = image.size

    # FFmpeg 8 NVENC uses g=0 for all-intra. Keep B frames below the GOP.
    ffmpeg_gop = 0 if gop == 1 else gop
    effective_bframes = 0 if gop == 1 else (min(3, gop - 2) if bframes is None else bframes)
    options = {'rc': 'constqp', 'qp': str(qp),
               'g': str(ffmpeg_gop), 'bf': str(effective_bframes)}
    if preset is not None:
        options['preset'] = preset
    if encoder_threads is not None:
        options['threads'] = str(encoder_threads)

    with av.open(str(video_path), 'w') as output:
        stream = output.add_stream(vcodec, fps, options=options)
        if gop == 1:
            # PyAV does not apply g=0 from stream options to NVENC by itself.
            stream.codec_context.gop_size = 0
            stream.codec_context.max_b_frames = 0
        stream.pix_fmt = 'yuv420p'
        stream.width = width
        stream.height = height
        for input_data in input_list:
            with Image.open(input_data) as image:
                packet = stream.encode(av.VideoFrame.from_image(image.convert('RGB')))
                if packet:
                    output.mux(packet)
        packet = stream.encode()
        if packet:
            output.mux(packet)

    if not video_path.exists():
        raise OSError(f'Video encoding did not work. File not found: {video_path}.')
