from decord import VideoReader, cpu
import math
import numpy as np
from PIL import Image

VIDEO_MAXLEN = 128
IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 1 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
def _get_video_sample_frames(video_stream, **kwargs) -> int:
    r"""
    Computes video sample frames according to fps.
    """
    video_maxlen: int = kwargs.get("video_maxlen")
    total_frames = len(video_stream)
    real_fps = video_stream.get_avg_fps()
    video_fps: float = kwargs.get("video_fps", real_fps)
    sample_frames = float(total_frames / real_fps) * video_fps
    sample_frames = min(total_frames, video_maxlen, sample_frames)
    return math.floor(sample_frames)  // 2 * 2

def process_video(video_path, **kwargs):
    """
    处理标准视频文件并返回提取的帧。
    """
    frames = []
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    sample_frames = _get_video_sample_frames(vr, **kwargs)
    sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
    
    # 批量读取指定帧
    batch_frames = vr.get_batch(sample_indices).asnumpy()
    frames = [_preprocess_image(Image.fromarray(frame), **kwargs) for frame in batch_frames]
    
    print("*"*10)
    print('video_path:', video_path)
    print('total_frames:', total_frames)
    print('sample_frames:', sample_frames)
    print('sample_indices:', sample_indices)
    print("*"*10)

    return frames


def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

def _preprocess_image(image, **kwargs):
    if image.mode != "RGB":
        image = image.convert("RGB")
    min_pixels = kwargs.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = kwargs.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, int(total_pixels // kwargs.get('video_maxlen', 64)) * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = kwargs.get("max_pixels", max_pixels)
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    image = image.resize((resized_width, resized_height), resample=Image.NEAREST)

    return image

