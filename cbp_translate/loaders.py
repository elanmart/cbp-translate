from pathlib import Path
from typing import Optional

import cv2
import ffmpeg
import numpy as np

from . import Arr

def with_suffix(path: str, suffix: str) -> str:
    """
    Examples
    --------
    >>> with_suffix("foo.mp4", ".mp3")
    'foo.mp3'
    """

    return str(Path(path).with_suffix(suffix))


def load_frames(path: str) -> tuple[list[Arr], int]:

    # Open the stream
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[..., ::-1])

    # Close the stream
    cap.release()

    # Done
    return frames, fps


def extract_audio(path: str, path_out: Optional[str] = None):
    """Extract audio from a video file using ffmpeg"""

    path_out = path_out or with_suffix(path, ".mp3")

    audio = ffmpeg.input(path).audio
    output = ffmpeg.output(audio, path_out)
    output = ffmpeg.overwrite_output(output)
    ffmpeg.run(output, quiet=True)

    return path_out


def save_frames(frames: list[Arr], fps: int, path_out: str):
    """Save frames to a video file using ffmpeg"""

    height, width, _ = frames[0].shape
    video = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{width}x{height}",
        framerate=fps,
    )

    output = ffmpeg.output(video, path_out, pix_fmt="yuv420p", vcodec="h264")
    output = ffmpeg.overwrite_output(output)
    process = ffmpeg.run_async(output, pipe_stdin=True, quiet=True)
    for frame in frames:
        frame = frame[:, :, ::-1].copy(order="C")
        process.stdin.write(frame.tobytes())  # type: ignore
    process.stdin.close()  # type: ignore
    process.wait()

    return path_out


def combine_streams(path_video: str, path_audio: str, path_out: str):
    """Use ffmpeg to combine video and audio streams"""

    video = ffmpeg.input(path_video)
    audio = ffmpeg.input(path_audio)
    output = ffmpeg.output(video, audio, path_out, vcodec="copy", acodec="aac")
    output = ffmpeg.overwrite_output(output)
    ffmpeg.run(output, quiet=True)

    return path_out
