""" Audio / Video IO utilities. """

from itertools import chain
from pathlib import Path
from typing import Iterator, Optional

import cv2
import ffmpeg
import numpy as np

Arr = np.ndarray


def with_suffix(path: str, suffix: str) -> str:
    """Like Path.with_suffix but for strings :)"""

    return str(Path(path).with_suffix(suffix))


def get_video_metadata(path: str) -> tuple[int, int, tuple[int, int]]:
    """Read the FPS, number of frames, and resolution of a video."""

    cap = cv2.VideoCapture(path)

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, frame = cap.read()

    finally:
        cap.release()

    return fps, length, frame.shape[:2]


def _iterate_capture(capture: cv2.VideoCapture) -> Iterator[Arr]:
    while True:
        ret, frame = capture.read()
        if not ret:
            return

        # For some reason OpenCV reads frames in BGR format
        yield frame[..., ::-1]


def frame_iterator(path: str) -> Iterator[Arr]:
    """Iterator yielding video frames as np arrays"""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        yield from _iterate_capture(cap)
    finally:
        cap.release()


def extract_audio(path: str, path_out: Optional[str] = None):
    """Extract audio from a video file using ffmpeg"""

    path_out = path_out or with_suffix(path, ".wav")

    audio = ffmpeg.input(path).audio
    output = ffmpeg.output(audio, path_out)
    output = ffmpeg.overwrite_output(output)
    ffmpeg.run(output, quiet=True)

    return path_out


def save_frames(frames: Iterator[Arr], fps: int, path_out: str):
    """Save frames to a video file using ffmpeg"""

    frames = iter(frames)
    first = next(frames)
    height, width = first.shape[:2]
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
    for frame in chain((first,), frames):
        frame = frame.copy(order="C")
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
