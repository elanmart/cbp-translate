import os
import tempfile
from dataclasses import dataclass
from logging import getLogger
from os import path
from pathlib import Path

import numpy as np

from .alignment import (
    FrameMetadata,
    assign_to_frames,
    match_speakers_to_faces,
    match_speakers_to_phrases,
)
from .asr import extract_segments
from .faces import extract_faces
from .loaders import (
    combine_streams,
    extract_audio,
    frame_iterator,
    get_video_metadata,
    save_frames,
)
from .modal_ import ROOT, cpu_image, deepl_secret, stub, volume
from .speakers import extract_speakers
from .subtitles import add_borders, add_speaker_marker, add_subtitles
from .translation import translate_segments

logger = getLogger(__name__)
Arr = np.ndarray


@dataclass
class Config:
    target_lang: str = "EN-GB"
    speaker_markers: bool = True
    border_size: float = 0.1


@stub.function(image=cpu_image, concurrency_limit=100)
def annotate_frames(item: tuple[Arr, list[FrameMetadata]], config: Config) -> Arr:
    frame, entries = item
    entries = entries[:2]
    picture_h = frame.shape[0]

    frame, border_h = add_borders(frame, config.border_size)

    for i, entry in enumerate(entries):

        kwd = dict(row=i, speaker=entry.speaker, border_h=border_h, picture_h=picture_h)

        frame = add_subtitles(
            frame,
            display_text=entry.text_src_displayed,
            full_text=entry.text_src_full,
            location="top",
            **kwd,
        )

        frame = add_subtitles(
            frame,
            display_text=entry.text_tgt_displayed,
            full_text=entry.text_tgt_full,
            location="bottom",
            **kwd,
        )

        if config.speaker_markers and (entry.face_loc is not None):
            frame = add_speaker_marker(
                frame,
                border_h=border_h,
                face_loc=entry.face_loc,
                speaker=entry.speaker,
            )

    return frame


@stub.function(
    image=cpu_image,
    secret=deepl_secret,
    shared_volumes={str(ROOT): volume},
    cpu=1.1,
    memory=6000,
    timeout=10_000,
)
def run(path_in: str, path_out: str, config: Config) -> Path:

    with tempfile.TemporaryDirectory(dir=(ROOT / "tmp")) as storage:

        logger.info(f"Processing {path_in} to {path_out} in {storage}")
        deepl_key = os.environ["DEEPL_KEY"]

        path_audio = path.join(storage, "audio.wav")
        path_video = path.join(storage, "video.mp4")
        fps, length, _ = get_video_metadata(path_in)

        path_audio = extract_audio(path_in, path_audio)
        speakers = extract_speakers.spawn(path_audio)
        segments = extract_segments.spawn(path_audio)
        faces = extract_faces.spawn(path_in)

        segments = segments.get()
        t_segments = translate_segments(
            segments, target_lang=config.target_lang, auth_key=deepl_key
        )

        speakers = speakers.get()
        segment_to_speaker = match_speakers_to_phrases(t_segments, speakers)

        faces = faces.get()
        face_to_speaker = match_speakers_to_faces(faces, speakers, fps, length)

        aligned = assign_to_frames(
            segments=t_segments,
            faces=faces,
            segment_to_speaker=segment_to_speaker,
            face_to_speaker=face_to_speaker,
            fps=fps,
        )

        frames = frame_iterator(path_in)
        items = zip(frames, aligned)
        processed = annotate_frames.map(items, kwargs={"config": config})

        save_frames(processed, fps, path_video)
        combine_streams(path_video, path_audio, path_out)

    return Path(path_out)
