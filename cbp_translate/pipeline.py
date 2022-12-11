import tempfile
from dataclasses import dataclass
from logging import getLogger
from os import path
from typing import Iterator

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


def annotate_frames(
    frames: Iterator[Arr], aligned: list[list[FrameMetadata]], config: Config
) -> Iterator[Arr]:

    for frame, entries in zip(frames, aligned):
        frame = frame.copy()
        entries = entries[:2]

        picture_h = frame.shape[0]
        frame, border_h = add_borders(frame, config.border_size)

        for i, entry in enumerate(entries):

            kwd = dict(
                row=i, speaker=entry.speaker, border_h=border_h, picture_h=picture_h
            )

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

        yield frame


def main(path_in: str, path_out: str, config: Config):

    with tempfile.TemporaryDirectory() as tmp:

        logger.info(f"Processing {path_in} to {path_out} in {tmp}")

        path_audio = path.join(tmp, "audio.wav")
        path_video = path.join(tmp, "video.mp4")

        path_audio = extract_audio(path_in, path_audio)
        speakers = extract_speakers(path_audio)
        segments = extract_segments(path_audio)
        t_segments = translate_segments(segments, target_lang=config.target_lang)

        fps, length, shape = get_video_metadata(path_in)
        frames = frame_iterator(path_in)
        faces = extract_faces(frames)
        faces = list(faces)

        segment_to_speaker = match_speakers_to_phrases(t_segments, speakers)
        face_to_speaker = match_speakers_to_faces(faces, speakers, fps, length)

        aligned = assign_to_frames(
            segments=t_segments,
            faces=faces,
            segment_to_speaker=segment_to_speaker,
            face_to_speaker=face_to_speaker,
            fps=fps,
        )

        frames = frame_iterator(path_in)
        processed = annotate_frames(frames, aligned, config)

        save_frames(processed, fps, path_video)
        combine_streams(path_video, path_audio, path_out)

    return path_out
