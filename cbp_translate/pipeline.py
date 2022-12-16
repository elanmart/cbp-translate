""" Combines all processing steps into an end-to-end live translation pipeline."""

import os
import tempfile
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import numpy as np

from cbp_translate.components.alignment import (
    FrameMetadata,
    assign_to_frames,
    match_speakers_to_faces,
    match_speakers_to_phrases,
)
from cbp_translate.components.asr import extract_segments
from cbp_translate.components.faces import extract_faces
from cbp_translate.components.loaders import (
    combine_streams,
    extract_audio,
    frame_iterator,
    get_video_metadata,
    save_frames,
)
from cbp_translate.components.speakers import extract_speakers
from cbp_translate.components.subtitles import (
    add_borders,
    add_speaker_marker,
    add_subtitles,
)
from cbp_translate.components.translation import translate_segments
from cbp_translate.modal_ import ROOT, cpu_image, deepl_secret, stub, volume

logger = getLogger(__name__)
Arr = np.ndarray


@dataclass
class Config:
    """Configuration for the pipeline.

    Parameters
    ----------
    target_lang: str
        The language to translate into. Must be a valid DeepL language key.
    speaker_markers: bool
        Whether to add speaker markers to the video.
        Currently markers are simply colored rectangles around the speaker's face.
    border_size: float
        The size of the border to add to the video where the subtitles are displayed.
        This should be a float which is a fraction of the video's height.
    """

    target_lang: str = "EN-GB"
    speaker_markers: bool = True
    border_size: float = 0.1


@stub.function(image=cpu_image, concurrency_limit=100)
def annotate_frames(item: tuple[Arr, FrameMetadata], config: Config) -> Arr:
    """Annotate a single frame with subtitles and speaker markers."""

    # Get the frame and the metadata
    frame, meta = item
    entries = meta.all_speakers[:2]
    picture_h = frame.shape[0]

    # Add black stripes to the top and bottom of the frame
    frame, border_h = add_borders(frame, config.border_size)

    # Add each entry
    for i, entry in enumerate(entries):

        # Shared kwargs
        kwd = dict(row=i, speaker=entry.speaker, border_h=border_h, picture_h=picture_h)

        # Top subtitles -- source language
        frame = add_subtitles(
            frame,
            display_text=entry.source.displayed,
            full_text=entry.source.full,
            location="top",
            **kwd,
        )

        # Bottom subtitles -- target language
        frame = add_subtitles(
            frame,
            display_text=entry.target.displayed,
            full_text=entry.target.full,
            location="bottom",
            **kwd,
        )

        # Speaker markers
        if config.speaker_markers and (entry.face_loc is not None):
            frame = add_speaker_marker(
                frame,
                border_h=border_h,
                face_loc=entry.face_loc,
                speaker=entry.speaker,
            )

    # Done
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
    """Runs the end-to-end live translation pipeline.

    Parameters
    ----------
    path_in: str
        The path to the input video.
    path_out: str
        The path where the output video will be saved.
    config: Config
        The configuration for the pipeline.
    """

    # This temporary directory will be created on a SharedVolume
    with tempfile.TemporaryDirectory(dir=(ROOT / "tmp")) as storage:

        # Setup
        logger.info(f"Processing {path_in} to {path_out} in {storage}")
        deepl_key = os.environ["DEEPL_KEY"]
        path_audio = os.path.join(storage, "audio.wav")
        path_video = os.path.join(storage, "video.mp4")
        fps, length, _ = get_video_metadata(path_in)

        # Run the backbone extraction
        path_audio = extract_audio(path_in, path_audio)
        speakers = extract_speakers.spawn(path_audio)
        segments = extract_segments.spawn(path_audio)
        faces = extract_faces.spawn(path_in)

        # Wait for the extracted text and translate it
        segments = segments.get()
        t_segments = translate_segments(
            segments, target_lang=config.target_lang, auth_key=deepl_key
        )

        # Wait for the speaker diarization, and annotate phrases with speaker IDs
        speakers = speakers.get()
        segment_to_speaker = match_speakers_to_phrases(t_segments, speakers)

        # Wait for the face detection & annotation, and match Speaker IDs with Face IDs
        faces = faces.get()
        face_to_speaker = match_speakers_to_faces(faces, speakers, fps, length)

        # Get complete metadata for each frame
        aligned = assign_to_frames(
            segments=t_segments,
            faces=faces,
            segment_to_speaker=segment_to_speaker,
            face_to_speaker=face_to_speaker,
            fps=fps,
        )

        # Add the metadata to each frame. We parallelize this since it's quite slow otherwise.
        frames = frame_iterator(path_in)
        items = zip(frames, aligned)
        processed = annotate_frames.map(items, kwargs={"config": config})

        # Now save the frames to an mp4 and add the original audio
        save_frames(processed, fps, path_video)
        combine_streams(path_video, path_audio, path_out)

    # We're done here
    return Path(path_out)
