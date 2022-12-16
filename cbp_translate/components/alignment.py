""" Various heuristics to combine ASR, diarization, and face recognition results. """

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, cast

from cbp_translate.components.faces import FaceLocation, OnFrameRecognized
from cbp_translate.components.speakers import SpeakerSegment
from cbp_translate.components.translation import TranslatedSegment


@dataclass
class Text:
    """Text displayed on a single frame. Also includes the full subtitle for proper alignment."""

    displayed: str = ""
    full: str = ""


@dataclass
class SpeakerMetadata:
    """Metadata needed to annotate a single speaker on a single frame."""

    source: Text = field(default_factory=Text)
    target: Text = field(default_factory=Text)
    speaker: int = 0
    face_loc: Optional[FaceLocation] = None


@dataclass
class FrameMetadata:
    """All metadata needed to annotate a single frame."""

    all_speakers: list[SpeakerMetadata] = field(default_factory=list)


def match_speakers_to_faces(
    faces: Iterable[OnFrameRecognized],
    speakers: list[SpeakerSegment],
    fps: int,
    n_frames: int,
) -> dict[int, str]:

    """Match speaker IDs to Face IDs by simple co-occurrence metric."""

    # Convert the timestamp to frame index
    def idx(t: float) -> int:
        return math.ceil(t * fps)

    # All voices heard in a given frame (you know what I mean)
    voices = [[] for _ in range(n_frames)]

    # Add all voices to the list
    for segment in speakers:
        start = idx(segment.start)
        end = idx(segment.end)
        end = min(end, n_frames - 1)
        for i in range(start + 1, end):
            voices[i].append(segment.id_)

    # Count the overlap between faces and voices
    coocurrence = defaultdict(lambda: defaultdict(int))
    for frame_voices, frame_faces in zip(voices, faces):
        for face in frame_faces:
            for voice in frame_voices:
                voice = cast(str, voice)
                coocurrence[face.person_id][voice] += 1

    # find the best match
    matched = {}
    for face_id in coocurrence:
        # x[1] here is selecting the count from a (voice_id, count) pair
        voice_id, count = max(coocurrence[face_id].items(), key=lambda x: x[1])
        matched[face_id] = voice_id

    return matched


def match_speakers_to_phrases(
    segments: list[TranslatedSegment], speakers: list[SpeakerSegment]
) -> list[str]:
    """Assign a speaker ID to each transcribed phrase."""

    def overlap(s0: TranslatedSegment, s1: SpeakerSegment) -> float:
        return max(0, min(s0.end, s1.end) - max(s0.start, s1.start))

    def length(speaker: SpeakerSegment):
        return speaker.end - speaker.start

    matched = []
    for text in segments:
        speaker = max(
            speakers, key=lambda speaker: overlap(text, speaker) / length(speaker)
        )
        matched.append(speaker.id_)

    return matched


def assign_to_frames(
    segments: list[TranslatedSegment],
    faces: list[OnFrameRecognized],
    segment_to_speaker: list[str],
    face_to_speaker: dict[int, str],
    fps: int,
) -> list[FrameMetadata]:
    """Take all the matched / aligned items, and produce a usable metadata for each frame."""

    # Convert timestampt to frame index
    def idx(t: float) -> int:
        return math.ceil(t * fps)

    # Get number of characters to display
    def n_chars(
        full_text: str, curr_frame: int, start_frame: int, end_frame: int
    ) -> int:
        end_frame = max(start_frame + 1, end_frame - fps // 2)
        length = end_frame - start_frame
        progress = (curr_frame - start_frame) / length
        return 1 + int(len(full_text) * progress)

    # We'll produce metadata for each frame
    n_frames = len(faces)
    metadatas = [[] for _ in range(n_frames)]

    # Brrrr
    for s_idx, segment in enumerate(segments):

        # Get the start and end frame for this segment
        start = idx(segment.start)
        end = idx(segment.end)

        # We always want to show the text for 1 second longer
        display_end = idx(segment.end + 1)

        # Speaker ID
        speaker = segment_to_speaker[s_idx]
        speaker_id = int(speaker.split("_")[1])

        # Add this speaker ID to each frame where the phrase is spoken
        for frame_idx in range(start, min(display_end, len(faces))):

            # Face IDs and locations
            matched_faces = [
                f for f in faces[frame_idx] if face_to_speaker[f.person_id] == speaker
            ]
            face_loc = None if len(matched_faces) == 0 else matched_faces[0].location

            # Counting how much text to display
            n_chars_src = n_chars(segment.text_src, frame_idx, start, end)
            n_chars_src = min(n_chars_src, len(segment.text_src))
            start_src = max(0, n_chars_src - 105)
            n_chars_tgt = n_chars(segment.text_tgt, frame_idx, start, end)
            n_chars_tgt = min(n_chars_tgt, len(segment.text_tgt))
            start_tgt = max(0, n_chars_tgt - 105)

            # Produce a metadata entry
            meta = SpeakerMetadata(
                source=Text(
                    displayed=segment.text_src[start_src:n_chars_src],
                    full=segment.text_src,
                ),
                target=Text(
                    displayed=segment.text_tgt[start_tgt:n_chars_tgt],
                    full=segment.text_tgt,
                ),
                speaker=speaker_id,
                face_loc=face_loc,
            )

            metadatas[frame_idx].append(meta)

    # Convert to a nice format and we're done
    return [FrameMetadata(items) for items in metadatas]
