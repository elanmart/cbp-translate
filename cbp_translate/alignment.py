import math
from typing import Optional, Iterable
from dataclasses import dataclass
from collections import defaultdict

from .faces import OnFrameFaces, FaceLocation
from .speakers import SpeakerSegment
from .translation import TranslatedSegment


@dataclass
class FrameMetadata:
    text_src_displayed: str = ""
    text_src_full: str = ""
    text_tgt_displayed: str = ""
    text_tgt_full: str = ""
    speaker: int = 0
    face_loc: Optional[FaceLocation] = None


def match_speakers_to_faces(
    faces: Iterable[OnFrameFaces], speakers: list[SpeakerSegment], fps: int, n_frames: int
) -> dict[int, str]:

    def idx(t: float) -> int:
        return math.ceil(t * fps)

    voices = [[] for _ in range(n_frames)]

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
                coocurrence[face.id_][voice] += 1

    # Simple heuritic to find the best match
    matched = {}
    for face_id in coocurrence:
        matched[face_id] = max(coocurrence[face_id].items(), key=lambda x: x[1])[0]

    return matched


def match_speakers_to_phrases(
    segments: list[TranslatedSegment], speakers: list[SpeakerSegment]
) -> list[str]:
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
    faces: list[OnFrameFaces],
    segment_to_speaker: list[str],
    face_to_speaker: dict[int, str],
    fps: int,
) -> list[list[FrameMetadata]]:
    def idx(t: float) -> int:
        return math.ceil(t * fps)

    def n_chars(text: str, frame: int, start_frame: int, end_frame: int) -> int:
        end_frame = max(start_frame + 1, end_frame - fps // 2)
        length = end_frame - start_frame
        progress = (frame - start_frame) / length
        return 1 + int(len(text) * progress)

    n_frames = len(faces)
    frame_metadata = [[] for _ in range(n_frames)]

    # Assign the text
    for s_idx, segment in enumerate(segments):

        start = idx(segment.start)
        end = idx(segment.end)
        display_end = idx(segment.end + 1)

        speaker = segment_to_speaker[s_idx]
        speaker_id = int(speaker.split("_")[1])

        for frame_idx in range(start, min(display_end, len(faces))):
            matched_faces = [
                f for f in faces[frame_idx] if face_to_speaker[f.id_] == speaker
            ]
            face_loc = None if len(matched_faces) == 0 else matched_faces[0].location

            n_chars_src = n_chars(segment.text_src, frame_idx, start, end)
            n_chars_src = min(n_chars_src, len(segment.text_src))
            start_src = max(0, n_chars_src - 105)
            
            n_chars_tgt = n_chars(segment.text_tgt, frame_idx, start, end)
            n_chars_tgt = min(n_chars_tgt, len(segment.text_tgt))
            start_tgt = max(0, n_chars_tgt - 105)

            meta = FrameMetadata(
                text_src_displayed=segment.text_src[start_src : n_chars_src],
                text_src_full=segment.text_src,
                text_tgt_displayed=segment.text_tgt[start_tgt : n_chars_tgt],
                text_tgt_full=segment.text_tgt,
                speaker=speaker_id,
                face_loc=face_loc,
            )

            frame_metadata[frame_idx].append(meta)

    return frame_metadata
