import tempfile
from dataclasses import dataclass
from os import path

from .alignment import (
    assign_to_frames,
    match_speakers_to_faces,
    match_speakers_to_phrases,
)
from .asr import extract_segments
from .faces import extract_faces
from .loaders import combine_streams, extract_audio, load_frames, save_frames
from .speakers import extract_speakers
from .subtitles import add_speaker_marker, add_subtitles
from .translation import translate_segments


@dataclass
class Config:
    target_lang: str = "EN-GB"
    subtitles_location: str = "bottom"
    speaker_markers: bool = True


def main(path_in: str, path_out: str, config: Config):

    with tempfile.TemporaryDirectory() as tmp:

        path_audio = path.join(tmp, "audio.mp3")
        path_video = path.join(tmp, "frames.mp4")

        frames, fps = load_frames(path_in)
        path_audio = extract_audio(path_in, path_audio)

        segments = extract_segments(path_audio)
        t_segments = translate_segments(segments)

        speakers = extract_speakers(path_audio)
        _, faces = extract_faces(frames)

        segment_to_speaker = match_speakers_to_phrases(t_segments, speakers)
        face_to_speaker = match_speakers_to_faces(faces, speakers, fps)

        aligned = assign_to_frames(
            segments=t_segments,
            faces=faces,
            segment_to_speaker=segment_to_speaker,
            face_to_speaker=face_to_speaker,
            fps=fps,
        )

        processed = []
        for frame, entries in zip(frames, aligned):
            frame = frame.copy()
            entries = entries[:2]

            for i, entry in enumerate(entries):

                frame = add_subtitles(
                    frame,
                    display_text=entry.text_src_displayed,
                    full_text=entry.text_src_full,
                    location="top",
                    row=i,
                    speaker=entry.speaker,
                )

                frame = add_subtitles(
                    frame,
                    display_text=entry.text_tgt_displayed,
                    full_text=entry.text_tgt_full,
                    location="bottom",
                    row=i,
                    speaker=entry.speaker,
                )

                if entry.face_loc is not None:
                    frame = add_speaker_marker(frame, entry.face_loc, entry.speaker)

            processed.append(frame[..., ::-1])

        save_frames(processed, fps, path_video)
        combine_streams(path_video, path_audio, path_out)

    return path_out
