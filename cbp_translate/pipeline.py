import re
from collections import deque
from pathlib import Path

import deepl
import numpy as np
import whisper
from unidecode import unidecode

from . import Arr
from .cv import add_text_to_frame
from .io import combine_streams, extract_audio, load_frames, save_frames


def deepl_key():
    return Path("~/.deepL/token.txt").expanduser().read_text().strip()


def translate(text: str, preserve_formatting: bool = False) -> str:
    translator = deepl.Translator(deepl_key())
    result = translator.translate_text(
        text, target_lang="EN-GB", preserve_formatting=preserve_formatting
    )

    assert not isinstance(result, list)
    return result.text


def remove_whitespace(text: str) -> str:
    """Remove double and trailing whitespace"""
    return " ".join(text.strip().split())


def split_sentences(text: str) -> list[str]:
    """Split the text into sentences using regex"""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    sentences = [remove_whitespace(s) for s in sentences]

    return sentences


def translate_segments(segments: list[dict]) -> list[dict]:
    text = "\n".join([s["text"] for s in segments])
    text_en = translate(text, preserve_formatting=True)
    for s, t in zip(segments, text_en.splitlines()):
        s["text_en"] = t

    return segments


def assign_timestamps(segments: list[dict], key: str) -> deque[tuple[str, int]]:
    chars = []

    for s in segments:
        text = s[key]
        start = s["start"]
        end = max(start + 0.05, s["end"] - 1.0)

        for i, c in enumerate(text):
            chars.append((c, start + i * (end - start) / len(text)))

    return deque(chars)


class Buffer:
    def __init__(self):
        self.chars = []
        self.stops = {".", "!", "?", ";"}
        self.clear = False

    def check(self, t: float, q: deque):
        if t > q[0][1]:
            self.add(q.popleft()[0])
        return q

    def add(self, c):

        if self.clear:
            self.chars = []
            self.clear = False

        self.chars.append(c)

        if c in self.stops:
            self.clear = True

    @property
    def text(self):
        return "".join(self.chars)


def annotate_frames(
    frames: list[Arr],
    fps: int,
    segments: list[dict],
    key: str,
    location: int,
) -> list[Arr]:

    result = []
    chars = assign_timestamps(segments, key)
    buffer = Buffer()

    for i, frame in enumerate(frames):
        t = i / fps
        chars = buffer.check(t, chars)
        frame = add_text_to_frame(frame, text=buffer.text, position=(10, location))
        result.append(frame)

    return result


def end_to_end():
    path_src = "../data/samples/keanu_dlc.webm"
    path_audio = "./maklowicz.mp3"
    path_video = "./maklowicz-frames.mp4"
    path_out = "./maklowicz.mp4"

    frames, fps = load_frames(path_src)
    path_audio = extract_audio(path_src, path_audio)
    model = whisper.load_model("large", "cuda")
    result = whisper.transcribe(model, path_audio)
    segments = result["segments"]
    segments = translate_segments(segments)
    new_frames = [frame.copy() for frame in frames]
    new_frames = annotate_frames(new_frames, fps, segments, "text", 100)
    new_frames = annotate_frames(new_frames, fps, segments, "text_en", 150)
    path_video = save_frames(new_frames, fps, path_video)
    path_out = combine_streams(
        path_video=path_video, path_audio=path_audio, path_out=path_out
    )
