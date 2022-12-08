from dataclasses import dataclass
from typing import cast

import whisper


@dataclass
class SpeechSegment:
    start: float
    end: float
    text_src: str


def extract_segments(path: str) -> list[SpeechSegment]:
    model = whisper.load_model("large", "cuda")
    result = whisper.transcribe(model, path)

    segments = []
    for s in result["segments"]:
        s = cast(dict, s)
        segments.append(SpeechSegment(s["start"], s["end"], s["text"]))

    return segments
