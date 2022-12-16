from dataclasses import dataclass
from typing import cast

from cbp_translate.modal_ import ROOT, stub, gpu_image, volume


@dataclass
class SpeechSegment:
    start: float
    end: float
    text_src: str


@stub.function(image=gpu_image, gpu=True, shared_volumes={str(ROOT): volume}, memory=12287, timeout=30 * 60)
def extract_segments(path: str) -> list[SpeechSegment]:

    import whisper

    model = whisper.load_model("large", device="cuda", download_root=str(ROOT / "whisper"))
    result = whisper.transcribe(model, path)

    segments = []
    for s in result["segments"]:
        s = cast(dict, s)
        segments.append(SpeechSegment(s["start"], s["end"], s["text"]))

    return segments
