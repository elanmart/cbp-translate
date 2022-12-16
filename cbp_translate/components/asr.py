""" ASR - automatic speech recognition. """

from dataclasses import dataclass
from typing import cast

from cbp_translate.modal_ import ROOT, gpu_image, stub, volume


@dataclass
class SpeechSegment:
    start: float
    end: float
    text_src: str


@stub.function(
    image=gpu_image,
    gpu=True,
    shared_volumes={str(ROOT): volume},
    memory=12287,
    timeout=30 * 60,
)
def extract_segments(path: str) -> list[SpeechSegment]:
    """Runs Whisper over the provided audio file."""

    # Local imports are required for Modal
    import whisper

    # Note that we're downloading the model to a shared volume
    model = whisper.load_model(
        "large", device="cuda", download_root=str(ROOT / "whisper")
    )
    result = whisper.transcribe(model, path)

    # Extract the key information into human-readable objects
    segments = []
    for s in result["segments"]:
        s = cast(dict, s)
        segments.append(SpeechSegment(s["start"], s["end"], s["text"]))

    # Done
    return segments
