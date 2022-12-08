from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import login  # type: ignore
from pyannote.audio import Pipeline
from pyannote.core import Annotation


@dataclass
class SpeakerSegment:
    id_: str
    start: float
    end: float


def hf_login():
    token = Path("~/.huggingface/token").expanduser().read_text().strip()
    login(token=token)


def extract_speakers(path_audio: str) -> list[SpeakerSegment]:
    hf_login()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@develop", use_auth_token=True  # type: ignore
    )
    dia = pipeline(path_audio)

    ret = []
    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        ret.append(SpeakerSegment(speaker, speech_turn.start, speech_turn.end))

    return ret
