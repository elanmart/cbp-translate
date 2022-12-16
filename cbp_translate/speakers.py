import os
from dataclasses import dataclass

from .modal_ import ROOT, gpu_image, hf_secret, stub, volume


@dataclass
class SpeakerSegment:
    id_: str
    start: float
    end: float


@stub.function(
    image=gpu_image,
    gpu=True,
    shared_volumes={str(ROOT): volume},
    secret=hf_secret, 
    timeout=30 * 60,
)
def extract_speakers(path_audio: str) -> list[SpeakerSegment]:

    from pyannote.audio import Pipeline

    (cache_dir := (ROOT / ".hf")).mkdir(exist_ok=True)
    auth_token = os.environ["HUGGINGFACE_TOKEN"]
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1.1",
        use_auth_token=auth_token,
        cache_dir=cache_dir,
    )

    dia = pipeline(path_audio)
    ret = []

    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        ret.append(SpeakerSegment(speaker, speech_turn.start, speech_turn.end))

    return ret
