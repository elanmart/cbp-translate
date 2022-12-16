""" Speaker diarization - detecting and annotating unique speakers. """

import os
from dataclasses import dataclass

from cbp_translate.modal_ import ROOT, gpu_image, hf_secret, stub, volume


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
    
    # Local imports are required for Modal
    from pyannote.audio import Pipeline

    # Note that we're downloading the model to a shared volume
    (cache_dir := (ROOT / ".hf")).mkdir(exist_ok=True)
    auth_token = os.environ["HUGGINGFACE_TOKEN"]

    # Run the pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1.1",
        use_auth_token=auth_token,
        cache_dir=cache_dir,
    )

    # Convert the results into a human-readable format
    ret = []
    dia = pipeline(path_audio)
    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        ret.append(SpeakerSegment(speaker, speech_turn.start, speech_turn.end))

    # Done
    return ret
