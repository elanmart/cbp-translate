""" Speaker diarization - detecting and annotating unique speakers. """
import json
import os
from dataclasses import dataclass
from pathlib import Path

import wget
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

from cbp_translate.modal_ import ROOT, gpu_image, hf_secret, stub, volume


@dataclass
class SpeakerSegment:
    id_: str
    start: float
    end: float


def combine_segments(speakers: list[SpeakerSegment]) -> list[SpeakerSegment]:

    ret = []
    s = speakers[0]
    id_, start, end = s.id_, s.start, s.end

    for s in speakers[1:]:

        if s.id_ != id_:
            ret.append(SpeakerSegment(id_, start, end))
            id_, start, end = s.id_, s.start, s.end
        else:
            end = s.end

    ret.append(SpeakerSegment(id_, start, end))

    return ret


def parse_nemo_output(path: str):
    dia = Path(path).read_text()
    lines = dia.splitlines()
    lines = [l.strip() for l in lines if len(l.strip()) > 1]
    ret = []
    for line in lines:
        _, _, _, t0, dur, _, _, ID, *_ = line.split()
        t0, dur = float(t0), float(dur)
        seg = SpeakerSegment(ID.capitalize(), start=t0, end=t0 + dur)
        ret.append(seg)

    return ret


@stub.function(
    image=gpu_image,
    gpu=True,
    shared_volumes={str(ROOT): volume},
    secret=hf_secret,
    timeout=30 * 60,
)
def extract_speakers(path: str, combine: bool = True) -> list[SpeakerSegment]:

    meta = {
        "audio_filepath": path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": 2,
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    manifest = Path(path).parent / "manifest.json"
    manifest.write_text(json.dumps(meta) + "\n")

    output_dir = Path(path).parent / "nemo-output"
    output_dir.mkdir(exist_ok=True)

    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml"
    MODEL_CONFIG = wget.download(config_url, str(output_dir))
    config = OmegaConf.load(MODEL_CONFIG)

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.num_workers = 1  # Workaround for multiprocessing hanging with ipython issue
    config.diarizer.manifest_filepath = str(manifest)
    config.diarizer.out_dir = output_dir

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [
        1.5,
        1.25,
        1.0,
        0.75,
        0.5,
    ]
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [
        0.75,
        0.625,
        0.5,
        0.375,
        0.1,
    ]
    config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05

    sd_model = ClusteringDiarizer(cfg=config)  # type: ignore
    sd_model.diarize()

    rttm = output_dir / "pred_rttms" / Path(path).with_suffix(".rttm").name
    parsed = parse_nemo_output(str(rttm))

    if combine:
        parsed = combine_segments(parsed)

    return parsed


@stub.function(
    image=gpu_image,
    gpu=True,
    shared_volumes={str(ROOT): volume},
    secret=hf_secret,
    timeout=30 * 60,
)
def extract_speakers_pyannote(path_audio: str) -> list[SpeakerSegment]:

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
