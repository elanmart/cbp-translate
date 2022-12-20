""" Defines the Modal infrastructure """

from pathlib import Path
import modal


APT = ["ffmpeg", "libsndfile1", "git", "build-essential"]
CONDA = ["cudatoolkit=11.7", "cudnn", "cuda-nvcc"]
DOCKER = [
    "RUN python -m pip install --upgrade pip setuptools",
]

PIP_CPU = [
    "deepl==1.11.0",
    "ffmpeg-python==0.2.0",
    "gradio==3.12.0",
    "ipython==8.7.0",
    "matplotlib==3.6.2",
    "numba==0.56.4",
    "numpy==1.23.5",
    "opencv-python==4.6.0.66",
    "pandas==1.5.2",
    "scikit-learn==1.1.3",
    "scipy==1.8.1",
    "yt-dlp==2022.11.11",
    "Cython==0.29.32",
]

PIP_GPU = [
    "braceexpand==0.1.7",
    "deepface==0.0.75",
    "editdistance==0.6.2",
    "hydra-core==1.3.0",
    "nemo-asr==0.9.0",
    "nemo-toolkit==1.13.0",
    "omegaconf==2.2.3",
    "pytorch-lightning==1.7.7",
    "pyannote.core==5.0.0",
    "pyannote.metrics==3.2.1",
    "sentencepiece==0.1.97",
    "tensorflow==2.11.0",
    "torch==1.13.1",
    "torchaudio==0.13.1",
    "torchmetrics==0.10.3",
    "webdataset==0.2.31",
    "wget==3.2",
    "youtokentome==1.0.6",
    "https://github.com/openai/whisper/archive/0b5dcfdef7ec04250b76e13f1630e32b0935ce76.tar.gz",
]


class Container:
    pass


SHARED = Path("/shared")
stub = modal.Stub("video-translator")
hf_secret = modal.Secret.from_name("hf-access")
nemo_secret = modal.Secret(
    {
        "NEMO_ENV_CACHE_DIR": "/shared/.nemo_env_cache",
        "NEMO_CACHE_DIR": "/shared/.nemo_cache",
    }
)
deepl_secret = modal.Secret.from_name("deepl-access")
volume = modal.SharedVolume().persist("video-translator-volumne")
cpu_image = modal.Image.conda().apt_install(APT).pip_install(PIP_CPU)
gpu_image = (
    modal.Image.conda()
    .apt_install(APT)
    .dockerfile_commands(DOCKER)
    .pip_install(PIP_CPU)
    .pip_install(PIP_GPU)
)
