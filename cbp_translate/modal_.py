from pathlib import Path

import modal

APT = ["ffmpeg", "git", "build-essential"]
CONDA = ["cudatoolkit=11.7", "cudnn", "cuda-nvcc"]
DOCKER = [
    "RUN python -m pip install --upgrade pip",
    "RUN python -m pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu117",
]

PIP_CPU = [
    "deepl==1.11.0",
    "ffmpeg-python==0.2.0",
    "gradio==3.12.0",
    "matplotlib==3.6.2",
    "numba==0.56.4",
    "numpy==1.23.5",
    "opencv-python==4.6.0.66",
    "pandas==1.5.2",
    "scikit-learn==1.1.3",
    "scipy==1.8.1",
    "seaborn==0.12.1",
    "yt-dlp==2022.11.11",
]

PIP_GPU = [
    "deepface==0.0.75",
    "huggingface-hub==0.11.1",
    "pyannote.audio==2.1.1",
    "pyannote.core==4.5",
    "pyannote.database==4.1.3",
    "pyannote.metrics==3.2.1",
    "pyannote.pipeline==2.3",
    "retina-face==0.0.12",
    "tensorflow==2.11.0",
    "https://github.com/openai/whisper/archive/0b5dcfdef7ec04250b76e13f1630e32b0935ce76.tar.gz",
]

ROOT = Path("/shared")

stub = modal.Stub("video-translator")
hf_secret = modal.Secret.from_name("hf-access")
deepl_secret = modal.Secret.from_name("deepl-access")
volume = modal.SharedVolume().persist("video-translator-volumne")
cpu_image = modal.Image.conda().apt_install(APT).pip_install(PIP_CPU)
gpu_image = (
    modal.Image.conda()
    .apt_install(APT)
    .dockerfile_commands(DOCKER)
    .pip_install(PIP_CPU + PIP_GPU)
)
