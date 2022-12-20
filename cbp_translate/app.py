""" Runs the app using Modal """

import shutil
import tempfile
from functools import partial
from logging import basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory as TempDir

import gradio as gr
import modal
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from cbp_translate.components.download import download
from cbp_translate.components.translation import LANGUAGES
from cbp_translate.modal_ import SHARED, cpu_image, stub, volume
from cbp_translate.pipeline import Config, run

basicConfig(level="INFO", format="%(asctime)s :: %(levelname)s :: %(message)s")

examples = Path(__file__).parent / "assets" / "videos"
logger = getLogger(__name__)
web_app = FastAPI()


def check_input(url: str, video: str) -> Path:
    """If a URL was provided, download it to a temporary directory.
    Otherwise, use the uploaded video file.
    """

    if url:
        tmp = Path(tempfile.mkdtemp())
        path_tmpl = tmp / "input"
        path_in = download(url, path_tmpl)
    elif video:
        path_in = Path(video)
    else:
        raise ValueError("Both URL and video are missing.")

    return path_in


def check_language(language: str) -> str:
    """Make sure the language was provided, and convert it into DeepL's key"""

    if not language:
        raise ValueError("Missing language selection")
    lang_key = LANGUAGES[language]

    return lang_key


def main(tempdir_root: str, url: str = "", video: str = "", language: str = ""):
    """Produce a processed video"""

    with TempDir(dir=tempdir_root) as shared_tmp:

        # Config
        shared_tmp = Path(shared_tmp)
        lang = check_language(language)
        config = Config(target_lang=lang, speaker_markers=True)

        # Download the video and move the file to a shared volume
        local_input = check_input(url, video)
        shared_input = shared_tmp / local_input.name
        shutil.move(local_input, shared_input)

        # Produce the output, storing it on a shared volume as well
        shared_output = run.call(
            path_in=str(shared_input),
            path_out=str(shared_tmp / "translated.mp4"),
            path_tmp=str(shared_tmp),
            config=config,
        )

        # Move the output back to the local filesystem
        local_output = Path(tempfile.mkdtemp()) / shared_output.name
        shutil.move(shared_output, local_output)

    # TODO: double-check that local temporary storage gets cleaned up properly
    return local_output


@stub.asgi(
    image=cpu_image,
    shared_volumes={str(SHARED): volume},
    mounts=[modal.Mount(local_dir=examples, remote_dir="/resources")],
    concurrency_limit=10,
)
def fastapi_app():

    tempdir_root = (SHARED / "tmp")
    tempdir_root.mkdir(exist_ok=True)

    interface = gr.Interface(
        fn=partial(main, str(tempdir_root)),
        title="AutoTranslate",
        inputs=[
            gr.Text(label="YouTube URL"),
            gr.Video(label="Video"),
            gr.Dropdown(list(LANGUAGES.keys()), label="Target Language"),
        ],
        examples=[
            ["", "/resources/keanu-reeves-interview.mp4", "Polish"],
            ["", "/resources/keanu-reeves-interview-short.mp4", "Polish"],
            ["", "/resources/dukaj-onet-interview.mp4", "English (British)"],
            ["", "/resources/dukaj-outdoor-interview.mp4", "English (British)"],
            ["", "/resources/political-interview-RMF.mp4", "English (British)"],
        ],
        outputs=gr.Video(),
    )

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )
