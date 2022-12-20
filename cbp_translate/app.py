""" Runs the app using Modal """

import tempfile
from functools import partial
from logging import basicConfig, getLogger
from pathlib import Path
from typing import Optional

import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from modal.functions import FunctionCall

from cbp_translate.components.translation import LANGUAGES
from cbp_translate.modal_ import SHARED, cpu_image, stub, volume
from cbp_translate.pipeline import Config, run

basicConfig(level="INFO", format="%(asctime)s :: %(levelname)s :: %(message)s")

examples = Path(__file__).parent.parent / "assets" / "videos"
logger = getLogger(__name__)
web_app = FastAPI()


def check_language(language: str) -> str:
    """Make sure the language was provided, and convert it into DeepL's key"""

    if not language:
        raise ValueError("Missing language selection")
    lang_key = LANGUAGES[language]

    return lang_key


@stub.function(
    image=cpu_image,
    shared_volumes={str(SHARED): volume},
    timeout=30 * 60,
)
def main(
    storage: str, language: str = "", video: bytes = b"", suffix: str = ".mp4"
) -> bytes:
    """Produce a processed video"""

    # Create a dedicated directory for this run
    with tempfile.TemporaryDirectory(dir=storage) as tmp:

        # Configuration
        dirpath = Path(tmp)
        lang = check_language(language)
        config = Config(target_lang=lang, speaker_markers=True)

        # Download the video and move the file to a shared volume
        shared_input = (dirpath / "input").with_suffix(suffix)
        shared_input.write_bytes(video)

        # Produce the output
        shared_output: Path = run.call(
            path_in=str(shared_input),
            path_out=str(dirpath / "translated.mp4"),
            path_tmp=str(dirpath),
            config=config,
        )

        return shared_output.read_bytes()


def result(text: str = "", video: Optional[str] = None):
    return [text, video]


def submit(storage: str, job_id: str, video: str, language: str):
    """A helper utility to get around Modal's 45 sec timeout for @asgi apps."""

    if job_id:
        call = FunctionCall.from_id(job_id)
        try:
            output = call.get(timeout=0)
        except TimeoutError:
            return result(text="Not ready.")
        else:
            path = Path(tempfile.mkdtemp()) / "output.mp4"
            path.write_bytes(output)
            return result(text="Ready", video=str(path))

    else:

        path = Path(video)
        call = main.spawn(
            storage=storage,
            video=path.read_bytes(),
            suffix=path.suffix,
            language=language,
        )

        return result(text=call.object_id)


@stub.asgi(
    image=cpu_image,
    shared_volumes={str(SHARED): volume},
    concurrency_limit=10,
    # TODO: somehow Gradio refuses to use the examples mounted this way:
    #   mounts=[modal.Mount(local_dir=examples, remote_dir="/videos")],
)
def fastapi_app():

    tempdir_root = SHARED / "tmp"
    tempdir_root.mkdir(exist_ok=True)

    interface = gr.Interface(
        fn=partial(submit, str(tempdir_root)),
        title="AutoTranslate",
        inputs=[
            gr.Textbox(label="Existing Job ID"),
            gr.Video(label="Video"),
            gr.Dropdown(list(LANGUAGES.keys()), label="Target Language"),
        ],
        outputs=[
            gr.Textbox(label=" <--- Paste this into the box on the left"),
            gr.Video(label="Output"),
        ],
    )

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


if __name__ == "__main__":
    stub.serve()
