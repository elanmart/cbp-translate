import sys
import shutil
import tempfile
from logging import basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory as TempDir

import gradio as gr
import modal
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from cbp_translate.download import download
from cbp_translate.modal_ import ROOT, cpu_image, stub, volume
from cbp_translate.pipeline import Config, run
from cbp_translate.translation import LANGUAGES

basicConfig(level="INFO", format="%(asctime)s :: %(levelname)s :: %(message)s")

resources = Path(__file__).parent / ".resources"
logger = getLogger(__name__)
web_app = FastAPI()


def maybe_download(url: str, video: str) -> Path:
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


def main(url: str, video: str, language: str):
    """Produce a processed video"""

    # Create a temporary storage on a shared Modal volume
    (tmp := (ROOT / "tmp")).mkdir(exist_ok=True)
    with TempDir(dir=tmp) as shared_tmp:

        # Config
        shared_tmp = Path(shared_tmp)
        lang = check_language(language)
        config = Config(target_lang=lang, speaker_markers=True)

        # Download the video and move the file to a shared volume
        local_input = maybe_download(url, video)
        shared_input = shared_tmp / local_input.name
        shutil.move(local_input, shared_input)

        # Produce the output, storing it on a shared volume as well
        shared_output = run.call(
            path_in=str(shared_input),
            path_out=str(shared_tmp / "translated.mp4"),
            config=config,
        )

        # Move the output back to the local filesystem
        local_output = Path(tempfile.mkdtemp()) / shared_output.name
        shutil.move(shared_output, local_output)

    # TODO: double-check that local temporary storage gets cleaned up properly
    return local_output


@stub.function(
    image=cpu_image,
    shared_volumes={str(ROOT): volume},
    mounts=[modal.Mount("/resources", local_dir="./samples")],
    concurrency_limit=10,
    secret=modal.Secret({"DEEPFACE_HOME": str(ROOT)}),
    timeout=10_000,
)
def _main_():
    out = main("", "/resources/keanu-colbert-30s.mp4", "Polish")
    
    import sys
    print(">>> DONE", file=sys.stderr)

    with open(out, "rb") as f:
        return f.read()


if __name__ == "__main__":
    with stub.run():
        with open("samples/keanu-colbert-30s.translated.mp4", "wb") as f:
            f.write(_main_.call())


# @stub.asgi(
#     image=cpu_image,
#     shared_volumes={str(ROOT): volume},
#     mounts=[modal.Mount("/resources", local_dir=resources)],
#     concurrency_limit=10,
# )
# def fastapi_app():

#     interface = gr.Interface(
#         fn=main,
#         title="AutoTranslate",
#         inputs=[
#             gr.Text(label="YouTube URL"),
#             gr.Video(label="Video"),
#             gr.Dropdown(list(LANGUAGES.keys()), label="Target Language"),
#         ],
#         examples=[
#             ["", "/resources/foo.mp4", "Polish"],
#             ["", "/resources/bar.mp4", "English"],
#             ["", "/resources/baz.mp4", "English"],
#         ],
#         outputs=gr.Video(),
#     )

#     return mount_gradio_app(
#         app=web_app,
#         blocks=interface,
#         path="/",
#     )

# $7.52 - $8.17
