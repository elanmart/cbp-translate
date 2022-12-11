import tempfile
from logging import basicConfig, getLogger
from pathlib import Path

import gradio as gr

from cbp_translate.download import download
from cbp_translate.pipeline import Config, main
from cbp_translate.translation import LANGUAGES

basicConfig(level="INFO", format="%(asctime)s :: %(levelname)s :: %(message)s")
logger = getLogger(__name__)


def translate(url: str, video: str, lang_key: str):
    logger.info(f"Triggering translation for {url}")

    tmp = Path(tempfile.mkdtemp())
    config = Config(target_lang=LANGUAGES[lang_key])

    path_tmpl = tmp / "input"
    path_in = download(url, path_tmpl)

    path_out = tmp / "translated.mp4"
    path_out = main(str(path_in), str(path_out), config=config)

    return path_out


app = gr.Interface(
    fn=translate,
    title="AutoTranslate",
    inputs=[
        gr.Text(label="YouTube URL"),
        gr.Video(label="Video"),
        gr.Dropdown(list(LANGUAGES.keys()), label="Target Language"),
    ],
    outputs=gr.Video(),
)

app.launch()
