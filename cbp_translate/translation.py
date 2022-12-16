import os
import re
from dataclasses import dataclass
from pathlib import Path

import deepl

from .asr import SpeechSegment

LANGUAGES = {
    "Bulgarian": "BG",
    "Czech": "CS",
    "Danish": "DA",
    "German": "DE",
    "Greek": "EL",
    "English (British)": "EN-GB",
    "English (American)": "EN-US",
    "Spanish": "ES",
    "Estonian": "ET",
    "Finnish": "FI",
    "French": "FR",
    "Hungarian": "HU",
    "Indonesian": "ID",
    "Italian": "IT",
    "Japanese": "JA",
    "Lithuanian": "LT",
    "Latvian": "LV",
    "Dutch": "NL",
    "Polish": "PL",
    "Portuguese (Brazilian)": "PT-BR",
    "Portuguese (Other)": "PT-PT",
    "Romanian": "RO",
    "Russian": "RU",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Swedish": "SV",
    "Turkish": "TR",
    "Ukrainian": "UK",
    "Chinese (simplified)": "ZH",
}


@dataclass
class TranslatedSegment:
    start: float
    end: float
    text_src: str
    text_tgt: str


def deepl_key():
    return Path("~/.deepL/token.txt").expanduser().read_text().strip()


def translate(
    text: str,
    auth_key: str,
    preserve_formatting: bool = False,
    target_lang: str = "EN-GB",
) -> str:
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(
        text, target_lang=target_lang, preserve_formatting=preserve_formatting
    )

    assert not isinstance(result, list)
    return result.text


def remove_whitespace(text: str) -> str:
    """Remove double and trailing whitespace"""
    return " ".join(text.strip().split())


def split_sentences(text: str) -> list[str]:
    """Split the text into sentences using regex"""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    sentences = [remove_whitespace(s) for s in sentences]

    return sentences


def translate_segments(
    segments: list[SpeechSegment], target_lang: str = "EN-GB", auth_key: str = ""
) -> list[TranslatedSegment]:

    auth_key = auth_key or os.environ["DEEPL_KEY"]

    text_src = "\n".join([s.text_src for s in segments])
    text_tgt = translate(
        text=text_src,
        auth_key=auth_key,
        preserve_formatting=True,
        target_lang=target_lang,
    )

    translated = []
    for s, txt in zip(segments, text_tgt.splitlines()):
        translated.append(TranslatedSegment(s.start, s.end, s.text_src, txt))

    return translated
