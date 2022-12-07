import re
from pathlib import Path

import deepl

from . import Arr
from .asr import SpeechSegment


def deepl_key():
    return Path("~/.deepL/token.txt").expanduser().read_text().strip()


def translate(
    text: str, preserve_formatting: bool = False, target_lang: str = "EN-GB"
) -> str:
    translator = deepl.Translator(deepl_key())
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
    segments: list[SpeechSegment], target_lang: str = "EN-GB"
) -> list[SpeechSegment]:
    text = "\n".join([s.text_src for s in segments])
    text_en = translate(text, preserve_formatting=True, target_lang=target_lang)
    for s, t in zip(segments, text_en.splitlines()):
        s.text_tgt = t

    return segments
