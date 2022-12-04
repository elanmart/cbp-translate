import argparse
from pathlib import Path

import yaml
import yt_dlp


def download(url: str, path_out: Path):

    audio = {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
    kwd = dict(
        outtmpl=str(path_out), quiet=True, keepvideo=True, postprocessors=[audio]
    )

    with yt_dlp.YoutubeDL(kwd) as ydl:
        ydl.download([url])


def downlad_all(inputs: str, output: str):

    out = Path(output)
    out.mkdir(exist_ok=True)

    with open(inputs, "r") as f:
        clips = yaml.load(f, yaml.SafeLoader)

    for name, url in clips.items():
        path_out = out / f"{name}.%(ext)s"

        print(">>> Fetching {}".format(name))
        download(url=url, path_out=path_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    downlad_all(**args.__dict__)
