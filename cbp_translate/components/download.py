import re
from logging import getLogger
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.utils import download_range_func

logger = getLogger(__name__)


def parse_yt_url(url: str) -> tuple[str, int]:

    if urlparse(url).netloc not in {"youtu.be", "youtube.com", "www.youtube.com"}:
        raise ValueError("Not a YouTube URL")

    pattern = r"[\?&]?t=(\d+)[s]?"
    timestamp = re.search(pattern, url)
    if timestamp:
        timestamp = int(timestamp.group(1))
        url = re.sub(pattern, "", url)
    else:
        timestamp = 0

    return url, timestamp


def download(url: str, path_template: Path, time_limit: int = 60):

    url, timestamp = parse_yt_url(url)

    download_range = {}
    if time_limit > 0:
        download_range = {
            "download_ranges": download_range_func(
                None, [(timestamp, timestamp + time_limit)]
            ),
        }

    kwd = dict(
        outtmpl=str(path_template) + ".%(ext)s",
        quiet=True,
        keepvideo=False,
        **download_range,
    )

    with yt_dlp.YoutubeDL(kwd) as ydl:
        logger.info(
            f"Downloading {url} to {path_template} (range: {timestamp} - {timestamp + time_limit})"
        )
        ydl.download([url])

    (path_out,) = path_template.parent.glob(f"{path_template.name}.*")
    return path_out
