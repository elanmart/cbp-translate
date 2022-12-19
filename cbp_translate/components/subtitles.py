""" OpenCV / PIL code for adding subtitles to a video frame """

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Arr = np.ndarray

FONTS = {
    "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "lexi": "/usr/share/fonts/truetype/lexi/LexiGulim.ttf",
}

COLORS_RGB = [
    (52, 235, 64),  # Green
    (131, 52, 235),  # Purple
    (235, 64, 52),  # Red
    (52, 235, 214),  # Cyan
    (255, 255, 255),  # White
]


def add_subtitles(
    frame: Arr,
    display_text: str,
    full_text: str,
    speaker: int,
    location: str,
    row: int,
    border_h: int,
    picture_h: int,
    font_family: str = "dejavu",
):
    """Add subtitles to a frame

    Parameters
    ----------
    frame: np.ndarray
        The current frame
    display_text: str
        The text that should be displayed on the current frame
    full_text: str
        The full text of the subtitle
        We need this to properly size and center the display text
    speaker: int
        The speaker ID
        We need this to color the text
    location: str
        Either "top" or "bottom"
        This determines whether the subtitles are displayed at the top or bottom of the frame
    row: int
        The row number
        Substitles can land either on the first or second row (first is the bottom one)
    border_h: int
        The height of the added border
    picture_h: int
        The height of the original frame
    font_family: str
        The font family to use
    """

    frame_w = frame.shape[1]
    text_h = int(0.33 * border_h)
    offset = int(0.05 * border_h)

    # Prep the font and get the size of the full text in pixels
    font_path = FONTS[font_family]
    font = ImageFont.truetype(font_path, text_h)
    text_w = font.getlength(full_text)
    text_w = min(text_w, frame_w)

    # Get the y position, depending on the language (src / target)
    if location == "top":
        y = offset + text_h * row + offset * row
    elif location == "bottom":
        y = border_h + picture_h + offset + text_h * row + offset * row
    else:
        raise ValueError(location)

    color = COLORS_RGB[speaker]
    x = max(
        10,
        int(frame_w // 2 - text_w // 2),
    )
    pos = (x, y)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text(pos, display_text, color, font=font)
    return np.array(img)


def add_borders(frame: Arr, size: float = 0.1) -> tuple[Arr, int]:
    """Add black borders to the top and bottom of a frame. That's where the subtitles will be displayed."""

    height, width, _ = frame.shape
    border_h = int(round(size * height))
    border = np.zeros((border_h, width, 3), np.uint8)
    frame = np.concatenate((border, frame, border), axis=0)

    return frame, border_h


def add_speaker_marker(
    img: Arr,
    border_h: int,
    face_loc: tuple[int, int, int, int],
    speaker: int,
    alpha: float = 0.0,
):
    """Draw a colored rectangle around the speakers face"""

    color = COLORS_RGB[speaker]

    x0, y0, x1, y1 = face_loc
    y0, y1 = y0 + border_h, y1 + border_h

    shapes = np.zeros_like(img, np.uint8)
    cv2.rectangle(shapes, (x0, y0), (x1, y1), color, thickness=3, lineType=cv2.FILLED)

    out = img.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]

    return out
