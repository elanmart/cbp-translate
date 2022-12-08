import cv2
import numpy as np
from unidecode import unidecode
import PIL
from PIL import Image, ImageDraw, ImageFont

from . import BGR, Arr

FONTS = {
    "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "lexi": "/usr/share/fonts/truetype/lexi/LexiGulim.ttf",
}

COLORS_RGB = [
    (52, 235, 64),  # Green
    (235, 64, 52),  # Red
    (131, 52, 235),  # Purple
    (52, 235, 214),  # Cyan
]

def add_subtitles(
    frame: Arr,
    display_text: str,
    full_text: str,
    speaker: int,
    location: str,
    row: int,
    border_size: float = 0.1,
    font_family: str = "dejavu",
):

    # Get basic dimensions and add the black borders for subtitles
    frame_h, frame_w, _ = frame.shape
    frame, border_h = add_borders(frame, size=border_size)
    text_h = int(0.425 * border_h)
    offset = int(0.05 * border_h)

    # Prep the font and get the size of the full text in pixels
    font_path = FONTS[font_family]
    font = ImageFont.truetype(font_path, text_h)
    text_w = font.getlength(full_text)

    # Get the y position, depending on the language (src / target)
    if location == "top":
        y = offset + text_h * row + offset * row
    elif location == "bottom":
        y = border_h + frame_h + offset + text_h * row + offset * row
    else:
        raise ValueError(location)

    color = COLORS_RGB[speaker]
    pos = (int((frame_w - text_w) / 2), y)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text(pos, display_text, color, font=font)
    return np.array(img)


def add_borders(frame: Arr, size: float = 0.1):

    # TODO: design a less hacky solution
    # Here we simply try to detect if the padding is already there
    if (frame[0, ...] != 0).any():
        height, width, _ = frame.shape
        subtitles_height = int(round(size * height))
        border = np.zeros((subtitles_height, width, 3), np.uint8)
        frame = np.concatenate((border, frame, border), axis=0)
    else:
        subtitles_height = np.nonzero(frame[:, 0, 0])[0].min()

    return frame, subtitles_height


def add_speaker_marker(
    img: Arr,
    face_loc: tuple[int, int, int, int],
    speaker: int,
    alpha: float = 0.5,
):

    color = COLORS_RGB[speaker]

    x0, y0, x1, y1 = face_loc
    face_h = y1 - y0

    x_center = int((x0 + x1) / 2)
    y_center = int(y0 - face_h * 0.25)
    radius = (x1 - x0) // 4
    
    shapes = np.zeros_like(img, np.uint8)
    cv2.circle(shapes, (x_center, y_center), radius, color, thickness=-1, lineType=cv2.FILLED)

    out = img.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]

    return out
