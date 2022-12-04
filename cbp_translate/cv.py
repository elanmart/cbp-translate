import cv2
import numpy as np

from unidecode import unidecode

from . import Arr, BGR


def add_transparent_rectangle(
    img: Arr,
    pos: tuple[int, int],
    size: tuple[int, int],
    alpha: float = 0.5,
    color: BGR = (0, 0, 0),
):

    shapes = np.zeros_like(img, np.uint8)
    cv2.rectangle(shapes, pos, size, color, cv2.FILLED)

    out = img.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]

    return out



def add_text_to_frame(
    frame: Arr,
    text: str,
    position: tuple[int, int],
    scale: int = 1,
    thickness: int = 2,
    color: BGR = (0, 255, 0),
    border: int = 10,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):

    text = unidecode(text)
    x, y = position
    text_x, text_y = x - border, y - border
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    text_pos = (text_x, text_y)
    text_size = (x + text_w + border * 2, y + text_h + border * 2)

    frame = add_transparent_rectangle(
        frame, pos=text_pos, size=text_size, color=(1, 1, 1), alpha=0.25
    )
    cv2.putText(
        img=frame,
        text=text,
        org=(x, y + text_h + scale - 1),
        fontFace=font,
        fontScale=scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return frame
