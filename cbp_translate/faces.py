from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import distance as adst
from deepface.commons import functions
from numba import cuda
from retinaface import RetinaFace
from retinaface.commons import postprocess
from tensorflow.keras.preprocessing import image  # type: ignore

from . import Arr

FaceId = int


@dataclass
class FaceEmbedding:
    id_: "FaceId"
    embedding: Arr
    frequency: int = 1


class FaceLocation(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


class Face(NamedTuple):
    id_: FaceId
    location: FaceLocation


OnFrameFaces = list[Face]
FaceDB = dict[FaceId, FaceEmbedding]


def _process_face(img: Arr, target_size: tuple[int, int]):
    """Copied from DeepFace.commons.functions.preprocess_face
    We had to make minor tweaks for our use case.

    Credits: https://github.com/serengil/deepface/
    """

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        img = np.pad(
            img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),  # type: ignore
            "constant",  # type: ignore
        )

    # ------------------------------------------

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------

    # normalizing the image pixels

    img_pixels = image.img_to_array(img)  # what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    return img_pixels


def _extract_faces(
    img_path, threshold=0.9, model=None, align=True, allow_upscaling=True
):
    """Copied from retinaface.RetinaFace to return both the image and the facial area.

    Credits: https://github.com/serengil/retinaface
    """

    resp = []
    img = RetinaFace.get_image(img_path)
    obj = RetinaFace.detect_faces(
        img_path=img, threshold=threshold, model=model, allow_upscaling=allow_upscaling
    )

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]  # type: ignore

            facial_area = identity["facial_area"]
            facial_img = img[
                facial_area[1] : facial_area[3], facial_area[0] : facial_area[2]
            ]  # type: ignore

            if align == True:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                mouth_right = landmarks["mouth_right"]
                mouth_left = landmarks["mouth_left"]

                facial_img = postprocess.alignment_procedure(
                    facial_img, right_eye, left_eye, nose
                )

            facial_img = facial_img[:, :, ::-1]  # type: ignore
            resp.append((facial_img, facial_area))

    return resp


def add(database: dict[FaceId, FaceEmbedding], embedding: Arr):

    for id_, face in database.items():
        dist = adst.findCosineDistance(embedding, face.embedding)
        if dist < 0.68:
            face.frequency += 1
            face.embedding = (
                face.embedding * (face.frequency - 1) + embedding
            ) / face.frequency
            return id_
    else:
        id_ = len(database)
        database[id_] = FaceEmbedding(id_=id_, embedding=embedding)
        return id_


def assign_and_localize_faces(frames: list[Arr]) -> tuple[FaceDB, list[OnFrameFaces]]:

    database: FaceDB = {}
    faces: list[OnFrameFaces] = []

    model = DeepFace.build_model("ArcFace")
    retina = RetinaFace.build_model()
    target_size = functions.find_input_shape(model)

    for frame in frames:

        detected = _extract_faces(frame[..., ::-1], align=True, model=retina)
        on_frame = []

        for face_img, face_loc in detected:

            face_loc = FaceLocation(*face_loc)

            face_img = _process_face(face_img, target_size)
            face_img = functions.normalize_input(face_img, normalization="base")

            embedding = model.predict(face_img)[0]
            face_id = add(database, embedding)

            on_frame.append(Face(face_id, face_loc))

        faces.append(on_frame)

    del model, retina
    device = cuda.get_current_device()
    device.reset()

    return database, faces


def postprocess_faces(
    database: FaceDB, faces: list[OnFrameFaces]
) -> tuple[FaceDB, list[OnFrameFaces]]:

    database = {key: face for key, face in database.items() if face.frequency > 24}
    faces = [
        [face for face in frame_faces if face.id_ in database] for frame_faces in faces
    ]

    return database, faces


def extract_faces(frames: list[Arr]) -> tuple[FaceDB, list[OnFrameFaces]]:
    database, faces = assign_and_localize_faces(frames)
    database, faces = postprocess_faces(database, faces)

    return database, faces
