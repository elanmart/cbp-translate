from typing import Iterable, Iterator, NamedTuple

import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from retinaface import RetinaFace
from retinaface.commons import postprocess
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tensorflow.keras.preprocessing import image  # type: ignore

from . import Arr

FaceId = int
Image = np.ndarray
Embedding = np.ndarray


class FaceLocation(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


class DetectedFace(NamedTuple):
    location: FaceLocation
    image: Image = np.array([])
    embedding: Embedding = np.array([])
    id_: FaceId = -1

    @property
    def area(self):
        return (self.location.x1 - self.location.x0) * (
            self.location.y1 - self.location.y0
        )

    @property
    def height_ratio(self):
        return (self.location.y1 - self.location.y0) / (
            self.location.x1 - self.location.x0
        )


OnFrameFaces = list[DetectedFace]


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


def _detect_faces(
    img_path, threshold=0.95, model=None, align=True, allow_upscaling=True
) -> OnFrameFaces:
    """Copied from retinaface.RetinaFace to return both the image and the facial area.

    Credits: https://github.com/serengil/retinaface
    """

    faces: OnFrameFaces = []

    img = RetinaFace.get_image(img_path)
    obj = RetinaFace.detect_faces(
        img_path=img, threshold=threshold, model=model, allow_upscaling=allow_upscaling
    )

    if type(obj) == dict:
        for key in obj:  # type: ignore
            identity = obj[key]  # type: ignore

            facial_area = identity["facial_area"]  # type: ignore
            facial_img = img[
                facial_area[1] : facial_area[3], facial_area[0] : facial_area[2]
            ]  # type: ignore

            if align == True:
                landmarks = identity["landmarks"]  # type: ignore
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]

                try:
                    # TODO: investigate why retinaface is failing here for some frames
                    facial_img = postprocess.alignment_procedure(
                        facial_img, right_eye, left_eye, nose
                    )
                except ValueError as e:
                    continue

            img = facial_img[:, :, ::-1]  # type: ignore
            loc = FaceLocation(*facial_area)
            face = DetectedFace(loc, image=img)

            faces.append(face)

    return faces


def _filter_faces(
    faces: OnFrameFaces, top_k: int = 3, ratio: float = 3.0
) -> OnFrameFaces:
    faces = [f for f in faces if f.height_ratio < ratio]
    faces = sorted(faces, key=lambda f: f.area, reverse=True)
    faces = faces[:top_k]

    return faces


def get_face_embeddings(
    frames: Iterable[Arr],
    top_k: int = 3,
    ratio: float = 3.0,
) -> Iterator[OnFrameFaces]:

    model = DeepFace.build_model("ArcFace")
    retina = RetinaFace.build_model()
    target_size = functions.find_input_shape(model)

    for frame in frames:

        embedded = []
        detected = _detect_faces(frame[..., ::-1], align=True, model=retina)
        detected = _filter_faces(detected, top_k=top_k, ratio=ratio)

        for face in detected:

            face_img = _process_face(face.image, target_size)
            face_img = functions.normalize_input(face_img, normalization="base")
            embedding = model.predict(face_img)[0]

            embedded.append(DetectedFace(face.location, embedding=embedding))

        yield embedded


def face_clustering(faces: Iterable[OnFrameFaces]) -> Arr:

    embeddings = [item.embedding for frame_faces in faces for item in frame_faces]
    embeddings = np.stack(embeddings)

    agc = AgglomerativeClustering(
        n_clusters=None, affinity="cosine", linkage="average", distance_threshold=0.68  # type: ignore
    )
    preds = agc.fit_predict(embeddings)
    inds, counts = np.unique(preds, return_counts=True)
    inds = inds[counts > 48]

    centers = np.empty((len(inds), embeddings.shape[1]))
    for i in range(agc.n_clusters_):
        index = inds[i]
        centers[i] = embeddings[preds == index].mean(axis=0)

    return centers


def assign_face_ids(
    faces: Iterable[OnFrameFaces], cluster_centers: Arr
) -> Iterator[OnFrameFaces]:

    for frame_faces in faces:

        frame_faces = sorted(frame_faces, key=lambda f: f.area, reverse=True)
        embeddings = np.stack([face.embedding for face in frame_faces])
        distances = pairwise_distances(embeddings, cluster_centers, metric="cosine")
        annotated = []

        for idx, face in enumerate(frame_faces):

            cluster_idx = np.argmin(distances[idx, :])
            distances[:, cluster_idx] = np.inf
            annotated.append(DetectedFace(face.location, id_=int(cluster_idx)))

        yield annotated


def extract_faces(frames: Iterable[Arr]) -> Iterator[OnFrameFaces]:

    embeddings = get_face_embeddings(frames)
    embeddings = list(embeddings)
    cluster_centers = face_clustering(embeddings)
    faces = assign_face_ids(embeddings, cluster_centers)

    yield from faces
