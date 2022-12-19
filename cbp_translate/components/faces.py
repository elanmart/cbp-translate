""" Face detection, embedding, clustering, and recognition."""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, NamedTuple, Optional

import cv2
import modal
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from cbp_translate.components.loaders import frame_iterator, get_video_metadata
from cbp_translate.modal_ import ROOT, Container, cpu_image, gpu_image, stub, volume

MTCNN = Any
FaceId = int
Array = np.ndarray
BgrImage = np.ndarray
RgbImage = np.ndarray
Embedding = np.ndarray


class FaceLocation(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class DetectedFace:
    location: FaceLocation
    image: RgbImage = field(default_factory=partial(np.array, [], dtype=np.uint8))
    embedding: Embedding = field(
        default_factory=partial(np.array, [], dtype=np.float64)
    )

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


@dataclass
class RecognizedFace:
    person_id: FaceId
    location: FaceLocation


OnFrameDetected = list[DetectedFace]
OnFrameRecognized = list[RecognizedFace]


def _process_face(img: RgbImage, target_size: tuple[int, int]):
    """Copied from DeepFace.commons.functions.preprocess_face
    We had to make minor tweaks for our use case.

    Credits: https://github.com/serengil/deepface/
    """

    from tensorflow.keras.preprocessing import image  # type: ignore

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


def _detect_faces(mtcnn: MTCNN, img: RgbImage) -> OnFrameDetected:
    """Copied from retinaface.RetinaFace to return both the image and the facial area.

    Credits: https://github.com/serengil/retinaface
    """

    # Local imports are needed for Modal
    from deepface.detectors import MtcnnWrapper

    # DeepFace expects BGR
    img = img[..., ::-1]
    det = MtcnnWrapper.detect_face(mtcnn, img, align=True)

    # And we map the results back to RGB
    det = [(face[..., ::-1], loc) for face, loc in det]
    faces = []

    for (img, (x, y, w, h)) in det:
        loc = FaceLocation(x, y, x + w, y + h)
        face = DetectedFace(location=loc, image=img)
        faces.append(face)

    return faces


def _filter_faces(
    faces: OnFrameDetected, top_k: int = 3, ratio: float = 3.0
) -> OnFrameDetected:
    """Basic filtering: we keep only faces which are roughly facing the camera, and then
    select top_k largest ones."""

    faces = [f for f in faces if f.height_ratio < ratio]
    faces = sorted(faces, key=lambda f: f.area, reverse=True)
    faces = faces[:top_k]

    return faces


class GetFaceEmbedding(Container):
    """Detect & calculate embeddings for all the faces visible on a single frame"""

    kwd = dict(
        image=gpu_image,
        gpu=False,
        memory=8000,
        cpu=1,
        shared_volumes={str(ROOT): volume},
        secret=modal.Secret({"DEEPFACE_HOME": str(ROOT)}),
    )

    def __enter__(self):
        """Load the models only once on container startup"""

        import tensorflow as tf

        assert len(tf.config.list_physical_devices("GPU")) == 1

        from deepface import DeepFace
        from deepface.commons import functions
        from deepface.detectors import MtcnnWrapper

        self.mtcnn = MtcnnWrapper.build_model()
        self.facenet = DeepFace.build_model("Facenet512")
        self.target_size = functions.find_input_shape(self.facenet)

    @stub.function(**kwd, concurrency_limit=1)
    def download(self):
        """Dummy function triggering the download to the shared storage"""
        pass

    @stub.function(**kwd, concurrency_limit=100)
    def f(self, frame: RgbImage, top_k: int = 3, ratio: float = 3.0):
        """Actual processing"""

        from deepface.commons import functions

        # Detect, align, filter out small faces / side views
        detected = _detect_faces(mtcnn=self.mtcnn, img=frame)
        detected = _filter_faces(detected, top_k=top_k, ratio=ratio)
        embedded = []

        for face in detected:

            # Get the embedding using another model
            face_img = _process_face(face.image, self.target_size)
            face_img = functions.normalize_input(face_img, normalization="Facenet")
            embedding = np.array(self.facenet.predict(face_img, verbose=0)[0])

            # This will be returned for clustering
            embedded.append(DetectedFace(face.location, embedding=embedding))

        # Done
        return embedded


def face_clustering(
    detected: list[OnFrameDetected],
    metric: str = "euclidean",
    linkage: str = "ward",
    threshold: Optional[float] = None,
) -> Array:
    """Cluster faces using AgglomerativeClustering."""

    # Fetch the embeddings
    embeddings = np.stack(
        [item.embedding for frame_sublist in detected for item in frame_sublist]
    )

    # The 0.68 threshold is based on the DeepFace recommended default for the ArcFace model
    if linkage == "ward":
        metric = "euclidean"

    if threshold is None:
        threshold = {"euclidean": 23.56, "cosine": 0.30}[metric]

    agc = AgglomerativeClustering(
        n_clusters=None, affinity=metric, linkage=linkage, distance_threshold=threshold  # type: ignore
    )
    preds = agc.fit_predict(embeddings)

    # Filter out faces that are not present in at least 50 frames (2 sec for a 25 fps video)
    # TODO: this could be improved by number of seconds, mean face size etc.
    inds, counts = np.unique(preds, return_counts=True)
    inds = inds[counts > 50]

    # Compute the mean embedding for each cluster
    # TODO: can you read it out fromt the agc tree?
    n_clusters = len(inds)
    centers = np.empty((n_clusters, embeddings.shape[1]))
    for i in range(n_clusters):
        index = inds[i]
        centers[i] = embeddings[preds == index].mean(axis=0)

    return centers


@stub.function(image=cpu_image)
def assign_face_ids(
    frame_faces: OnFrameDetected, cluster_centers: Array
) -> OnFrameRecognized:
    """Assigns a unique ID to each face in the video."""

    # Sort the faces by size
    frame_faces = sorted(frame_faces, key=lambda f: f.area, reverse=True)

    # Skip if nothing to do
    if len(frame_faces) == 0:
        return []

    # Calculate the cosine distance to each cluster center
    embeddings = np.stack([face.embedding for face in frame_faces])
    distances = pairwise_distances(
        embeddings, cluster_centers, metric="cosine", n_jobs=-1
    )
    annotated = []
    for idx, face in enumerate(frame_faces):

        # Assign the ID of the closest cluster center
        cluster_idx = np.argmin(distances[idx, :])
        recog = RecognizedFace(int(cluster_idx), face.location)
        annotated.append(recog)

        # Remove the ID from the list of available IDs
        distances[:, cluster_idx] = np.inf

    return annotated


def detect_faces(path_in: str) -> list[OnFrameDetected]:
    # Trigger model download
    obj = GetFaceEmbedding()
    _ = obj.download.call()

    # Process the video
    frames = frame_iterator(path_in)
    detected = list(obj.f.map(frames))

    return detected


def recognize_faces(detected: list[OnFrameDetected]):
    # Cluster the embeddings
    cluster_centers = face_clustering(detected, metric="cosine", linkage="average", threshold=0.3)

    # Assign a unique ID to each face
    recognized = assign_face_ids.map(
        detected, kwargs={"cluster_centers": cluster_centers}
    )

    # Trigger the computation
    recognized = list(recognized)

    # Done
    return recognized


def filter_flickering(
    recognized: list[OnFrameRecognized], fps: int, window: float = 1.0
) -> list[OnFrameRecognized]:
    """Filter out flickering faces."""

    ids: list[set[int]] = []
    for r in recognized:
        ids.append({f.person_id for f in r})

    filtered = []
    active = set()

    for i, items in enumerate(recognized):

        kept = []
        local_ids = set()

        for face in items:

            # Check if the face is on screen for the next `window` seconds
            # If it is, mark it as active, otherwise we ignore it
            if face.person_id not in active:

                last = min(i + int(window * fps) + 1, len(ids))
                for j in range(i + 1, last):
                    if face.person_id not in ids[j]:
                        break
                else:
                    active.add(face.person_id)

            # Keep only the active one, and also add a sanity check
            # To avoid two faces with the same ID (which can happen due to naive clustering)
            # Note that faces are ordered by size, so the first one is the largest
            # which makes it OK to ignore the other ones
            if (face.person_id in active) and (face.person_id not in local_ids):
                kept.append(face)
                local_ids.add(face.person_id)

            # De-activate missing faces
            active = {a for a in active if a in ids[i]}

        filtered.append(kept)

    # Re-map face IDs
    new_ids = {}
    for sublist in filtered:
        for face in sublist:
            if face.person_id not in new_ids:
                new_ids[face.person_id] = len(new_ids)
            face.person_id = new_ids[face.person_id]

    return filtered


@stub.function(
    image=cpu_image, memory=6000, shared_volumes={str(ROOT): volume}, timeout=30 * 60
)
def extract_faces(path_in: str) -> list[OnFrameRecognized]:
    fps, _, _ = get_video_metadata(path_in)
    detected = detect_faces(path_in)
    recognized = recognize_faces(detected)
    filtered = filter_flickering(recognized, fps=fps)

    return filtered
