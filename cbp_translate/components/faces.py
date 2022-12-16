from typing import Iterator, NamedTuple

import cv2
import modal
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from cbp_translate.components.loaders import frame_iterator
from cbp_translate.modal_ import ROOT, cpu_image, gpu_image, stub, volume

Arr = np.ndarray
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


class RecognizedFace(NamedTuple):
    person_id: FaceId
    location: FaceLocation


OnFrameDetected = list[DetectedFace]
OnFrameRecognized = list[RecognizedFace]


def _process_face(img: Arr, target_size: tuple[int, int]):
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


def _detect_faces(
    img_path, threshold=0.95, model=None, align=True, allow_upscaling=True
) -> OnFrameDetected:
    """Copied from retinaface.RetinaFace to return both the image and the facial area.

    Credits: https://github.com/serengil/retinaface
    """

    from retinaface import RetinaFace
    from retinaface.commons import postprocess

    faces: OnFrameDetected = []

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
    faces: OnFrameDetected, top_k: int = 3, ratio: float = 3.0
) -> OnFrameDetected:
    """Basic filtering: we keep only faces which are roughly facing the camera, and then
    select top_k largest ones."""

    faces = [f for f in faces if f.height_ratio < ratio]
    faces = sorted(faces, key=lambda f: f.area, reverse=True)
    faces = faces[:top_k]

    return faces


class GetFaceEmbedding:
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

        import tensorflow as tf

        assert len(tf.config.list_physical_devices("GPU")) == 0

        from deepface import DeepFace
        from deepface.commons import functions
        from retinaface import RetinaFace

        self.retina = RetinaFace.build_model()
        self.model = DeepFace.build_model("ArcFace")
        self.target_size = functions.find_input_shape(self.model)

    @stub.function(**kwd, concurrency_limit=1)
    def download(self):
        """Dummy function triggering the download to the shared storage"""
        pass

    @stub.function(**kwd, concurrency_limit=100)
    def f(self, frame: Arr, top_k: int = 3, ratio: float = 3.0):

        from deepface.commons import functions

        detected = _detect_faces(frame[..., ::-1], align=True, model=self.retina)
        detected = _filter_faces(detected, top_k=top_k, ratio=ratio)
        embedded = []

        for face in detected:

            face_img = _process_face(face.image, self.target_size)
            face_img = functions.normalize_input(face_img, normalization="base")
            embedding = np.array(self.model.predict(face_img)[0])

            embedded.append(DetectedFace(face.location, embedding=embedding))

        return embedded


def face_clustering(detected: list[OnFrameDetected]) -> Arr:
    """Cluster faces using AgglomerativeClustering."""

    # Fetch the embeddings
    embeddings = np.stack(
        [item.embedding for frame_sublist in detected for item in frame_sublist]
    )

    # The 0.68 threshold is based on the DeepFace recommended default for the ArcFace model
    agc = AgglomerativeClustering(
        n_clusters=None, affinity="cosine", linkage="average", distance_threshold=0.68  # type: ignore
    )
    preds = agc.fit_predict(embeddings)

    # Filter out faces that are not present in at least 50 frames (2 sec for a 25 fps video)
    # TODO: this could be improved by number of seconds, mean face size etc.
    inds, counts = np.unique(preds, return_counts=True)
    inds = inds[counts > 50]

    # Compute the mean embedding for each cluster
    centers = np.empty((len(inds), embeddings.shape[1]))
    for i in range(agc.n_clusters_):
        index = inds[i]
        centers[i] = embeddings[preds == index].mean(axis=0)

    return centers


@stub.function(image=cpu_image)
def assign_face_ids(
    frame_faces: OnFrameDetected, cluster_centers: Arr
) -> OnFrameRecognized:
    """Assigns a unique ID to each face in the video."""

    frame_faces = sorted(frame_faces, key=lambda f: f.area, reverse=True)
    embeddings = np.stack([face.embedding for face in frame_faces])
    distances = pairwise_distances(
        embeddings, cluster_centers, metric="cosine", n_jobs=-1
    )
    annotated = []

    for idx, face in enumerate(frame_faces):

        cluster_idx = np.argmin(distances[idx, :])
        distances[:, cluster_idx] = np.inf
        recog = RecognizedFace(int(cluster_idx), face.location)
        annotated.append(recog)

    return annotated


@stub.function(
    image=cpu_image, memory=6000, shared_volumes={str(ROOT): volume}, timeout=30 * 60
)
def extract_faces(path_in: str) -> list[OnFrameRecognized]:
    """For each frame, detect faces and extract their embeddings.
    Then cluster the embeddings, and assign a unique id to each face in the video stream.
    """

    import sys

    obj = GetFaceEmbedding()
    _ = obj.download.call()

    frames = frame_iterator(path_in)
    detected = list(obj.f.map(frames))
    cluster_centers = face_clustering(detected)
    recognized = assign_face_ids.map(
        detected, kwargs={"cluster_centers": cluster_centers}
    )
    recognized = list(recognized)

    return recognized
