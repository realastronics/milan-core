"""
extract_landmarks.py — T-A3

Offline landmark extraction: converts INCLUDE video clips into the fixed 259-dim
per-frame feature vectors defined in `model/data/landmark_schema.md`, using the
official AI4Bharat label/split manifest (build_include_manifest.py) rather than
inferring labels from folder names.

This is one half of the model<->extension contract. The in-browser extraction
(extension/src/landmarks/) must produce byte-for-byte comparable output for the
same input frame — see landmark_schema.md §6 for the shared unit-test approach.

Setup (once):
    bash model/data/download_include.sh /data/milan/include/raw
    python model/data/build_include_manifest.py --output model/data/include_manifest.csv

Usage:
    python model/preprocessing/extract_landmarks.py \
        --raw_dir /data/milan/include/raw \
        --output_dir /data/milan/include/landmarks \
        --fps 20

Requires the MediaPipe Tasks model bundles (hand_landmarker.task, pose_landmarker.task,
face_landmarker.task) — downloaded automatically on first run, see download_models() below.

iSign support is intentionally not in this script yet — this phase targets INCLUDE only.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------- #
# Schema constants — must match model/data/landmark_schema.md exactly.
# If you change anything here, update that doc in the same PR.
# --------------------------------------------------------------------------- #

NUM_HAND_LANDMARKS = 21          # fixed MediaPipe Hand Landmarker topology
POSE_UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose, shoulders, elbows, wrists, hips
LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12                  # positions *within* the full 33-point pose output

# Eye/mouth: small set of individually well-established, stable Face Mesh landmark IDs
# (outer/inner corners and top/bottom lid or lip midpoints), not a large hand-typed list.
EYE_LEFT_INDICES = [33, 133, 159, 145]     # outer corner, inner corner, upper lid, lower lid
EYE_RIGHT_INDICES = [263, 362, 386, 374]
MOUTH_OUTER_INDICES = [61, 291, 0, 17, 37, 267, 84, 314]  # corners, top/bottom center, 2 upper-side, 2 lower-side

FACE_INDEX_CACHE = Path(__file__).resolve().parent.parent / "data" / "face_landmark_indices.json"

# NOTE: landmark_schema.md §2 quotes 259 as a planning estimate, with face-block sizes
# marked "~" specifically because eyebrow point count depends on what MediaPipe's own
# connection graph resolves to for the installed version. Rather than assert against a
# guessed constant (which would crash on the very first run if the true count differs),
# FEATURE_DIM is computed at runtime from the actual resolved indices — see
# LandmarkExtractor.__init__. Once this runs once, log the printed total back into
# landmark_schema.md §2 as the final, confirmed number.


def resolve_face_indices(cache_path: Path = FACE_INDEX_CACHE) -> dict:
    """
    Resolves eyebrow point indices from MediaPipe's own connection graph (so these are
    always correct for the installed MediaPipe version, never hand-typed), combines them
    with the stable eye/mouth corner IDs above, and caches the result — both this script
    and the in-browser extractor should read from the cached file once it exists, per
    landmark_schema.md §3.3.
    """
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    face_mesh_connections = mp.solutions.face_mesh

    def _points_from_connections(connections) -> list[int]:
        points = set()
        for a, b in connections:
            points.add(a)
            points.add(b)
        return sorted(points)

    resolved = {
        "eyebrow_left": _points_from_connections(face_mesh_connections.FACEMESH_LEFT_EYEBROW),
        "eyebrow_right": _points_from_connections(face_mesh_connections.FACEMESH_RIGHT_EYEBROW),
        "eye_left": EYE_LEFT_INDICES,
        "eye_right": EYE_RIGHT_INDICES,
        "mouth_outer": MOUTH_OUTER_INDICES,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(resolved, indent=2))
    print(f"Resolved face landmark indices, cached to {cache_path}")
    return resolved


# --------------------------------------------------------------------------- #
# Model asset download
# --------------------------------------------------------------------------- #

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}


def download_models(model_dir: Path) -> None:
    """Fetches the MediaPipe Tasks model bundles if not already present locally."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        dest = model_dir / filename
        if dest.exists():
            continue
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
    print(f"Model bundles ready in {model_dir}")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@dataclass
class NormalizationState:
    """Holds the last valid shoulder-based origin/scale, for fallback on frames
    where pose detection fails — see landmark_schema.md §4."""
    origin: np.ndarray | None = None
    scale: float | None = None

    def update(self, left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> None:
        self.origin = (left_shoulder + right_shoulder) / 2.0
        self.scale = float(np.linalg.norm(left_shoulder - right_shoulder))

    @property
    def is_valid(self) -> bool:
        return self.origin is not None and self.scale is not None and self.scale > 1e-6


class LandmarkExtractor:
    """Wraps the three MediaPipe Tasks detectors and produces one 259-dim vector
    per frame, laid out exactly per landmark_schema.md §2."""

    def __init__(self, model_dir: Path):
        self.face_indices = resolve_face_indices()
        self._norm_state = NormalizationState()

        n_eyebrows = len(self.face_indices["eyebrow_left"]) + len(self.face_indices["eyebrow_right"])
        n_mouth = len(self.face_indices["mouth_outer"])
        n_eyes = len(self.face_indices["eye_left"]) + len(self.face_indices["eye_right"])
        self.feature_dim = (
            2 * NUM_HAND_LANDMARKS * 3          # both hands
            + len(POSE_UPPER_BODY_INDICES) * 4  # pose (x,y,z,visibility)
            + n_eyebrows * 3 + n_mouth * 3 + n_eyes * 3
            + 3   # head pose (yaw, pitch, roll)
            + 4   # presence flags
        )
        print(
            f"Resolved feature vector size: {self.feature_dim} dims "
            f"(eyebrows={n_eyebrows}, mouth={n_mouth}, eyes={n_eyes}). "
            "If this differs from landmark_schema.md §2, update that doc's total in the same PR."
        )

        self.hand_detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task")),
                num_hands=2,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
        )
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_dir / "pose_landmarker.task")),
                running_mode=mp_vision.RunningMode.VIDEO,
            )
        )
        self.face_detector = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
                running_mode=mp_vision.RunningMode.VIDEO,
                output_facial_transformation_matrixes=True,
            )
        )

    def close(self) -> None:
        self.hand_detector.close()
        self.pose_detector.close()
        self.face_detector.close()

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

        left_hand, right_hand, left_present, right_present = self._extract_hands(hand_result)
        pose_points, pose_present = self._extract_pose(pose_result)
        eyebrows, mouth, eyes, head_pose, face_present = self._extract_face(face_result)

        # Normalization frame — see landmark_schema.md §4.
        if pose_present:
            shoulder_l = pose_points[POSE_UPPER_BODY_INDICES.index(LEFT_SHOULDER_IDX)][:3]
            shoulder_r = pose_points[POSE_UPPER_BODY_INDICES.index(RIGHT_SHOULDER_IDX)][:3]
            self._norm_state.update(shoulder_l, shoulder_r)

        origin, scale = self._current_norm_frame()

        left_hand = self._normalize_points(left_hand, origin, scale)
        right_hand = self._normalize_points(right_hand, origin, scale)
        pose_points = self._normalize_pose(pose_points, origin, scale)
        eyebrows = self._normalize_points(eyebrows, origin, scale)
        mouth = self._normalize_points(mouth, origin, scale)
        eyes = self._normalize_points(eyes, origin, scale)

        presence = np.array(
            [float(left_present), float(right_present), float(pose_present), float(face_present)],
            dtype=np.float32,
        )

        vector = np.concatenate(
            [
                left_hand.flatten(), right_hand.flatten(),
                pose_points.flatten(),
                eyebrows.flatten(), mouth.flatten(), eyes.flatten(),
                head_pose,
                presence,
            ]
        ).astype(np.float32)

        assert vector.shape[0] == self.feature_dim, (
            f"Feature vector is {vector.shape[0]} dims, expected {self.feature_dim} — "
            "a block's extraction logic drifted out of sync with its declared size."
        )
        return vector

    # -- extraction helpers ------------------------------------------------ #

    @staticmethod
    def _extract_hands(hand_result) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        left = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        right = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        left_present = right_present = False

        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            # MediaPipe reports handedness from the subject's perspective (mirrored view),
            # so "Left" in the result is the signer's left hand.
            if handedness[0].category_name == "Left":
                left, left_present = points, True
            else:
                right, right_present = points, True

        return left, right, left_present, right_present

    @staticmethod
    def _extract_pose(pose_result) -> tuple[np.ndarray, bool]:
        points = np.zeros((len(POSE_UPPER_BODY_INDICES), 4), dtype=np.float32)
        if not pose_result.pose_landmarks:
            return points, False

        full_pose = pose_result.pose_landmarks[0]
        for i, idx in enumerate(POSE_UPPER_BODY_INDICES):
            lm = full_pose[idx]
            points[i] = [lm.x, lm.y, lm.z, lm.visibility]
        return points, True

    def _extract_face(self, face_result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        n_brow_l, n_brow_r = len(self.face_indices["eyebrow_left"]), len(self.face_indices["eyebrow_right"])
        eyebrows = np.zeros((n_brow_l + n_brow_r, 3), dtype=np.float32)
        mouth = np.zeros((len(self.face_indices["mouth_outer"]), 3), dtype=np.float32)
        eyes = np.zeros((len(self.face_indices["eye_left"]) + len(self.face_indices["eye_right"]), 3), dtype=np.float32)
        head_pose = np.zeros(3, dtype=np.float32)

        if not face_result.face_landmarks:
            return eyebrows, mouth, eyes, head_pose, False

        landmarks = face_result.face_landmarks[0]

        def gather(indices) -> np.ndarray:
            return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices], dtype=np.float32)

        eyebrows = np.concatenate([gather(self.face_indices["eyebrow_left"]), gather(self.face_indices["eyebrow_right"])])
        mouth = gather(self.face_indices["mouth_outer"])
        eyes = np.concatenate([gather(self.face_indices["eye_left"]), gather(self.face_indices["eye_right"])])

        if face_result.facial_transformation_matrixes:
            head_pose = self._yaw_pitch_roll_from_matrix(face_result.facial_transformation_matrixes[0])

        return eyebrows, mouth, eyes, head_pose, True

    @staticmethod
    def _yaw_pitch_roll_from_matrix(matrix: np.ndarray) -> np.ndarray:
        """Extracts yaw/pitch/roll (radians) from MediaPipe's 4x4 facial transformation matrix."""
        r = np.asarray(matrix)[:3, :3]
        yaw = np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
        pitch = np.arctan2(r[2, 1], r[2, 2])
        roll = np.arctan2(r[1, 0], r[0, 0])
        return np.array([yaw, pitch, roll], dtype=np.float32)

    # -- normalization helpers ---------------------------------------------- #

    def _current_norm_frame(self) -> tuple[np.ndarray, float]:
        if self._norm_state.is_valid:
            return self._norm_state.origin, self._norm_state.scale
        # No valid shoulder frame yet (e.g. very first frame, pose not detected) —
        # fall back to an identity transform rather than crashing.
        return np.zeros(3, dtype=np.float32), 1.0

    @staticmethod
    def _normalize_points(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
        if points.size == 0:
            return points
        normalized = points.copy()
        normalized[:, :3] = (points[:, :3] - origin) / scale
        return normalized

    @classmethod
    def _normalize_pose(cls, points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
        # Pose has a 4th (visibility) column that must pass through untouched.
        normalized = points.copy()
        normalized[:, :3] = (points[:, :3] - origin) / scale
        return normalized


# --------------------------------------------------------------------------- #
# Clip-level processing and CLI
# --------------------------------------------------------------------------- #

def process_clip(video_path: Path, extractor: LandmarkExtractor, target_fps: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(1, round(source_fps / target_fps))

    vectors = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            timestamp_ms = int(frame_idx / source_fps * 1000)
            vectors.append(extractor.process_frame(frame, timestamp_ms))
        frame_idx += 1

    cap.release()
    return np.stack(vectors) if vectors else np.zeros((0, extractor.feature_dim), dtype=np.float32)


def save_clip(vectors: np.ndarray, output_dir: Path, clip_name: str, fps: int, label: str, split: str) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{clip_name}.npy", vectors)
    metadata = {
        "fps": fps,
        "source_dataset": "include",
        "label": label,
        "split": split,
        "num_frames": int(vectors.shape[0]),
    }
    (split_dir / f"{clip_name}.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract landmark feature vectors for the INCLUDE dataset, using the official "
            "AI4Bharat label/split manifest rather than inferring labels from folder names."
        )
    )
    parser.add_argument("--raw_dir", type=Path, required=True, help="Root of the unzipped raw INCLUDE videos")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to write .npy + .json pairs")
    parser.add_argument(
        "--manifest", type=Path, default=Path("model/data/include_manifest.csv"),
        help="Output of build_include_manifest.py — run that first if this doesn't exist yet",
    )
    parser.add_argument("--fps", type=int, default=20, help="Target extraction frame rate — see landmark_schema.md §5")
    parser.add_argument("--include_50_only", action="store_true", help="Only process the INCLUDE-50 subset (faster iteration)")
    parser.add_argument("--model_dir", type=Path, default=Path("model/assets"), help="Where the .task model bundles live")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"{args.manifest} not found. Run: python model/data/build_include_manifest.py --output {args.manifest}"
        )

    manifest = pd.read_csv(args.manifest)
    if args.include_50_only:
        manifest = manifest[manifest["include_50"]]

    download_models(args.model_dir)
    extractor = LandmarkExtractor(args.model_dir)

    print(f"Processing {len(manifest)} clips from manifest ({args.manifest})")
    missing = 0

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        video_path = args.raw_dir / row.video_path
        if not video_path.exists():
            missing += 1
            continue

        clip_name = video_path.stem
        vectors = process_clip(video_path, extractor, args.fps)
        save_clip(vectors, args.output_dir, clip_name, args.fps, row.label, row.split)

        if i % 100 == 0 or i == len(manifest):
            print(f"  [{i}/{len(manifest)}] {clip_name} ({row.split}/{row.label}) -> {vectors.shape[0]} frames")

    if missing:
        print(f"Warning: {missing} clips listed in the manifest were not found under {args.raw_dir} — check download_include.sh ran fully.")

    extractor.close()
    print(f"Done. Landmarks written to {args.output_dir}, split into train/val/test subfolders.")


if __name__ == "__main__":
    main()
