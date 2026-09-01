# Landmark Schema — T-A1

**This is the contract between Track A (model) and Track B (extension).** Both the offline training-data extraction (`model/preprocessing/extract_landmarks.py`) and the in-browser extraction (`extension/src/landmarks/`) must produce a vector matching this document *exactly*. If either side needs to deviate, this file changes first, in a PR both tracks review.

Target location in the repo: `model/data/landmark_schema.md`.

---

## 1. Source

All landmarks come from **MediaPipe Tasks Vision** (not the legacy Holistic solution, which isn't in the current Tasks API):

- **Hand Landmarker** — for both hands
- **Pose Landmarker** — for upper-body pose
- **Face Landmarker** — for a curated non-manual-marker subset (not the full 478-point mesh)

Running three models per frame is heavier than one, which is why every block below is deliberately trimmed to the minimum that carries signal, and why frame rate (§5) is the primary lever if profiling later shows this is too heavy.

---

## 2. Per-Frame Feature Vector

One frame produces a single flat `Float32Array` (JS) / `float32` NumPy row (Python) of **259 values**, laid out as fixed contiguous blocks in this order:

| Block | Points | Values/point | Size | Offset |
|---|---|---|---|---|
| Left hand | 21 | 3 (x, y, z) | 63 | 0–62 |
| Right hand | 21 | 3 (x, y, z) | 63 | 63–125 |
| Pose (upper body) | 9 | 4 (x, y, z, visibility) | 36 | 126–161 |
| Face — eyebrows | ~10 | 3 (x, y, z) | ~30 | 162–191 |
| Face — mouth (outer contour) | ~12 | 3 (x, y, z) | ~36 | 192–227 |
| Face — eyes (minimal) | 8 | 3 (x, y, z) | 24 | 228–251 |
| Head pose | — | yaw, pitch, roll | 3 | 252–254 |
| Presence flags | — | 4 binary | 4 | 255–258 |

**Total: 259 floats per frame.** At 20fps this is ~5.2 KB/sec of feature data before any model runs — trivial for both the training pipeline and the in-browser buffer.

Exact face-block point counts are marked `~` because they should be pulled programmatically from MediaPipe's named landmark groups (§3.3), not hand-typed — don't let anyone hardcode a guessed index list into training or extraction code.

---

## 3. Block Definitions

### 3.1 Hands (left, right — 63 values each)

Standard MediaPipe Hand Landmarker topology, 21 points per hand, fixed order:

```
0  WRIST
1  THUMB_CMC     2  THUMB_MCP     3  THUMB_IP      4  THUMB_TIP
5  INDEX_MCP     6  INDEX_PIP     7  INDEX_DIP     8  INDEX_TIP
9  MIDDLE_MCP   10  MIDDLE_PIP   11  MIDDLE_DIP   12  MIDDLE_TIP
13 RING_MCP     14  RING_PIP     15  RING_DIP     16  RING_TIP
17 PINKY_MCP    18  PINKY_PIP    19  PINKY_DIP    20  PINKY_TIP
```
Each point: `(x, y, z)`, post-normalization (§4). No visibility field — Hand Landmarker doesn't provide one; absence is instead captured by the presence flags (§3.6).

### 3.2 Pose — upper body only (36 values)

Standard MediaPipe Pose Landmarker (BlazePose) indices, using only the 9 upper-body points relevant to signing:

```
0  NOSE
11 LEFT_SHOULDER   12 RIGHT_SHOULDER
13 LEFT_ELBOW      14 RIGHT_ELBOW
15 LEFT_WRIST      16 RIGHT_WRIST
23 LEFT_HIP        24 RIGHT_HIP
```
Each point: `(x, y, z, visibility)`. Shoulders (11, 12) and hips (23, 24) are kept even though they're not directly expressive — shoulders anchor the normalization frame (§4), hips give a stable lower-torso reference that helps the model separate signing-space motion from whole-body sway.

### 3.3 Face — curated subset (~90 values total)

Pull point indices from MediaPipe's named face-mesh landmark groups rather than hardcoding numbers here — extract them programmatically in `extract_landmarks.py` (T-A3) from:

- `FACEMESH_LEFT_EYEBROW` / `FACEMESH_RIGHT_EYEBROW` → **eyebrows** block
- `FACEMESH_LIPS`, filtered to the **outer** contour only → **mouth** block
- `FACEMESH_LEFT_EYE` / `FACEMESH_RIGHT_EYE`, reduced to 4 points per eye (inner corner, outer corner, top-lid midpoint, bottom-lid midpoint) → **eyes** block

Whatever the resulting exact point counts are, they must be fixed once T-A3 is implemented and then treated as immutable — write the final resolved index list into `model/data/face_landmark_indices.json` at that point, and both extraction paths (offline Python, in-browser JS) read from the same resolved list. This doc's `~30 / ~36 / 24` sizing is the planning target, not the final source of truth once that file exists.

### 3.4 Head pose (3 values)

Not raw landmark coordinates — a derived `(yaw, pitch, roll)` triple, in radians, computed from MediaPipe Face Landmarker's built-in facial transformation matrix output (`output_facial_transformation_matrixes=True`). This is cheaper and more directly meaningful than asking the model to infer head tilt from raw points, and it's already computed internally by the Face Landmarker — no extra geometry code needed.

### 3.5 Presence flags (4 values)

Binary (`0.0` / `1.0`), in order: `left_hand_present`, `right_hand_present`, `pose_present`, `face_present`. Whenever a given model fails to detect its target in a frame (hand out of frame, face turned away, etc.), that block's values are filled with `0.0` and its presence flag is set to `0`. This makes "missing" explicit and learnable, rather than indistinguishable from "detected at the origin."

---

## 4. Normalization — body-relative, shoulder-width scaled

Applied identically to every block (hands, pose, face) so the whole frame lives in one consistent, signer- and camera-distance-invariant space:

1. **Origin**: the midpoint of `LEFT_SHOULDER` (pose index 11) and `RIGHT_SHOULDER` (pose index 12). Subtract this point from every `(x, y, z)` in the frame.
2. **Scale**: the Euclidean distance between `LEFT_SHOULDER` and `RIGHT_SHOULDER` in the original (pre-normalization) frame. Divide every coordinate by this distance, so shoulder width = `1.0` unit after normalization.
3. Rotation is **not** normalized in v1 — coordinates stay in the camera's original orientation, just re-centered and rescaled. This is a deliberate scope cut (see §7); revisit only if off-axis camera angles turn out to hurt accuracy.

If shoulder landmarks are themselves missing on a given frame (pose not detected), the whole frame's normalization falls back to the *previous valid frame's* scale/origin rather than an arbitrary default — implement this as part of T-A3/T-B3, not left ad hoc per-caller.

---

## 5. Frame Rate

**20 fps.** Landmark extraction should run at this rate on both sides — this is the number the chunk buffer (T-B4) and training-data windowing are built against. If later profiling (T-B10/T-B11) shows this is too heavy on modest hardware, this section is the one to revisit first, since frame rate is the cheapest lever on both CPU load and per-second data volume.

---

## 6. Serialization

**Offline (training data, Python):** one file per clip, `float32` NumPy array of shape `(num_frames, 259)`, saved as `.npy`, plus a small JSON sidecar with the same base filename holding: `fps`, `source_dataset` (`"include"` / `"isign"`), `signer_id` (if available), and the label (isolated sign name, or sentence/gloss for iSign).

**Runtime (in-browser, JS):** the same 259-value layout as a `Float32Array` per frame, pushed into the chunk buffer (T-B4) in order. No metadata sidecar needed at runtime — the buffer only ever needs the raw vectors plus a timestamp per frame for chunk windowing.

Both sides must agree on **column order** exactly as laid out in §2 — a training/runtime mismatch here fails silently (the model runs, just on scrambled input) rather than loudly, so this is worth a shared unit test early: same synthetic frame, extracted through both the Python and JS paths, byte-for-byte comparable.

---

## 7. Explicitly Deferred (not in v1)

- Rotation-invariant normalization (only translation + scale for now, §4)
- Iris/gaze tracking
- Legs/lower-body pose
- Two-hand-overlap / occlusion-specific handling beyond the presence flags
- Blendshape-coefficient features (MediaPipe can output ~52 ARKit-style blendshapes, e.g. `browUpLeft`, `mouthOpen`, directly — a plausible lighter-weight alternative to raw face-point coordinates. Worth a look if the face block ever needs to shrink further, but not adopted now since it changes what the model is learning from, not just how much data.)

---

## 8. Open Item Before This Is Final

Face-block exact point indices (§3.3) are not yet resolved to concrete numbers — that happens as part of implementing T-A3, and the result should be committed to `model/data/face_landmark_indices.json`. Until that file exists, treat the `~30/~36/24` sizing here as the target, not a guarantee.
