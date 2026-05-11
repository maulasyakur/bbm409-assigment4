"""
BBM409 Assignment 4 - Feature Extraction
Extracts lip landmarks from video files using MediaPipe Face Mesh
and saves word-level sequences as .npz files.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── MediaPipe setup ────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# MediaPipe Face Mesh lip landmark indices (inner + outer lips, 40 points total)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LIP_LANDMARKS = [
    # Outer upper lip
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Outer lower lip
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Inner upper lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # Inner lower lip
    324, 318, 402, 317, 14, 87, 178, 88, 95,
]
NUM_LANDMARKS = len(LIP_LANDMARKS)  # 40
FEATURE_DIM = NUM_LANDMARKS * 2     # x, y per landmark = 80 features


# ── Alignment parsing ──────────────────────────────────────────────────────────
def parse_align(align_path):
    """
    Parse a .align file into a list of (start_frame, end_frame, word) tuples.
    The raw numbers are in units of 1/1000 of a frame, so divide by 1000.
    Silent segments ('sil') are excluded.
    """
    segments = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, word = int(parts[0]), int(parts[1]), parts[2]
            if word == "sil":
                continue
            # Convert from 1/1000 units to actual frame indices
            start_frame = start // 1000
            end_frame = end // 1000
            segments.append((start_frame, end_frame, word))
    return segments


# ── Landmark extraction ────────────────────────────────────────────────────────
def extract_landmarks_from_video(video_path):
    """
    Run MediaPipe on every frame of the video.
    Returns an array of shape (num_frames, NUM_LANDMARKS * 2),
    and a list of frame indices where detection succeeded.
    Frames where no face is detected are filled with NaN.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    all_landmarks = []
    frame_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        # NEW (Tasks API)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect(mp_image)

        if result.face_landmarks:
            lms = result.face_landmarks[0]  # list of NormalizedLandmark
            coords = []
            for idx in LIP_LANDMARKS:
                coords.extend([lms[idx].x, lms[idx].y])
            all_landmarks.append(coords)
        else:
            # No face detected — fill with NaN so we can handle it later
            all_landmarks.append([np.nan] * FEATURE_DIM)

        frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    return np.array(all_landmarks, dtype=np.float32), np.array(frame_indices)


# ── Normalization ──────────────────────────────────────────────────────────────
def normalize_sequence(seq):
    """
    Normalize lip landmarks per frame relative to the mouth bounding box.
    This removes variation from head distance and position.

    seq: shape (T, NUM_LANDMARKS * 2)
    returns: normalized array of same shape
    """
    normalized = seq.copy()
    T = seq.shape[0]

    for t in range(T):
        frame = seq[t].reshape(NUM_LANDMARKS, 2)  # (40, 2)
        if np.any(np.isnan(frame)):
            continue  # leave NaN frames as-is

        x_coords, y_coords = frame[:, 0], frame[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        w = x_max - x_min if x_max > x_min else 1e-6
        h = y_max - y_min if y_max > y_min else 1e-6

        # Normalize to [0, 1] within the lip bounding box
        norm_x = (x_coords - x_min) / w
        norm_y = (y_coords - y_min) / h

        normalized[t] = np.stack([norm_x, norm_y], axis=1).reshape(-1)

    return normalized


# ── Main extraction loop ───────────────────────────────────────────────────────
def process_dataset(data_root, output_root):
    """
    Walk through the dataset and extract .npz files for every word segment.

    Expected structure:
        data_root/
          alignments/s1/*.align
          s1/*.mpg

    Output structure:
        output_root/
          <word>/
            <video_stem>_<start>_<end>.npz
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Find all speakers (s1, s2, ...)
    speakers = [d for d in (data_root).iterdir() if d.is_dir() and d.name.startswith("s")]

    total_saved = 0
    total_failed = 0

    for speaker_dir in sorted(speakers):
        speaker = speaker_dir.name
        align_dir = data_root / "alignments" / speaker
        video_dir = data_root / speaker

        video_files = sorted(video_dir.glob("*.mpg"))
        print(f"\n[{speaker}] Found {len(video_files)} videos")

        for video_path in video_files:
            stem = video_path.stem  # e.g. "bbaf2n"
            align_path = align_dir / f"{stem}.align"

            if not align_path.exists():
                print(f"  SKIP {stem}: no alignment file")
                continue

            # Parse word segments
            segments = parse_align(align_path)
            if not segments:
                continue

            # Extract landmarks for the whole video (once per video)
            try:
                landmarks, frame_indices = extract_landmarks_from_video(video_path)
            except Exception as e:
                print(f"  ERROR {stem}: {e}")
                total_failed += 1
                continue

            # Slice into word-level sequences and save
            for start_f, end_f, word in segments:
                # Clamp to valid frame range
                start_f = max(0, start_f)
                end_f = min(end_f, len(landmarks))

                if end_f <= start_f:
                    continue

                raw_seq = landmarks[start_f:end_f]          # (T, 80)
                norm_seq = normalize_sequence(raw_seq)       # (T, 80)
                seg_frames = frame_indices[start_f:end_f]   # (T,)

                # Save to output_root/<word>/<stem>_<start>_<end>.npz
                word_dir = output_root / word
                word_dir.mkdir(parents=True, exist_ok=True)
                out_path = word_dir / f"{stem}_{start_f}_{end_f}.npz"

                np.savez(
                    out_path,
                    landmarks=raw_seq,          # raw x,y coordinates
                    normalized=norm_seq,         # normalized sequence
                    frame_indices=seg_frames,    # original frame numbers
                    label=word,                  # word label (str)
                    video_path=str(video_path), # source video
                )
                total_saved += 1

            print(f"  {stem}: {len(segments)} words saved")

    print(f"\nDone. Saved {total_saved} sequences, {total_failed} videos failed.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Adjust these paths to match your setup
    DATA_ROOT = "archive/data"
    OUTPUT_ROOT = "extracted_features"

    process_dataset(DATA_ROOT, OUTPUT_ROOT)