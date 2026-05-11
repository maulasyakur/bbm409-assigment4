import numpy as np
from pathlib import Path

def verify_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    
    landmarks = data["landmarks"]
    normalized = data["normalized"]
    frames = data["frame_indices"]
    label = str(data["label"])
    video = str(data["video_path"])

    T = landmarks.shape[0]
    nan_frames = np.isnan(landmarks).any(axis=1).sum()
    norm_min = np.nanmin(normalized)
    norm_max = np.nanmax(normalized)

    print(f"File     : {npz_path.name}")
    print(f"Label    : {label}")
    print(f"Video    : {video}")
    print(f"Shape    : {landmarks.shape}  (frames x features)")
    print(f"NaN frames: {nan_frames}/{T}")
    print(f"Norm range: [{norm_min:.3f}, {norm_max:.3f}]  ← should be ~[0, 1]")
    print(f"Frames   : {frames[0]} → {frames[-1]}")
    print()

# Check a few files across different word classes
output_root = Path("extracted_features")
for word_dir in sorted(output_root.iterdir())[:5]:       # first 5 words
    for npz in sorted(word_dir.glob("*.npz"))[:2]:       # first 2 files each
        verify_npz(npz)