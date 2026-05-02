# AGENTS.md - Lip Movement Classification Assignment

## Critical Setup Issues

**MediaPipe 0.10.35+ uses the Tasks API** (not the old solutions API):
```python
# WRONG - will fail
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# CORRECT - use Tasks API
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import ImageFormat

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
```

**Model file required**: Download `face_landmarker.task` from:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

## Environment

- **Python 3.14**, venv activated at `venv/`
- **Key deps already present**: mediapipe, opencv-contrib-python, numpy, matplotlib, jupyter
- **Missing deps needed**: tslearn, scikit-learn, torch (install per method)

## Key References

- Dataset download link is in the PDF assignment file
- GAK tutorial: https://shota.io/2017/04/11/global-alignment-kernels.html
- tslearn GAK: https://tslearn.readthedocs.io/en/stable/user_guide/kernel.html
- tslearn Shapelets: https://tslearn.readthedocs.io/en/latest/user_guide/shapelets.html

## Required Methods (all need ablation studies)

1. **GAK + SVM** - use tslearn kernel, scikit-learn SVM
2. **Shapelet + MLP** - tslearn shapelet transform, PyTorch MLP
3. **LSTM** - PyTorch nn.LSTM
4. **Extra** - choose 1D-CNN, DTW-kNN, Rocket, or similar

## Execution Order

1. Download dataset (link in PDF)
2. Install missing deps: `pip install tslearn scikit-learn torch`
3. Extract landmarks from videos using MediaPipe
4. Segment by alignment files (ignore "sil" segments)
5. Implement 4 classification approaches with ablation
6. Compare methods, generate confusion matrices

## Submission

- `project_studentIDs.zip` (name/surname in notebook)
- Contents: `project.ipynb` + `project.py`
- **Do NOT include dataset**
- Due: May 25, 2026