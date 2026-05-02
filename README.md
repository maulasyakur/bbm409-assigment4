# Lip Movement Classification Project

This project implements multiple classification approaches for lip movement recognition from video data.

## Dataset Setup

1. **Download the dataset** from Kaggle:
   https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset/data

2. **Extract the zip file** in the project root directory
   - This should create an `archive/` folder with the following structure:
   ```
   archive/
   ├── data/
   │   ├── s1/  (speaker 1 videos)
   │   ├── s2/  (speaker 2 videos)
   │   ├── s3/  (speaker 3 videos)
   │   └── ...
   ```

## Project Structure

```
.
├── archive/              # Extracted dataset
│   └── data/
│       └── s1/          # Speaker subdirectories (s1, s2, ...)
├── face_landmarker.task # MediaPipe model file
├── requirements.txt    # Python dependencies
├── test_mediapipe.py   # MediaPipe test script
├── report.ipynb        # Jupyter notebook for analysis
├── venv/               # Python virtual environment
└── README.md           # This file
```

## Environment Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies:**
   ```bash
   pip install tslearn scikit-learn torch
   ```

## MediaPipe Model

Download the required MediaPipe face landmarker model:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Place the downloaded file in the project root directory as `face_landmarker.task`.

## Running the Project

The main analysis is in `report.ipynb`. Open with Jupyter:
```bash
jupyter notebook report.ipynb
```

## Dependencies

- Python 3.14
- mediapipe
- opencv-contrib-python
- numpy
- matplotlib
- jupyter
- tslearn
- scikit-learn
- torch