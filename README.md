# Lip Movement Classification Project

This project implements multiple classification approaches for lip movement recognition from video data, as part of BBM409 Assignment 4.

## Dataset Setup

1. **Download the dataset** from Kaggle:
   https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset/data

2. **Extract the zip file** in the project root directory. This should create an `archive/` folder with the following structure:
   ```
   archive/
   └── data/
       ├── alignments/
       │   ├── s1/   # word-level alignment files (.align)
       │   ├── s2/
       │   └── ...
       ├── s1/       # speaker 1 videos (.mpg)
       ├── s2/
       └── ...
   ```

> **Note:** Do not commit the `archive/` folder or `extracted_features/` to Git. They are listed in `.gitignore`.

## Project Structure

```
./
├── archive/                     # Extracted dataset (not committed)
├── extracted_features/          # Output of feature extraction (not committed)
├── src/                         # Python source codes folder
│   ├── extract_landmarks.py     # Extracts lip landmarks from videos → .npz files
│   ├── npz_check_visual.py      # Checks npz file structure visually
│   └── npz_check.py             # Checks npz file structure
├── report.ipynb                 # Main notebook (code + report)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## Environment Setup

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate    # Windows
   source venv/bin/activate   # Linux/Mac
   ```

2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install tslearn scikit-learn torch
   ```

## MediaPipe Model

Download the required MediaPipe face landmarker model and place it in the project root:

```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

> **Note:** Do not commit `face_landmarker.task` to Git — it is listed in `.gitignore`.

## Running the Project

**Step 1 — Extract lip landmarks from all videos:**

```bash
python feature_extraction.py
```

This processes all `.mpg` files, extracts mouth landmark sequences using MediaPipe, and saves one `.npz` file per word segment under `extracted_features/<word>/`.

**Step 2 — Open the main notebook:**

```bash
jupyter notebook report.ipynb
```

## Dependencies

- Python 3.11+
- mediapipe >= 0.10
- opencv-contrib-python
- numpy
- matplotlib
- jupyter
- tslearn
- scikit-learn
- torch
