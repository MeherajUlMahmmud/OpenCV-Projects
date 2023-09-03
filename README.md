# Face and Smile Detection Repository

This repository contains Python scripts for face and smile detection using OpenCV and MediaPipe. There are five main scripts included:

1. [`face_detection.py`](face_detection.py): Detects faces in a live video stream and draws rectangles around them.

2. [`face_eye_detection.py`](face_eye_detection.py): Detects faces and eyes in a live video stream and draws rectangles around them.

3. [`face_eye_smile_detection.py`](face_eye_smile_detection.py): Detects faces, eyes, and smiles in a live video stream and draws rectangles around them.

4. [`face_smile_detection.py`](face_smile_detection.py): Detects faces and smiles in a live video stream and draws rectangles around them.

5. [`face_detection_with_mediapipe.py`](face_detection_with_mediapipe.py): Uses the MediaPipe library to perform face detection in a live video stream. It uses a different method than the other scripts, and it is more accurate. It detects faces with 468 landmarks.

## Prerequisites

Before running these scripts, ensure that you have the following dependencies installed:

-   Python 3.x
-   OpenCV
-   MediaPipe (for the [`face_detection_with_mediapipe.py`](face_detection_with_mediapipe.py) script)

You can install the required Python packages using `pip`:

```bash
pip install opencv-python mediapipe
```

## Usage

To use any of the scripts, simply run the corresponding Python file:

```bash
python face_detection.py
```

This will open a live video stream with face and/or smile detection, depending on the script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   This project uses OpenCV for computer vision tasks.
-   The [`face_detection_with_mediapipe.py`](face_detection_with_mediapipe.py) script utilizes the MediaPipe library for face detection.
