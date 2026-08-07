# Deepfake Detection Project

Detect deepfake videos by combining face cropping, frame sampling, and a PyTorch video classifier built with a ResNeXt50 backbone and an LSTM head.

## Model at a Glance

- **Input:** video clips
- **Preprocessing:** extract frames, detect faces with `face_recognition`, crop faces, and resize frames to 112 x 112
- **Backbone:** `ResNeXt50_32x4d` pretrained on ImageNet
- **Sequence model:** `LSTM`
- **Classifier:** fully connected layer for binary prediction (`REAL` / `FAKE`)
- **Output:** class label plus confidence score

## What This Repository Contains

- `Model Creation/` - notebooks and helper scripts for preprocessing, training, and prediction
- `Django Application/` - web app for uploading videos and running inference
- `labels/` - CSV labels used for training
- `how_to_run.txt` - setup and execution notes

## How the Pipeline Works

1. Videos are loaded from the dataset.
2. Frames are extracted from each video.
3. Faces are detected and cropped from the frames.
4. The cropped frame sequence is passed to the model.
5. The model predicts whether the video is real or fake.

## Training Summary

The training notebook uses preprocessed face-only videos and a label CSV. The model is trained with transfer learning using a pretrained ResNeXt50 CNN feature extractor and an LSTM to learn temporal patterns across frames.

## Prediction Summary

The prediction notebook and Django app load a saved `.pt` checkpoint, apply the same preprocessing, and run inference on new videos. The app also generates a heatmap-style visual explanation of the prediction.

## Datasets

The project references these datasets:

- FaceForensics++
- Celeb-DF
- Deepfake Detection Challenge (DFDC)

## Getting Started

If you want to run the project locally, start with:

1. Install the dependencies listed in `Django Application/requirements.txt`.
2. Follow the setup steps in `how_to_run.txt`.
3. Use the notebooks under `Model Creation/` to understand preprocessing, training, and prediction.
4. Launch the Django app inside `Django Application/` for web inference.

## Notes

- The model expects face-cropped video frames, not raw full-frame videos.
- The saved model checkpoints are `.pt` files.
- The notebooks were originally written for Google Colab, so some paths may need updating for local use.
