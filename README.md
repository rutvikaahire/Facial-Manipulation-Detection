# Facial Manipulation Detection

A manipulated-face video classification pipeline built in Python and PyTorch, with a Django web app for running predictions. It follows an end-to-end workflow that includes dataset exploration, preprocessing, face extraction from video frames, feature extraction, model training, evaluation, and inference on unseen videos.

## Resume-Style Project Highlights

- Built a Django-based deepfake detection application that accepts uploaded videos, validates input, and runs inference to classify videos as real or fake.
- Developed an end-to-end PyTorch pipeline for video preprocessing, face extraction, sequence modeling, training, and evaluation using a ResNeXt50 + LSTM architecture.
- Worked with large-scale manipulation datasets, including FaceForensics++, Celeb-DF, and DFDC, to train and validate a binary classification model.
- Added inference support with prediction visualization so users can review model output and heatmap-style explanations.

## Project Summary

The project works with video data from FaceForensics++, Celeb-DF, and DFDC, which contain authentic and manipulated face sequences. The workflow starts with metadata inspection and sample analysis, then moves into face cropping from frames, training a sequence model, and finally running predictions with the saved checkpoint.

## Main Model Used

The core model is a PyTorch transfer-learning architecture built from:

- `ResNeXt50_32x4d` as the pretrained CNN backbone
- `LSTM` for learning frame-to-frame temporal patterns
- a final linear layer for binary classification

This model predicts whether a video is `REAL` or `FAKE` after processing face-cropped frame sequences.

## Workflow

1. Inspect the dataset structure, file types, class labels, and metadata.
2. Extract frames from each video and detect faces with `face_recognition`.
3. Crop the face region, resize frames to 112 x 112, and normalize the images.
4. Train the `ResNeXt50_32x4d + LSTM` model on the prepared sequences.
5. Evaluate the model with validation metrics and confusion-matrix style analysis.
6. Run inference on new videos and generate prediction visualizations.

## What the Project Includes

- video frame extraction and face cropping
- sequence modeling with a ResNeXt50 + LSTM architecture
- training and validation on labeled datasets
- inference for new videos through notebooks and the web app
- heatmap-style visual explanation outputs

## Model Training and Evaluation

The training notebook splits data into training and validation sets, loads labels from CSV, trains on GPU, tracks loss and accuracy, and saves the learned weights as a `.pt` file. It also includes confusion-matrix style evaluation to inspect correct and incorrect predictions.

## Prediction and Inference

The prediction notebooks and Django app load the saved model checkpoint and run inference on new videos using the same preprocessing pipeline. The code also generates a heatmap-style visualization to show which regions influenced the prediction.

## Datasets Used

- FaceForensics++
- Celeb-DF
- Deepfake Detection Challenge (DFDC)

## Repository Structure

- `Application/` - Django app, deployment files, and project-level overview
- `Model Creation/` - notebooks for preprocessing, training, and prediction
- `labels/` - metadata and training labels

## Recommended Runtime

Use [Google Colab](https://colab.research.google.com/) for the notebooks, because the preprocessing and training steps were written for GPU execution.

## Useful Files

- `Application/Django Application/requirements.txt` - application dependencies
- `Model Creation/Helpers/deepfake-starter-kit.ipynb` - exploratory data analysis and metadata inspection
- `Model Creation/preprocessing.ipynb` - frame extraction and face-only video creation
- `Model Creation/Model_and_train_csv.ipynb` - training pipeline and evaluation
- `Model Creation/Predict_final.ipynb` - inference and heatmap generation
- `labels/Gobal_metadata.csv` - training labels

## Getting Started

1. Install the dependencies listed in `Application/Django Application/requirements.txt`.
2. Use the notebooks in `Model Creation/` to review preprocessing, training, and prediction.
3. Launch the Django app inside `Application/Django Application/` for web inference.

## Notes

- The model expects face-cropped video frames, not raw full-frame videos.
- The saved model checkpoints are `.pt` files.
- The notebooks were originally written for Google Colab, so some paths may need updating for local use.
