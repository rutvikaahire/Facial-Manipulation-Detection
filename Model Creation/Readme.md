# Model Creation

This project is a manipulated-face video classification pipeline built in Python and PyTorch. It follows an end-to-end workflow for analyzing video datasets from FaceForensics++, Celeb-DF, and DFDC, inspecting metadata, extracting face crops from frames, training a sequence model, and running predictions on unseen videos.

## Main Model Used

The core model in this project is a PyTorch transfer-learning architecture built from:

- `ResNeXt50_32x4d` as the pretrained CNN backbone
- `LSTM` for learning frame-to-frame temporal patterns
- a final linear layer for binary classification

This model predicts whether a video is `REAL` or `FAKE` after processing face-cropped frame sequences.

## What I Did

### 1. Data Exploration

I inspected the dataset structure, file types, class labels, and metadata before training. The notebooks also check missing values, class counts, and sample distributions so the data can be prepared in a way that is suitable for modeling.

### 2. Data Preprocessing

I converted raw videos into model-ready inputs by:

- extracting frames from each video
- detecting faces with `face_recognition`
- cropping the face region
- resizing frames to 112 x 112
- normalizing the images for PyTorch training

This step turns unstructured video data into a structured machine learning dataset.

### 3. Machine Learning Model

The training notebook uses a PyTorch transfer learning architecture:

- `ResNeXt50_32x4d` as the pretrained CNN backbone
- `LSTM` for temporal sequence learning across frames
- a final linear layer for binary classification

This setup learns both spatial features from faces and sequence patterns across video frames.

### 4. Model Training and Evaluation

The training notebook splits data into training and validation sets, loads the labels from CSV, trains on GPU, tracks loss and accuracy, and saves the learned weights as a `.pt` file. It also includes confusion-matrix style evaluation to inspect correct and incorrect predictions.

### 5. Prediction and Inference

The prediction notebooks load the saved model checkpoint and run inference on new videos using the same preprocessing pipeline. The code also generates a heatmap-style visualization to show which regions influenced the prediction.

## Recommended Runtime

Use [Google Colab](https://colab.research.google.com/) for the notebooks, because the preprocessing and training steps were written for GPU execution.

## Useful Files

- `Helpers/deepfake-starter-kit.ipynb` - exploratory data analysis and metadata inspection
- `preprocessing.ipynb` - frame extraction and face-only video creation
- `Model_and_train_csv.ipynb` - training pipeline and evaluation
- `Predict_final.ipynb` - inference and heatmap generation
- `labels/Gobal_metadata.csv` - training labels

## Datasets Used

- FaceForensics++
- Celeb-DF
- Deepfake Detection Challenge (DFDC)

## Bottom Line

The project covers the full machine learning workflow: exploring the data, preparing it, training the model, evaluating it, and using the saved checkpoint for prediction.

