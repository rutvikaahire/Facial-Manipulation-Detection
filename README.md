# Facial Manipulation Detection

This repository contains an end-to-end project for detecting manipulated face videos. It covers dataset exploration, preprocessing, model training, evaluation, and inference, along with a Django web app for uploading videos and running predictions.

## Repository Structure

- [Application/](Application/) - Django app, deployment files, and project-level overview
- [Model Creation/](Model%20Creation/) - notebooks and notes for preprocessing, training, and prediction
- [labels/](labels/) - metadata and training labels

## What the Project Includes

- video frame extraction and face cropping
- sequence modeling with a ResNeXt50 + LSTM architecture
- training and validation on labeled datasets
- inference for new videos through notebooks and the web app
- optional visual explanation outputs such as heatmaps

## Datasets

- FaceForensics++
- Celeb-DF
- Deepfake Detection Challenge (DFDC)

## Start Here

If you want to understand the project quickly, read the [Application README](Application/README.md) first, then open the notebooks in [Model Creation/](Model%20Creation/).
