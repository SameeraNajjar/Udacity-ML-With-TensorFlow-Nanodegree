# Image Classifier Project

## Overview

This project focuses on building an image classifier to recognize different species of flowers using deep learning techniques. The goal is to create a model that can be used in applications such as smartphone apps to identify flowers from images in real-time.

## Project Structure

1. **Data Exploration**
   - Explore the Oxford 102 Flowers dataset to understand the structure and features.
   - Visualize sample images and their corresponding labels.

2. **Data Preprocessing**
   - Load and preprocess the data, including resizing and normalizing images.
   - Prepare training, validation, and test datasets with the appropriate format required by the deep learning model.

3. **Model Implementation**
   - Build and train a deep learning model using the MobileNet architecture from TensorFlow Hub.
   - Fine-tune the model and optimize it to achieve high accuracy on the validation set.

4. **Model Evaluation**
   - Evaluate the performance of the model on a separate test set to measure its accuracy.
   - Visualize the training and validation accuracy and loss over time to assess model performance.

5. **Inference**
   - Implement an inference pipeline to make predictions on new images.
   - Use the trained model to classify new flower images and identify the most likely species.

## Data Disclaimer

The dataset used in this project is the Oxford 102 Flowers dataset, which contains 102 different categories of flowers. The project code is available, but you will need to download the dataset from the official source to replicate the results.

## Installation

You will need the following libraries:
- `tensorflow`
- `tensorflow_hub`
- `tensorflow_datasets`
- `matplotlib`
- `numpy`
- `Pillow`

Install the dependencies using pip:

```bash
pip install tensorflow tensorflow_hub tensorflow_datasets matplotlib numpy Pillow
```

## Running the Project
1. Data Preparation
    - Load the Oxford 102 Flowers dataset using TensorFlow Datasets.
    - Preprocess the images by resizing them to 224x224 pixels and normalizing pixel values.

2. Model Training
    - Load the MobileNet model from TensorFlow Hub.
    - Build a custom classifier on top of the pre-trained model.
    - Train the model using the training dataset and validate it using the validation dataset.

3. Evaluation
    - Evaluate the trained model on the test dataset.
    - Visualize the performance using plots of accuracy and loss over epochs.

4. Inference
    - Use the trained model to classify new images of flowers.
    - Visualize the predictions and the corresponding probabilities for the top predictions.

## Results
- The final model successfully classifies flower images with a test accuracy of approximately 70%.
- Visualizations illustrate the model's performance, showing the training and validation accuracy over time.

## Conclusion
This project demonstrates the application of deep learning to image classification, focusing on recognizing different species of flowers. The trained model can be integrated into applications for real-time flower identification, showcasing the power of AI in everyday tasks.