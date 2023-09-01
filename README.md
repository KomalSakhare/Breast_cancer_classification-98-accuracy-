# Breast_cancer_classification-98 %-accuracy-
This repository contains a breast cancer diagnosis prediction project using deep learning techniques. The project aims to predict whether a breast cancer diagnosis is malignant (M) or benign (B) based on various features such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, and more.

Dataset:
The dataset used for this project includes the following columns:

diagnosis: The target variable, indicating whether the diagnosis is M (malignant) or B (benign).
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, and others: Various features describing the characteristics of cell nuclei.

The dataset is publicly available and can be found here [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset]

Project Structure:
data/: Contains the dataset file.
notebooks/: Jupyter notebooks used for exploratory data analysis (EDA) and model development.
models/: Saved deep learning models.
results/: Saved results and evaluation metrics.
requirements.txt: List of Python packages required to run the project.

Exploratory Data Analysis (EDA):
Before building the deep learning model, an Exploratory Data Analysis was conducted to gain insights into the dataset.

Deep Learning Model:
A neural network model was developed for breast cancer diagnosis prediction. The steps involved in model building and training are as follows:

1. Data preprocessing and splitting into training and testing sets.
2. Building a deep neural network using TensorFlow/Keras.
3. Implementing early stopping to prevent overfitting.
4. Applying dropout layers to improve generalization.
5. Training the model on the training data.
6. Evaluating the model's performance using classification metrics.
7. Displaying a confusion matrix.

Results:
The deep learning model achieved an accuracy of 98% on the test dataset.


