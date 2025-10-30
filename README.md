# 🌸 Iris Flower Classification

This project implements a machine learning model to classify **Iris flower species** (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*) based on sepal and petal measurements. It demonstrates **data preprocessing**, **exploratory data analysis (EDA)**, **model training**, and **evaluation** using Python and scikit-learn.

!Python
!scikit-learn
!Status
!License: MIT


## 📌 Overview

- **Goal:** Train a classifier to predict the Iris species from four features:
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Dataset:** Kaggle — `saurabh00007/iriscsv` (file: `Iris.csv`)
  - Original columns: `Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species`
  - The project automatically standardizes them to lowercase canonical names:
    `sepal_length, sepal_width, petal_length, petal_width, species`


## ✅ Features
- Load and clean Iris dataset from Kaggle (`Iris.csv`)
- Standardize column names for consistency
- Perform EDA (class balance, feature distributions)
- Train multiple models:
  - **Support Vector Machine (SVM)**
  - **Logistic Regression**
  - **Random Forest**
  - **k-Nearest Neighbors**
- Evaluate using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
- Save trained model and metrics
- Predict species for new samples via CLI


## 🚀 Quickstart

### Clone & set up the environment (Windows PowerShell)

```powershell
git clone https://github.com/manoj-dilshan/CodeAlpha_Iris-Flower-Classification.git
cd CodeAlpha_Iris-Flower-Classification

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
