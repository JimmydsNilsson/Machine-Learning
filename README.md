# MNIST Digit Classifier – Machine Learning Project

This repository contains a complete machine learning pipeline built using the MNIST dataset.  
The project includes data exploration, preprocessing, model training, evaluation, and deployment through an interactive Streamlit application.

---

## 📌 Project Overview

The goal of this project was to build a digit classification model capable of recognizing handwritten digits (0–9).  
The workflow includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and scaling
- Training three different models:
  - Logistic Regression  
  - Random Forest  
  - K-Nearest Neighbors (KNN)
- Model comparison using accuracy, classification report, and confusion matrix
- Saving the best-performing model
- Deploying the model in a Streamlit web application

The final chosen model was **KNN**, which achieved the highest accuracy on the test dataset.

---

## 📊 Notebook

The full machine learning workflow is implemented in:

**`kunskapskontroll_2_mnist.ipynb`**

It includes:

- Dataset inspection  
- Visualization of sample digits  
- Train/test split  
- Feature scaling  
- Model training  
- Evaluation metrics  
- Saving the best model (`mnist_model.pkl`)

---

## 🧠 Model File

The best model is saved as:

**`mnist_model.pkl`**

This file is loaded by the Streamlit application to make predictions.

---

## 🖥️ Streamlit Application

You can run the interactive digit classifier using:

streamlit run streamlit_app.py

The app allows the user to draw a digit on a canvas.  
The image is preprocessed to match MNIST formatting (inversion, cropping, centering, padding), and the model predicts the digit in real time.

---

## 📄 Report

The project report is included as:

**`report.pdf`**  
(or `.docx` depending on submission format)

It describes:

- Methodology  
- EDA findings  
- Model selection  
- Evaluation results  
- Streamlit deployment  
- Final conclusions  

---

## 🛠️ Requirements

Install dependencies using:

pip install -r requirements.txt

The `requirements.txt` file includes all necessary libraries for running the notebook and the Streamlit app.

---

## ✔ Summary

This project demonstrates how to build a complete machine learning solution — from data analysis and model training to deployment in a user-friendly application.  
The final KNN model performs well on both MNIST test data and user-drawn digits thanks to improved preprocessing.

---


