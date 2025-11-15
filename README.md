# 🌾 Crop Prediction System

## 📌 Overview
The Crop Prediction System is a Machine Learning–powered web application built using Python, Scikit-learn, and Streamlit.
It predicts the most suitable crop based on environmental parameters such as:
      - Nitrogen
      - Phosphorus
      - Potassium
      - Temperature
      - Humidity
      - pH
      - Rainfall
    This project helps farmers and researchers identify the best crop to grow for maximum yield.

## 🚀 Features
  - ✔ User-friendly Streamlit interface
  - ✔ Real-time crop prediction
  - ✔ Pre-trained ML model (Random Forest / SVM / Decision Tree)
  - ✔ Dataset pre-processing & model training pipeline included
  - ✔ Joblib model loading support
  - ✔ Lightweight & fast inference

## 🧠 Algorithms Used

The project includes multiple machine learning algorithms, but the final model uses:

- Random Forest Classifier
    - Handles nonlinear data
    - Works well with agricultural features
    - Low overfitting
    - High accuracy

- Other tested algorithms:
    - Logistic Regression
    - SVM
    - Decision Tree
    - Naive Bayes
    - KNN

Random Forest gave the best results.

## 🛠 Technologies Used

| Component       | Technology                  |
| --------------- | --------------------------- |
| Language        | Python 3.10                 |
| ML Framework    | Scikit-Learn                |
| UI Framework    | Streamlit                   |
| Model Saving    | Joblib                      |
| Data Processing | Pandas, NumPy               |
| Visualization   | Matplotlib / Seaborn        |
| Dataset         | Crop Recommendation Dataset |

## 📁 Project Structure
<img width="666" height="377" alt="image" src="https://github.com/user-attachments/assets/279f66b3-9b96-4a49-bd70-24480703a648" />

## 🧪 Example Prediction Inputs
| Feature     | Example Value |
| ----------- | ------------- |
| Nitrogen    | 90            |
| Phosphorus  | 42            |
| Potassium   | 43            |
| Temperature | 25            |
| Humidity    | 80            |
| pH          | 6.5           |
| Rainfall    | 200           |


