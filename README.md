# IntrusionDetection-ML

Cyber Security Threat Classification Using Machine Learning

📌 Project Overview
This project focuses on classifying cybersecurity threats using **Machine Learning**. The goal is to preprocess a given dataset, train multiple ML models, and evaluate their effectiveness in detecting and categorizing network intrusions.

We use **Random Forest** and **Support Vector Machine (SVM)** for classification and compare their performance using various evaluation metrics.


📂 Dataset
- Dataset Used: ('synthetic_network_intrusion.csv')
- Features: Network parameters such as **Packet Size, Duration, Flow Rate, Source Port, Destination Port, Protocol**, etc.
- Target Variable: Attack type (Normal vs. Intrusion categories)


⚙️ Installation & Setup
Follow these steps to set up and run the project on your system.

INSTALL DEPENDENCIES
Ensure Python and necessary libraries are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

RUN THE PYTHON SCRIPT
```bash
python main.py
```

PROJECT WORKFLOW
1️. Data Preprocessing
Load the dataset and inspect its structure.
Handle missing values and inconsistent data.
Encode categorical variables (e.g., protocol types).
Normalize numerical features for better model performance.

2️. Feature Selection
Identify key features that contribute to attack classification.
Use feature importance techniques (e.g., correlation heatmaps, Random Forest feature importance).

3️. Model Training
Split the dataset into Training (80%) and Testing (20%) sets.
Train two classification models:
Random Forest Classifier
Support Vector Machine (SVM).

4️. Model Evaluation
Compare models based on:
Accuracy
Precision
Recall
F1-Score

Visualize the results using:
Confusion Matrix
Feature Importance plots

RESULTS
Random Forest gave comparitively better results than Support Vector Machine.

FUTURE IMPROVEMENTS
Optimize for better and more flexible results.
Implement Deep Learning (Neural Networks) for improved accuracy.
Use real-time threat detection with live network traffic.
Test with larger, real-world datasets for better generalization.

AUTHOR: Khushi Gupta
E-MAIL: khushhiii.28@gmail.com
