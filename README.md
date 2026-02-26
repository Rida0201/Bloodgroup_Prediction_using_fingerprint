Bloodgroup Prediction Using Fingerprint
A Machine Learning–based approach for non-invasive blood group prediction using fingerprint biometrics.
Overview
Traditional blood group identification requires invasive laboratory testing. This project proposes a non-invasive, data-driven alternative that leverages fingerprint patterns and machine learning algorithms to predict blood groups.
By analyzing biometric fingerprint features, the system classifies individuals into their respective blood groups using supervised learning techniques.
Objectives
Develop a non-invasive blood group prediction system
Extract meaningful biometric features from fingerprint images
Train and evaluate machine learning classification models
Enable real-time testing using a biometric fingerprint sensor
System Architecture
Copy code

Fingerprint Image
        ↓
Image Preprocessing
        ↓
Feature Extraction
        ↓
Model Training
        ↓
Blood Group Classification
🛠 Tech Stack
Programming Language: Python
Libraries: OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib
Machine Learning Models: Convolutional Neural Network(CNN)
Hardware: Biometric Fingerprint Sensor (for real-time implementation)
🔍 Methodology
1️⃣ Data Acquisition
Fingerprint samples were collected and labeled according to known blood groups.
2️⃣ Preprocessing
Grayscale conversion
Noise filtering
Image enhancement
Normalization
3️⃣ Feature Extraction
Ridge pattern analysis
Texture-based features
Minutiae characteristics
4️⃣ Model Training & Evaluation
Supervised learning algorithms were trained and evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Results-
Achieved Accuracy: 82%%
Model demonstrated promising classification performance on the dataset.
 Future Enhancements
Implementation using Convolutional Neural Networks (CNN)
Larger and more diverse dataset
Deployment as a web-based application
Integration into healthcare systems
Disclaimer
This project is developed for academic research purposes. Clinical validation is required before practical medical use.
Author
Rida Amin Jafri
Final Year B.Tech Project
Department of Information Technology