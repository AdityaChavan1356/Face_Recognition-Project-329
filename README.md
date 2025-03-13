# Face Recognition with Scikit-learn

This repository demonstrates a basic face recognition system using Scikit-learn, focusing on traditional machine learning techniques rather than deep learning. This approach is suitable for understanding the fundamentals of face recognition and may be applicable in resource-constrained environments.

## Project Overview

This project implements a face recognition pipeline using Scikit-learn, leveraging techniques like Principal Component Analysis (PCA) for dimensionality reduction and Support Vector Machines (SVMs) or other classifiers for recognition. It aims to provide a clear and simple example of how to build a face recognition system using classical machine learning.

## Key Features

* **Scikit-learn Implementation:** Utilizes Scikit-learn for all machine learning tasks.
* **PCA for Dimensionality Reduction:** Employs PCA to reduce the dimensionality of face images, improving efficiency and reducing noise.
* **SVM or Other Classifiers:** Uses SVM (or other Scikit-learn classifiers) for face recognition based on the reduced feature space.
* **Data Loading and Preprocessing:** Includes functions for loading face image datasets and preprocessing them for analysis.
* **Model Training and Evaluation:** Demonstrates how to train and evaluate the face recognition model.
* **Simple Prediction:** Provides a basic example of how to make predictions on new face images.

## Technologies Used

* **Python 3.x:** The primary programming language.
* **Scikit-learn (sklearn):** For machine learning tasks, including PCA and SVM.
* **NumPy:** For numerical operations.
* **Pillow (PIL):** For image loading and manipulation.
* **Matplotlib:** For data visualization (optional).

## Getting Started

### Prerequisites

* Python 3.x
* Scikit-learn (install using `pip install scikit-learn`)
* NumPy (install using `pip install numpy`)
* Pillow (install using `pip install Pillow`)
* Matplotlib (install using `pip install matplotlib`, optional)

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/face_recognition_sklearn.git](https://www.google.com/search?q=https://www.google.com/search%3Fq%3Dhttps://github.com/your-username/face_recognition_sklearn.git)
    cd face_recognition_sklearn
    ```

2.  Install the required dependencies:

    ```bash
    pip install scikit-learn numpy Pillow matplotlib
    ```

### Usage

1.  **Prepare your face dataset:** Organize your face images into folders, where each folder represents a person.
2.  **Run the Python script:** Execute the `face_recognition_sklearn.py` script.
3.  **Model Training:** The script will load the dataset, preprocess the images, train the PCA and SVM models.
4.  **Model Evaluation:** The script will evaluate the model's performance.
5.  **Make predictions:** You can modify the script to load and predict on new face images.

**Dataset Download:**

The first time you run the code, it will download the LFW dataset, which may take some time.
Computational Resources: Training the SVM, especially with grid search, can be computationally intensive. The execution time will depend on your system's resources.
Adjusting Parameters: You can experiment with different PCA components, SVM kernels, and hyperparameter ranges to see how they affect performance.
Error handling: The provided code does not contain much error handling. When adapting this code, it is best to add error handling.
File paths: If you want to save the figures that are created, or save the model, you will need to add code that specifies the file paths to save the generated content to.

**Future Improvements-**

Experiment with different models (e.g., Convolutional Neural Networks).
Increase the dataset size or use data augmentation techniques.
Fine-tune the hyperparameters further for better performance.
Implement real-time face recognition using a webcam.

#OutPut-

![image](https://github.com/user-attachments/assets/503c52e7-e4b4-4eb3-8042-fcd8faf5e2a0)

![image](https://github.com/user-attachments/assets/8a7b3ee6-6bec-4cc0-973b-c130c67d239d)

