# Face Recognition with Scikit-learn

This project demonstrates face recognition using scikit-learn, specifically employing Support Vector Machines (SVM) and Principal Component Analysis (PCA) on the Labeled Faces in the Wild (LFW) dataset.

**Description:**
This project focuses on face recognition using Seaborn for data visualization and SciPy for scientific computing. It involves image processing, feature extraction, and face identification techniques based on mathematical and statistical methods. The goal is to analyze facial features and recognize faces using computational techniques. With the combination of Seaborn and SciPy, the project enables efficient data visualization and precise facial feature analysis. It also leverages additional libraries like NumPy for numerical operations and OpenCV for image handling.
Technologies Used:

Python: The core programming language.
Scikit-learn (sklearn): A powerful machine learning library for tasks like:
Loading datasets (sklearn.datasets)
Dimensionality reduction (PCA - sklearn.decomposition)
Support Vector Machines (SVM - sklearn.svm)
Model pipelines (sklearn.pipeline)
Model evaluation (sklearn.metrics)
Hyperparameter tuning (sklearn.model_selection)
Data splitting (sklearn.model_selection)
NumPy: For numerical computations and array manipulation.
Matplotlib: For data visualization and plotting.
IPython/Jupyter Notebook: (Implied by the %matplotlib inline and In [..]: notation) For interactive coding and execution.
Installation:

Python:

Ensure you have Python 3.6 or later installed. You can download it from python.org.
Virtual Environment (Recommended):

It's best practice to create a virtual environment to isolate project dependencies.
Bash

  python -m venv myenv
Activate the environment:
On Windows: myenv\Scripts\activate
On macOS/Linux: source myenv/bin/activate
Install Required Libraries:

Use pip, the Python package installer:
Bash

  pip install scikit-learn numpy matplotlib ipython
Steps to Run the Project:

Save the Code:

Copy the provided code and save it as a Python file (e.g., face_recognition.py) or in a Jupyter Notebook (face_recognition.ipynb).
Run the Code:

If you saved it as a Python file:
Open a terminal or command prompt.
Navigate to the directory where you saved the file.
Execute: python face_recognition.py
If you saved it as a Jupyter Notebook:
launch jupyter notebook from the command line in the directory that the notebook is located in.
open the notebook in the browser.
Run the cells sequentially by pressing Shift + Enter.
Observe the Output:

The code will:
Download and load the LFW dataset.
Display sample faces.
Train an SVM classifier with PCA.
Print the best hyperparameters found by grid search.
Display predicted labels for test images, with incorrect labels in red.
Print a classification report summarizing the model's performance.
Important Notes:

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

OutPut-

![image](https://github.com/user-attachments/assets/503c52e7-e4b4-4eb3-8042-fcd8faf5e2a0)

![image](https://github.com/user-attachments/assets/8a7b3ee6-6bec-4cc0-973b-c130c67d239d)

