**🧠 Breast Cancer Prediction Using Machine Learning**

This repository contains the implementation of the "Breast Cancer Prediction Using Machine Learning" guided project, successfully completed as part of the Coursera Project Network.



**📁 Project Structure**

📦Breast-Cancer-Prediction

    ┣ 📜Breast_Cancer_Prediction.ipynb
 
    ┣ 📜README.md
  
    ┗ 📜Certificate.pdf

**✅ Objective**

The main objective of this project is to build a machine learning model that can predict whether a tumor is malignant or benign, based on features derived from digitized images of a breast mass.


**🔍 Dataset**

* **Source:** Wisconsin Breast Cancer Dataset (available in sklearn.datasets)

* **Features:** Radius, Texture, Perimeter, Area, Smoothness, etc.

* **Target:** Diagnosis (M = Malignant, B = Benign)

**🛠️ Tools & Libraries**
* Python

* Pandas

* NumPy

* Seaborn & Matplotlib (Data Visualization)

* Scikit-learn (Modeling & Evaluation)

**📊 Workflow**
1. Data Loading & Exploration

   * Loaded the dataset using sklearn.datasets.load_breast_cancer()

   * Explored class balance and visualized feature correlations

2. Preprocessing

   * Converted to Pandas DataFrame

   * Checked for nulls

   * Encoded the target variable (Malignant = 1, Benign = 0)

3. Model Building

   * Used a Logistic Regression classifier

   * Trained/test split (80/20)

   * Model fitted on training data

4. Evaluation

   * Accuracy score, Confusion matrix


**📈 Results**
* Model Used: Logistic Regression

* Accuracy Achieved: ~96%

**📜 Certificate**

The project was successfully completed under the guidance of Priya Jha, Coursera Guided Projects Instructor.




🔗 [Click here to verify](https://coursera.org/verify/IY360L2U8E6L)
