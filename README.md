# Stroke Prediction Model

This project focuses on building a machine learning model to predict strokes based on a dataset containing various health-related features. The following steps outline the process used in this project:

---

## 1. Explore the Dataset (Business Understanding, Features)

- **Dataset**: The dataset used for this model contains information about patients, including features such as age, gender, smoking status, and health indicators like hypertension and heart disease. The target variable is whether the patient had a stroke (1) or not (0).
- **Features**: The key features include:

  - `age`: Age of the patient
  - `gender_encoded`: Encoded gender information
  - `hypertension`: Whether the patient has hypertension
  - `heart_disease`: Whether the patient has heart disease
  - `avg_glucose_level`: Average glucose level
  - `bmi`: Body mass index
  - `smoking_status_encoded`: Encoded smoking status

  The target variable is:

  - `stroke`: 1 for stroke, 0 for no stroke.

## 2. Cleaning the Data (Nettoyage)

- **Handling Missing Values**: The dataset was first explored to check for missing values. Some features, such as smoking status, had missing values, which were filled using **Label Encoding** for categorical columns and **IterativeImputer** for numerical columns.
- **Data Types**: Columns were checked to ensure the correct data types were applied, and any necessary transformations (like encoding categorical variables) were done.

## 3. Feature Selection (Sélection des Caractéristiques)

- Feature selection was performed to select the most relevant features for the model, which are:
  - `age`, `gender_encoded`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, and `smoking_status_encoded`.
- This feature selection was done based on the correlation between the features and the target variable (`stroke`), as well as domain knowledge of relevant health factors that contribute to strokes.

## 4. Balancing the Dataset (Équilibrage du Dataset)

- **SMOTETomek**: The dataset was highly imbalanced (with fewer strokes than non-strokes). To address this, we applied the **SMOTETomek** method, which combines **SMOTE** (Synthetic Minority Over-sampling Technique) for over-sampling the minority class (stroke) and **Tomek Links** for cleaning the majority class (non-stroke).
- The impact of this balancing method was studied by comparing the performance on the **training set** before and after balancing, and then evaluated on the **test set**.

## 5. Model Evaluation (Évaluation du Modèle)

- **Model**: A **RandomForestClassifier** was chosen for this project due to its robustness and ability to handle both numerical and categorical features.
- **Metrics**: The model's performance was evaluated using the following metrics:
  - **Accuracy**: Overall accuracy of the model.
  - **Precision**: Precision for both classes.
  - **Recall**: Recall for both classes.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **Confusion Matrix**: To analyze the number of correct and incorrect predictions.
  - **ROC Curve & AUC**: To evaluate the model's ability to distinguish between the two classes (stroke vs. no stroke).

### **Performance**:

- The model was trained on the balanced dataset and evaluated on the test set. The evaluation metrics and visualizations (Confusion Matrix, ROC Curve) helped assess the overall performance.

---

## Technologies Used (Technologies Utilisées)

- **Python 3.10+**
- **Libraries**:
  - **scikit-learn** for model training and evaluation.
  - **imblearn** for SMOTETomek.
  - **matplotlib** and **seaborn** for data visualization.
  - **pandas** for data manipulation.
  - **numpy** for numerical operations.

---

## Conclusion

This project demonstrates the effectiveness of using **SMOTETomek** for balancing imbalanced datasets and evaluates the performance of the **RandomForestClassifier** for stroke prediction. The model showed good performance in terms of classification metrics, especially after applying balancing techniques. Further improvements can be made by tuning hyperparameters and exploring other models.

---

## Installation

To run this project, ensure the following libraries are installed:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn numpy
```
