# Predict Shipping Delays Based on Vessel Data

This project explores how machine learning can be applied in the maritime industry to predict shipping delays based on vessel data. By analyzing features like vessel type, route distance, cargo weight, weather conditions, and delay status, the project demonstrates the end-to-end process of data science: dataset generation, preprocessing, modeling, and evaluation.

---

## **Project Overview**

Shipping delays can significantly affect logistics and supply chain operations. This project uses a synthetic dataset to build a machine learning model to predict delays, focusing on:

1. **Dataset Generation**: Simulated vessel data with delay status.

2. **Data Preprocessing**: Encoding categorical features, scaling numerical data, and splitting into training and testing sets.
3. **Modeling**: Training a Random Forest classifier to classify shipments as delayed or on-time.
4. **Evaluation**: Analyzing model performance using metrics like accuracy, precision, recall, and confusion matrix.

---

## **Dataset**

The synthetic dataset contains the following features:

- **Vessel_Type**: Type of vessel (`Container`, `Tanker`, `Bulk Carrier`, `RoRo`).
- **Route_Distance**: Distance of the shipping route (in nautical miles).
- **Cargo_Weight**: Weight of the cargo (in tons).
- **Departure_Time**: Timestamp of departure (used for temporal insights, dropped for simplicity).
- **Weather_Conditions**: Weather during the journey (`Clear`, `Stormy`, `Windy`, `Foggy`).
- **Delay**: Target variable (1 = delayed, 0 = on-time).

---

## **Project Structure**

The project is organized into three main scripts:

1. **`dataset.py`**:

   - Generates the synthetic dataset and saves it as `maritime_data.csv`.
   - **Output**: `maritime_data.csv`

2. **`dataPreprocessing.py`**:

   - Loads `maritime_data.csv` and preprocesses the data:
     - One-hot encoding for categorical variables.
     - Scaling for numerical variables.
     - Splits the dataset into training and testing sets.
   - **Outputs**: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

3. **`predictiveModel.py`**:
   - Loads the preprocessed training and testing sets.
   - Trains a Random Forest classifier.
   - Evaluates the model using a classification report and confusion matrix.

---

## **How to Run the Project**

### **Step 1: Generate the Dataset**

Run `dataset.py` to create the synthetic dataset:

```bash
python dataset.py
```

This will generate a file named `maritime_data.csv`. in the directory.

### **Step 2: Preprocess the Data**

Runen `dataPreprocessing.py` to preprocess the dataset:

```bash
python dataPreprocessing.py
```

This will generate the following files:

- `X_train.csv`:
- `X_test.csv`:
- `y_train.csv`:
- `y_test.csv`:

### **Step 3: Train and Evaluate the Model**

Run `predictiveModel.py` to train and evaluate the Random Forest classifier:

```bash
python predictiveModel.py
```

This script will output the classification report and confusion matrix for the model.

#### Example of the output:

## **Classification Report**

```
              precision    recall  f1-score   support

           0       0.88      0.88      0.88        16
           1       0.75      0.75      0.75         8

    accuracy                           0.83        24
   macro avg       0.81      0.81      0.81        24
weighted avg       0.83      0.83      0.83        24

```

## **Confusion Matrix**

```
[[25  2]
 [ 9  4]]

```

### Interpretation:

- The model predicts on-time shipments (Class 0) more effectively than delayed shipments (Class 1), with high precision (0.74) and recall (0.93) for Class 0.
- For delayed shipments, recall is low (0.31), indicating the model misses many actual delays.

### Technical Details:

#### Libraries Used:

- pandas
- numpy
- scikit-learn

#### Machine Learning Model:

- Random Forest Classifier

### Key Insights:

- The model performs well for predicting on-time shipments (Class 0), achieving high accuracy and recall.
- Predicting delays (Class 1) is more challenging, with a lower recall indicating the model misses some delayed shipments.
- Overall accuracy is 82%, indicating room for improvement.
