# Heart Disease Prediction using XGBoost

## About
This project focuses on predicting the presence of heart disease using machine learning techniques.
An XGBoost classifier is trained on a medical dataset containing patient health measurements such as age, blood pressure, cholesterol level, and exercise-related indicators.
The project includes data exploration, visual analysis, model training, and model evaluation using standard performance metrics.

---

## Files
- `heart.csv` → This file contains the heart disease dataset used for training and testing the model.
- `heart_disease.py` → This is the main Python script of the project.

---

## Dataset Description

The dataset contains medical and clinical features collected from patients for heart disease prediction.
Each row represents a single patient record with multiple health-related measurements and a target label.

## Target Variable

- target
  - 0 → No heart disease
  - 1 → Heart disease present

## Categorical Features

- sex      – Gender
- cp       – Chest pain type
- fbs      – Fasting blood sugar
- restecg  – Resting electrocardiographic results
- exang    – Exercise-induced angina
- slope    – Slope of the ST segment
- ca       – Number of major vessels colored by fluoroscopy
- thal     – Thalassemia

## Continuous Features

- age       – Age of the patient
- trestbps  – Resting blood pressure
- chol      – Serum cholesterol level
- thalach   – Maximum heart rate achieved
- oldpeak   – ST depression induced by exercise relative to rest

---

## Steps Included

### 1️⃣ Data Preprocessing

- Loading the Heart Disease Dataset

        The dataset is loaded from the `heart.csv` file using pandas.
        It contains medical and clinical records for patients, where each row represents a single patient.

- Checking for Missing Values

        The dataset is examined for missing or null values.
        No missing values were found, ensuring the data is complete and consistent.

- Separating Features and Target Variable

        The input features (X) are separated from the target label (y).
        The target variable represents whether a patient has heart disease or not.

- Identifying Feature Types

        Features are categorized into:
          - Categorical features (e.g., sex, cp, fbs, thal)
          - Continuous features (e.g., age, cholesterol, blood pressure)

- Handling Categorical Features

        Categorical features are already numerically encoded in the dataset.
        No additional encoding is required.

- Feature Scaling

        No feature scaling or normalization is applied.
        This is because tree-based models such as XGBoost are not sensitive to feature scaling.

- Splitting the Dataset

        The dataset is split into training and testing sets using an 80/20 ratio.
        Stratified sampling is applied to maintain the original class distribution.

---

### 2️⃣ Model Training


- Model Initialization

        An XGBoost Classifier is initialized as the base model.
        This model is suitable for structured medical data and binary classification tasks.

- Hyperparameter Tuning

        GridSearchCV is used to search for the best combination of hyperparameters.
        The following parameters are optimized:
          - Number of estimators
          - Maximum tree depth
          - Learning rate
          - Subsample ratio
          - Column sampling ratio

- Cross-Validation

        Five-fold cross-validation is applied during training.
        This helps ensure that the model performance is stable and not dependent on a single data split.

- Model Fitting

        The model is trained on the training dataset using the optimal hyperparameters.
        During training, each tree learns to correct the errors of previous trees.

- Prediction

        After training, the model is used to predict labels for the test dataset.
        These predictions are later used for performance evaluation.


---


## How to Run

1- Install Dependencies:
  ```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

```

2- Make sure data file exists in the root of your work:
  ```bash
heart.csv

```

3-Run :

  ```bash
python heart_disease.py
```

3- Application Preview :

### Dataset Preview (First 5 Rows)

```text
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   63    1   3       145   233    1        0      150      0      2.3      0   
1   37    1   2       130   250    0        1      187      0      3.5      0   
2   41    0   1       130   204    0        0      172      0      1.4      2   
3   56    1   1       120   236    0        1      178      0      0.8      2   
4   57    0   0       120   354    0        1      163      1      0.6      2   

   ca  thal  target  
0   0     1       1  
1   0     2       1  
2   0     2       1  
3   0     2       1  
4   0     2       1
```

This output shows the first 5 patient records in the dataset after loading heart.csv.
It is used to quickly verify that the data is loaded correctly and to preview the feature columns and values.

---

### Missing Values Check

```text
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          0
thal        0
target      0
dtype: int64
```

This output shows the number of missing (NaN) values in each column.
All values are 0, which means the dataset contains no missing data.

---

### Unique Values and Feature Types

```text
-----------------------------
age : [63 37 41 56 57 44 52 54 48 49 64 58 50 66 43 69 59 42 61 40 71 51 65 53
 46 45 39 47 62 34 35 29 55 60 67 68 74 76 70 38 77]
-----------------------------
sex : [1 0]
-----------------------------
cp : [3 2 1 0]
-----------------------------
trestbps : [145 130 120 140 172 150 110 135 160 105 125 142 155 104 138 128 108 134
 122 115 118 100 124  94 112 102 152 101 132 148 178 129 180 136 126 106
 156 170 146 117 200 165 174 192 144 123 154 114 164]
-----------------------------
chol : [233 250 204 236 354 192 294 263 199 168 239 275 266 211 283 219 340 226
 247 234 243 302 212 175 417 197 198 177 273 213 304 232 269 360 308 245
 208 264 321 325 235 257 216 256 231 141 252 201 222 260 182 303 265 309
 186 203 183 220 209 258 227 261 221 205 240 318 298 564 277 214 248 255
 207 223 288 160 394 315 246 244 270 195 196 254 126 313 262 215 193 271
 268 267 210 295 306 178 242 180 228 149 278 253 342 157 286 229 284 224
 206 167 230 335 276 353 225 330 290 172 305 188 282 185 326 274 164 307
 249 341 407 217 174 281 289 322 299 300 293 184 409 259 200 327 237 218
 319 166 311 169 187 176 241 131]
-----------------------------
fbs : [1 0]
-----------------------------
restecg : [0 1 2]
-----------------------------
thalach : [150 187 172 178 163 148 153 173 162 174 160 139 171 144 158 114 151 161
 179 137 157 123 152 168 140 188 125 170 165 142 180 143 182 156 115 149
 146 175 186 185 159 130 190 132 147 154 202 166 164 184 122 169 138 111
 145 194 131 133 155 167 192 121  96 126 105 181 116 108 129 120 112 128
 109 113  99 177 141 136  97 127 103 124  88 195 106  95 117  71 118 134
  90]
-----------------------------
exang : [0 1]
-----------------------------
oldpeak : [2.3 3.5 1.4 0.8 0.6 0.4 1.3 0.  0.5 1.6 1.2 0.2 1.8 1.  2.6 1.5 3.  2.4
 0.1 1.9 4.2 1.1 2.  0.7 0.3 0.9 3.6 3.1 3.2 2.5 2.2 2.8 3.4 6.2 4.  5.6
 2.9 2.1 3.8 4.4]
-----------------------------
slope : [0 2 1]
-----------------------------
ca : [0 2 1 3 4]
-----------------------------
thal : [1 2 3 0]
-----------------------------
target : [1 0]
-----------------------------
Categorical Features : ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
Continous Features : ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
```

This output lists the unique values for each feature to understand whether it is categorical or continuous.
At the end, the script summarizes which features are treated as categorical vs. continuous based on the number of unique values.

---

### Categorical Feature Distribution

<p align="center">
<img width="1218" height="1218" alt="download" src="https://github.com/user-attachments/assets/42af6e17-2ef9-4637-ac31-0243e88a73bc" />
</p>

This figure shows the distribution of categorical features for patients with and without heart disease.
The blue bars represent patients without heart disease, while the red bars represent patients with heart disease.

The plots illustrate how each categorical variable (such as sex, chest pain type, fasting blood sugar,
ECG results, exercise-induced angina, slope, number of vessels, and thalassemia)
is distributed across the two classes.

These visualizations help identify patterns and differences between patients with and without heart disease,
and provide insight into which categorical features may be more influential for prediction.

---


### Continuous Feature Distribution

<p align="center">
<img width="1222" height="1221" alt="download (1)" src="https://github.com/user-attachments/assets/1f5145c3-e23c-4c03-98c9-1d0260e82fbd" />
</p>


This figure shows the distribution of continuous features for patients with and without heart disease.
The blue histograms represent patients without heart disease, while the red histograms represent patients with heart disease.

The plots compare key continuous variables such as age, resting blood pressure, cholesterol level,
maximum heart rate achieved, and ST depression (oldpeak).

From these distributions, it can be observed that patients with heart disease tend to have:
- Higher age values
- Lower maximum heart rate (thalach)
- Higher ST depression values (oldpeak)

These visualizations help in understanding how continuous medical measurements differ between the two classes
and highlight features that may be important for heart disease prediction.

---




<p align="center">
<img width="1049" height="547" alt="download (2)" src="https://github.com/user-attachments/assets/68bd642e-6676-43f8-ab08-5efa887b0eeb" />
</p>


### Classification Report

This figure presents the classification report of the trained model using a heatmap visualization.
It summarizes the model’s performance in terms of precision, recall, and F1-score for each class.

- Class 0 (No Heart Disease):
  The model achieves high precision, indicating that most patients predicted as healthy are correctly classified.

- Class 1 (Heart Disease):
  The model shows very high recall, meaning it successfully identifies most patients who actually have heart disease.

- Overall Performance:
  The balanced F1-scores and an overall accuracy of approximately 85% indicate that the model performs reliably
  in distinguishing between patients with and without heart disease.

---





<p align="center">
<img width="519" height="470" alt="download (3)" src="https://github.com/user-attachments/assets/d0112628-6cb5-460c-b5ef-0bb1ae2a2920" />
</p>


### Confusion Matrix

This figure shows the confusion matrix of the heart disease classification model.
It illustrates the comparison between true labels and predicted labels.

- True Negatives (TN = 20):
  Patients without heart disease correctly classified as healthy.

- False Positives (FP = 8):
  Patients without heart disease incorrectly classified as having heart disease.

- False Negatives (FN = 1):
  Patients with heart disease incorrectly classified as healthy.

- True Positives (TP = 32):
  Patients with heart disease correctly classified as having heart disease.

The confusion matrix indicates that the model performs well in identifying heart disease cases,
with very few false negatives, which is critical in medical diagnosis tasks.

---





<p align="center">
<img width="536" height="393" alt="download (4)" src="https://github.com/user-attachments/assets/d05c5a83-b9d3-41f8-b39f-56cac2c68709" />
</p>


### ROC Curve

This figure shows the Receiver Operating Characteristic (ROC) curve for the heart disease classification model.
The ROC curve illustrates the relationship between the True Positive Rate and the False Positive Rate
at different classification thresholds.

The model achieves an Area Under the Curve (AUC) value of 0.88,
which indicates a strong ability to distinguish between patients with and without heart disease.

An AUC value close to 1.0 reflects good model performance,
confirming that the classifier is effective in medical diagnosis tasks.

---





<p align="center">
<img width="630" height="299" alt="download (5)" src="https://github.com/user-attachments/assets/26c45b85-c010-4097-9825-128144fa7f38" />
</p>


### Decision Tree Visualization

This figure shows a visualization of a single decision tree extracted from the trained XGBoost model.
Each node represents a decision based on a specific feature and threshold value.

The tree demonstrates how the model makes predictions by:
- Splitting the data based on important medical features
- Following decision paths that lead to a final classification outcome

Although XGBoost uses an ensemble of many trees, this visualization helps in understanding
how individual trees contribute to the final prediction and improves model interpretability.


---






<p align="center">
<img width="726" height="547" alt="download (6)" src="https://github.com/user-attachments/assets/9cb782ce-ce00-43c8-9f89-3e0945c7e87b" />
</p>


### Feature Importance

This figure displays the feature importance scores generated by the XGBoost model.
Each bar represents the relative contribution of a feature to the model’s prediction.

The most influential features in predicting heart disease are:
- Chest pain type (cp)
- Thalassemia (thal)
- Slope of the ST segment (slope)
- Number of major vessels (ca)

Features with lower importance, such as resting blood pressure and fasting blood sugar,
have a smaller impact on the final prediction.

This analysis helps in understanding which medical factors play a key role
in heart disease prediction and enhances model interpretability.


---


 ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.
