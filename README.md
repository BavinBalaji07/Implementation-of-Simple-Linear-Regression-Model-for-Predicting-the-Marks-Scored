# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 Step 1: Import Libraries
 Step 2: Create Dataset (Hours studied vs Marks scored)
 Step 3: Split into Features and Target
 Step 4: Train-test split
 Step 5: Train Linear Regression Model
 Step 6: Predictions
 Step 7: Model Evaluation
 Step 8: Visualization
 Step 9: Predict Marks for custom input
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bavin Balaji R
RegisterNumber:  212225040045

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)
print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
*/
```

## Output:


<img width="1475" height="723" alt="image" src="https://github.com/user-attachments/assets/36db3f9f-04f3-4411-87c7-b5bc7237005c" />
<img width="1443" height="116" alt="image" src="https://github.com/user-attachments/assets/8d15c9bf-b06d-4dc2-bcc2-ebb8c74b8ed8" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
