# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 

2.Upload the dataset and check for any null values using .isnull() function. 

3.Import LabelEncoder and encode the dataset. 

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Adchayakiruthika M S
RegisterNumber: 212223230005

import pandas as pd
data=pd.read_csv("/content/Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## HEAD(),INFO()&NULL():
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/ea3e65c5-436c-4689-9300-ee9cdd90977f)

![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/311befe3-a41a-490b-b565-55f8ce79fa6f)

![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/ac3cde54-50fd-4c41-9585-5987abe11b15)
## Converting string literals to numerical values using label encoder:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/0dea9f04-e1b6-43ff-b1d9-df6a60dea624)

## MEAN SQUARED ERROR:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/8a801b60-4671-4bd5-9287-e5ed452fb9b9)

## R2 (Variance):
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/bd46b474-3c90-48f9-a482-b04cb6379aae)

## DATA PREDICTION:
![image](https://github.com/Adchayakiruthika18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/147139995/db774f35-ad87-48ab-9764-14dddd718faf)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
