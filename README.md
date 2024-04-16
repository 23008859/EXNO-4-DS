# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/18c88e8d-770f-4f6b-99f2-76d6ca673100)
```
df.head()
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/52eebac9-fd62-4a5b-ba8d-67cdbdb1377f)
```
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/a9669f27-3849-4660-8e22-51935d0807cf)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/702218fb-3aa2-49d2-9b17-6484445dc534)
```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/5399ac8c-24db-4b96-9cf6-01db309259d6)
```
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/1f709d8e-8717-481c-8878-33d019e6917e)
```
df=pd.read_csv("/content/bmi.csv")
```
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/017fe938-28d7-4c3c-ade3-66b906f5ce1e)
```
import pandas as pd
import numpy as np
import seaborn as sns
```
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data = pd.read_csv("/content/income(1) (1).csv")
data
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/3f411723-b574-45b5-b3ca-eb1912049a59)

```
data.isnull().sum()
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/7ac693dd-5b20-478b-a695-b75216cc016d)
```
missing = data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/fe3463d0-aaeb-4bf6-b98b-f13bb426c030)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/01c35252-d6ef-46cf-b5a8-108c3e613fe1)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/dd266bac-493a-46ff-a00b-95a5c0d4d34b)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/4ba93d42-cc87-4768-b95e-c2d17f8c4b22)
```
data2
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/4c264d04-96eb-43c6-b8e7-7751ab513b73)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/1470c9b6-0181-4877-893f-319347671a2f)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/f4aca66b-d63b-4286-b6e5-1cb8f0fdd950)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/d265be98-1298-480d-be5c-75f1e81574b7)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/b0c4bd14-c826-4a92-930f-77634aca10e9)
```
x = new_data[features].values
print(x)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/c47ae518-581d-4af6-8658-66eb91a6582e)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state = 0)
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/005cf72a-aa5f-4337-8290-64ce13d3a0e5)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/e3e67f31-1e9a-4af2-be02-90f6f7c9c71a)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/1b4fd72d-1f79-464e-ac60-4f29f8dea201)
```
print( 'Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/6a915ce6-b269-4e6c-81e7-112f8584e06a)
```
data.shape
```
![image](https://github.com/23008859/EXNO-4-DS/assets/139117979/aa62b9ae-b81d-4778-958e-471b6f76bf00)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
