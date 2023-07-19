import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("newheart.csv")
df.head(303)

cols = [0]
df.drop(df.columns[cols],axis=1,inplace=True)
y = df.iloc[:,13].values

drop = ['fbs', 'restecg', 'trestbps', 'chol', 'age', 'slope','sex']
df = df.drop(drop, axis = 1)
# print(df)
x = df.iloc[:,:-1].values
x=x.T
y=y.T

# ================================================================================
# 'fbs', 'restecg', 'sex', 'trestbps', 'chol', 'age', 'slope', 'oldpeak', 'thalach', 'exang', 'thal', 'cp', 'ca',
# 'fbs', 'restecg', 'trestbps', 'chol', 'age', 'slope','thal'

cp = int(input())
thalach = int(input())
exang = int(input())
oldpeak = int(input())
ca = int(input())
thal = int(input())
x_test=np.array([cp,thalach,exang, oldpeak,ca, thal])
x_test=pd.DataFrame(x_test)

knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k
knn.fit(x.T, y.T)
prediction = knn.predict(x_test.T)
print(prediction)