# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/
```
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
predictions=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("classification Report:")
print(classification_report(y_test,predictions))
```

## Output:

![Screenshot 2024-04-30 161906](https://github.com/yogaraj2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153482637/0d1d33d6-ebe7-49fc-a0cb-e94f4e38f9d9)
![Screenshot 2024-04-30 161851](https://github.com/yogaraj2/Implementation-of-SVM-For-Spam-Mail-Detection/assets/153482637/4eb0a909-b209-4c5a-850f-c0f0fc05eea6)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
