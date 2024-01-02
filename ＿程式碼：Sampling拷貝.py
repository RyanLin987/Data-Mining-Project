#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:19:27 2023

Topic: Sampling

"""
import pandas as pd
ti=pd.read_csv("titanic-train(2).csv")
X=ti.iloc[:,1:-1]
y=ti["Survived"]
print(y.value_counts())

from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
X_resampled,y_resampled=rus.fit_resample(X,y)
print("undersampling")
print(y_resampled.value_counts())


from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X_resampled2,y_resampled2=ros.fit_resample(X,y)
print("oversampling")
print(y_resampled2.value_counts())


#ti.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(ti["Sex"])
cabin=le.fit_transform(ti["Cabin"])
embarked=le.fit_transform(ti["Embarked"])
X2=pd.DataFrame([ti["PassengerId"],ti["Pclass"],sex,ti["SibSp"],
                 ti["Parch"],ti["Ticket"],ti["Fare"],cabin,embarked]).T
X2.columns=["PassengerId","Pclass","Sex","SibSp","Parch",
            "Ticket","Fare","Cabin","Embarked"]
from imblearn.over_sampling import SMOTE
X_resampled3,y_resampled3=SMOTE(random_state=0).fit_resample(X2, y)
print("SMOTE")
print(y_resampled3.value_counts())



































