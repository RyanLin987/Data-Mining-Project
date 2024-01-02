

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:08:32 2023

@author: mac
"""


import pandas as pd
bank=pd.read_csv("bank-data(3).csv")
bank.info()
subdata=bank[(bank["region"]=="INNER_CITY")|
             (bank["region"]=="TOWN")|(bank["region"]=="RURAL")]

x_train,y_train=subdata[["age","income"]],subdata["region"]
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(x_train)
x_train_std=ss.transform(x_train)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_train=le.fit_transform(y_train)

from sklearn.svm import LinearSVC
m=LinearSVC(C=0.1, dual=False, class_weight="balanced")
m.fit(x_train_std, y_train)
y_pred=m.predict(x_train_std)
print("訓練資料集正確率＝", m.score(x_train_std, y_train))
print("訓練資料集分類錯誤筆數＝", (y_train!=y_pred).sum())
print(1-(y_train!=y_pred).sum()/538)
from sklearn.metrics import f1_score
print("F1-score=", f1_score(y_train, y_pred, average="weighted"))












