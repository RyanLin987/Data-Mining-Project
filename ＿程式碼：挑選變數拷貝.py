#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection
"""

import pandas as pd
bank=pd.read_csv("bank-data(2).csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
bank.info()
sex=le.fit_transform(bank["sex"])
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
children=le.fit_transform(bank["children"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])
X=pd.DataFrame([bank["age"],sex,region,bank["income"],married,children,car,save_act,current_act,mortgage]).T
X.columns=["age","sex","region","income","married","children","car","save_act","current_act","mortgage"]
y=bank["pep"]


#卡方挑出五個變數
from sklearn.feature_selection import SelectKBest, chi2
sk=SelectKBest(chi2,k=5)
sk.fit(X,y)
print(sk.get_feature_names_out()) 
X_new1=sk.transform(X)
X_new1=pd.DataFrame(X_new1)
X_new1.columns=["age","income","married","children","save"]



#建立clf模型
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=20231109,criterion="gini",
                           min_samples_split=0.25,
                           min_samples_leaf=2)
clf.fit(X,y)
print("建模方法的挑選變數正確率＝",clf.score(X, y))#0.6933
print(clf.feature_importances_)
#看看clf模型挑了哪些變數
from sklearn.feature_selection import SelectFromModel
sm=SelectFromModel(clf,max_features=5)
sm.fit(X,y)
print(sm.get_feature_names_out())
X_new2=sm.transform(X)
X_new2=pd.DataFrame(X_new2)
X_new2.columns=["income","married","children","save","mortage"]



#大比拼
clf.fit(X_new1, y)
print("卡方分配法找出的變數的模型正確率＝",clf.score(X_new1, y))#0.6783
clf.fit(X_new2, y)
print("模型找出的變數的模型正確率=",clf.score(X_new2, y))#0.6933



#若只採用四個變數
sm2=SelectFromModel(clf,max_features=4)
sm2.fit(X,y)
print(sm2.get_feature_names_out())
X_new3=sm2.transform(X)
X_new3=pd.DataFrame(X_new3)
X_new3.columns=["income","married","children","save"]
clf.fit(X_new3, y)
print("模型找出的前四強變數的模型正確率",clf.score(X_new3, y))#0.6783










