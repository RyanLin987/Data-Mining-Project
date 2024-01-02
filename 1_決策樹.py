#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:14:52 2023

@author: mac
"""

import pandas as pd
bank=pd.read_csv("#5-bank-data.csv")


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

sex=le.fit_transform(bank["sex"])
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
children=le.fit_transform(bank["children"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

X=pd.DataFrame([bank["age"],sex,region,bank["income"],
               married,children,car,save_act,current_act,mortgage]).T
X.columns=["age","sex","region","income",
               "married","children","car",
               "save_act","current_act","mortgage"]
y=bank["pep"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.3, random_state=20231026)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=20231026,criterion="entropy",
                           min_samples_leaf=2, min_samples_split=0.25)
clf.fit(X_train,y_train)
print("建模正確率=",format(clf.score(X_train,y_train)*100,".2f"),"%")
print("測試正確率=",format(clf.score(X_test,y_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf.get_n_leaves())
print("樹的深度有多少層=",clf.get_depth())


from sklearn import tree
dot_data=tree.export_graphviz(clf,out_file=None,
                              feature_names=X.columns,
                              leaves_parallel=False,
                              impurity=True,
                              proportion=True,
                              rounded=True)
import os
os.environ["PATH"] += os.pathsep + "/Users/mac/opt/anaconda3/lib/python3.9/site-packages/graphviz-0.20.1.dist-info/"


pip install graphviz
import graphviz
graph=graphviz.Source(dot_data)
graph.format="png"
graph.render("tree.gv",view=True)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X,y, cv=4, scoring="accuracy")
print("交叉驗證的正確率=",format(scores.mean()*100,".2f")+"%")











