#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:18:13 2023

@author: mac
作業：
"""

import pandas as pd

ti=pd.read_csv("titanic-train(1).csv")
ti.drop("PassengerId", axis=1,inplace=True)
import numpy as np
import statistics
ti.info()

print(statistics.mode(ti["Cabin"]))
ti["Cabin"]=np.where(ti["Cabin"].isnull(),statistics.mode(ti["Cabin"]),ti["Cabin"])
ti["Age"]=np.where(ti["Age"].isnull(), np.nanmean(ti["Age"]), ti["Age"])
ti.info()

print(
pd.cut(ti["Age"],bins=5).value_counts()
)
print(
pd.cut(ti["Age"],bins=5,labels=["Young","YoungAdult","Adult","Senior","Old"]).value_counts()
)
ti["Age"]=pd.cut(ti["Age"],bins=5,labels=["Young","YoungAdult","Adult","Senior","Old"])


print(
      pd.qcut(ti["Fare"],q=8).value_counts()
      )
ti["Fare"]=pd.qcut(ti["Fare"],q=8,
                   labels=["0~7","7~8","9~12","13~15","16~26","27~35","36~76","77~512"])

ti["SibSp"]=ti["SibSp"].astype(str)

from sklearn.preprocessing import scale
ti["Parch"]=scale(ti["Parch"])

ti.to_csv("newTitanic.csv", index=False, encoding="utf_8_sig")

























