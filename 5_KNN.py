
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:03:37 2023

KNN
"""

import pandas as pd
import numpy as np
import statistics

bank=pd.read_csv("#2-bank-data(1).csv")

bank["age"]= np.where(bank["age"].isnull(),
                      np.nanmedian(bank["age"]), bank["age"])
bank["married"]= np.where(bank["married"].isnull(),
                      statistics.mode(bank["married"]), bank["married"])

bank.info()

from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(sparse=False)
sex=ohe.fit_transform(bank[["sex"]])
sex=pd.DataFrame(sex)
sex.columns=ohe.categories_[0]

region=ohe.fit_transform(bank[["region"]])
region=pd.DataFrame(region)
region.columns=ohe.categories_[0]

married=ohe.fit_transform(bank[["married"]])
married=pd.DataFrame(married)
#married.columns=ohe.categories_[0]
married.columns=["married_"+s for s in ohe.categories_[0]]

car=ohe.fit_transform(bank[["car"]])
car=pd.DataFrame(car)
car.columns=["car_"+s for s in ohe.categories_[0]]

save_act=ohe.fit_transform(bank[["save_act"]])
save_act=pd.DataFrame(save_act)
save_act.columns=["save_act_"+s for s in ohe.categories_[0]]

current_act=ohe.fit_transform(bank[["current_act"]])
current_act=pd.DataFrame(current_act)
current_act.columns=["current_act_"+s for s in ohe.categories_[0]]

mortgage=ohe.fit_transform(bank[["mortgage"]])
mortgage=pd.DataFrame(mortgage)
mortgage.columns=["mortgage_"+s for s in ohe.categories_[0]]


X1=pd.concat([bank["age"],bank["income"],bank["children"]], axis=1)
              
              
from sklearn.preprocessing import StandardScaler
ss=StandardScaler().fit(X1)
X1=ss.transform(X1)
X1=pd.DataFrame(X1)
X1.columns=["age","income","children"]
              
X=pd.concat([X1["age"],sex,region,X1["income"],married,
            X1["children"],car,save_act,current_act,mortgage], axis=1)

y=bank["pep"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20231207)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

print("建模正確率:",knn.score(X_train, y_train))
y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score
print("測試正確率:",accuracy_score(y_test, y_pred))


acc=[]
for i in range(1,481):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    print("k=",i,"的測試正確率＝",accuracy_score(y_test, y_pred))
    acc.append(accuracy_score(y_test, y_pred))
  
print("測試正確率最高的＝",max(acc))

for i in range(1,481):
    if acc[i-1]==max(acc):
        bestK=i
 
print("K=",bestK,"的測試正確率最高=",max(acc))        





























