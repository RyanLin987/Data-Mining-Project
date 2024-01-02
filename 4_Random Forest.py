
"""
Created on Thu Dec 14 19:59:19 2023

@author: mac
"""

import pandas as pd
bank=pd.read_csv("bank-data(3).csv")
bank.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
married=le.fit_transform(bank["married"])
children=le.fit_transform(bank["children"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])
X=pd.DataFrame([bank["age"],sex,bank["income"],married,
               children,car,save_act,current_act,mortgage]).T
X.columns=["age","sex","income","married",
           "children","car","save_act","current_act","mortgage"]

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
region=ohe.fit_transform(bank[["region"]])
region=pd.DataFrame(region)
region.columns=ohe.categories_[0]

newX=pd.concat([X,region],axis=1)
y=bank["pep"]

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=200, max_depth=8, random_state=20231214)
clf.fit(newX,y)
print("隨機森林正確率＝", clf.score(newX, y))









