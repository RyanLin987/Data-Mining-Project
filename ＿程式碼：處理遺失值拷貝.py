#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:48:03 2023

@author: mac
Topic: Missing Value

"""
import pandas as pd

bank=pd.read_csv("#2-bank-data(1).csv")
#bank.info()

import pandas as pd
import numpy as np

bank=bank.dropna()

new=bank.drop(["id"],axis=1,inplace=True)

bank["children"]=bank["children"].astype(str)#數字轉為名目尺度

bank["age"]=pd.cut(bank["age"],bins=3, labels=["a","b","c"])#等距
bank["income"]=pd.qcut(bank["income"],q=3, labels=["L","M","H"])#等次數

bank.info()

print(bank["age"].value_counts())
print(bank["income"].value_counts())

'''
#插補法
bank["age"]=np.where(bank["age"].isnull(),
                     np.nanmedian(bank["age"]),
                     bank["age"])

bank["income"]=np.where(bank["income"].isnull(),
                     np.nanmean(bank["income"]),
                     bank["income"])

import statistics

bank["married"]=np.where(bank["married"].isnull(),
                     statistics.mode(bank["married"]),
                     bank["married"])

'''





