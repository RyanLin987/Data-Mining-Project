


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:16:20 2023

## basket data analysis
"""
#df.info()

import pandas as pd
df=pd.read_csv("Retail.csv")
df["Description"]=df["Description"].str.strip()
df.dropna(axis=0,subset=["InvoiceNo"],inplace=True)
df["InvoiceNo"]=df["InvoiceNo"].astype("str")
df=df[~df["InvoiceNo"].str.contains("C")]#不要含C的，C開頭代表退貨

basket=(
        df[df["Country"]=="France"]
       .groupby(["InvoiceNo","Description"])["Quantity"]
       .sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")
       )





def encode_units(x):
    if x<=0:
        return 0
    else:
        return 1

basket_sets=basket.applymap(encode_units)
basket_sets.drop("POSTAGE",inplace=True,axis=1)

#pip install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets=apriori(basket_sets, min_support=0.07, use_colnames=True)
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.1)
rules.to_csv("rules.csv")








