#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:06:37 2023

@author: mac

"""
import numpy as np

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

data=[[18,175,60],[20,171,58],[25,178,62]]
ss.fit(data)
print(ss.mean_) 
print(ss.var_) 

print(ss.transform(data)) 

print((data-ss.mean_)/np.sqrt(ss.var_)) #土法煉鋼










