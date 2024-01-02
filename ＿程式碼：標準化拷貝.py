#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 18:49:31 2023

@author: mac
Topic: 標準化
"""
import numpy as np
from sklearn.preprocessing import scale
data=[1,2,3,4,5,6,7,8,9,10]

average=np.mean(data)
std=np.std(data)
print(average)
print(std)

print(data-average)
print(scale(data,with_std=False))

print(data/std)
print(scale(data,with_mean=False))

print((data-average)/std)
print(scale(data))








