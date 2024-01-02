# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

###pandas
import pandas as pd
#concat
df1=pd.DataFrame({
    "A":["A0","A1","A2","A3"],
    "B":["B0","B1","B2","B3"],
    "C":["C0","C1","C2","C3"],
    "D":["D0","D1","D2","D3"] 
    },index=[0,1,2,3])
df2=pd.DataFrame({
    "A":["A4","A5","A6","A7"],
    "B":["B4","B5","B6","B7"],
    "C":["C4","C5","C6","C7"],
    "D":["D4","D5","D6","D7"] 
    },index=[4,5,6,7])
df3=pd.DataFrame({
    "A":["A8","A9","A10","A11"],
    "B":["B8","B9","B10","B11"],
    "C":["C8","C9","C10","C11"],
    "D":["D8","D9","D10","D11"] 
    },index=[8,9,10,11])
result1=pd.concat([df1,df2,df3],axis=0)
result2=pd.concat([df1,df2,df3],axis=1)


df4=pd.DataFrame({
    "E":["E2","E3","E6","E7"],
    "F":["F2","F3","F6","F7"],
    "G":["G2","G3","G6","G7"],
    "H":["H2","H3","H6","H7"] 
    },index=[2,3,6,7])
result3=pd.concat([df1,df4],axis=1)

#merge
left=pd.DataFrame({
    "key":["K0","K1","K2","K3"],
    "A":["A0","A1","A2","A3"],
    "B":["B0","B1","B2","B3"]
    })
right=pd.DataFrame({
    "key":["K0","K1","K2","K3"],
    "C":["C0","C1","C2","C3"],
    "D":["D0","D1","D2","D3"]
    })
result4=pd.merge(left,right,on="key")




