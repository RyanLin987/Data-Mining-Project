#%%0 資料前處理1
import pandas as pd
import numpy as np
#origin=pd.read_csv("2014-2021消防機關水域救援統計.csv", encoding="cp950")
data = pd.read_csv("2014-2021消防機關水域救援統計.csv", encoding="cp950")
# data.info()



data.drop(0,axis=0,inplace=True) #刪除中文欄位名
data.drop(["Number","Location_of_drowning"],axis=1,inplace=True) #刪除欄位
## 刪除理由：
# Number
# Location_of_drowning
# Patient_ID
# Swimming_skills



#調整年齡欄位
data["Age"]=np.where(data["Age"].isnull(), data["Unnamed: 13"], data["Age"])
data.drop("Unnamed: 13",axis=1,inplace=True)



#修正欄位名
data.rename(columns={'Types_of _waters': 'Types_of_waters'}, inplace=True)
# data.info()



#值的轉換（數值型）
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
data['Minute'] = pd.to_numeric(data['Minute'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
#值的轉換（字串型）
data['City_or_County'] = data['City_or_County'].str.strip()
data['Types_of_waters'] = data['Types_of_waters'].str.strip()
data['Drowning_reasons'] = data['Drowning_reasons'].str.strip()
data['Drowning_results'] = data['Drowning_results'].str.strip()
data['Gender'] = data['Gender'].str.strip()
# data.info()



#擷取需要的年份資料
data = data[(data["Year"]==2020)| (data["Year"]==2021)]
data["Age"]=np.where(data["Age"].isnull(), np.nanmedian(data["Age"]), data["Age"])






#%%0 資料前處理2
data['City_or_County'].value_counts() #衍生出區域
data['Year'].value_counts()  
data['Month'].value_counts() #衍生出季節
data['Day'].value_counts()  #衍生出日期
data['Hour'].value_counts() #衍生出時段
data['Minute'].value_counts()
data['Types_of_waters'].value_counts()
data['Drowning_reasons'].value_counts() #格式需處理
data['Drowning_results'].value_counts() #目標變數，將失蹤歸類進死亡
data['Gender'].value_counts() #不祥則填入眾數
data['Age'].value_counts() 
data['Patient_ID'].value_counts()
data['Swimming_skills'].value_counts()
# data["Age"]=np.where(data["Age"], , )


data['City_or_County'].value_counts() #衍生出區域
county_to_region = {
    '臺北市': '北部',
    '新北市': '北部',
    '桃園市': '北部',
    '臺中市': '中部',
    '臺南市': '南部',
    '高雄市': '南部',
    '基隆市': '北部',
    '新竹市': '北部',
    '嘉義市': '南部',
    '新竹縣': '北部',
    '苗栗縣': '中部',
    '彰化縣': '中部',
    '南投縣': '中部',
    '雲林縣': '中部',
    '嘉義縣': '南部',
    '屏東縣': '南部',
    '宜蘭縣': '東部',
    '花蓮縣': '東部',
    '臺東縣': '東部',
    '澎湖縣': '外島',
    '金門縣': '外島',
    '連江縣': '外島'
}
data['Region'] = data['City_or_County'].map(county_to_region)
data['Region'].value_counts()



#年月日衍生出日期
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']]) 
#日期衍生出星期幾
data['Day_of_Week'] = data['Date'].dt.day_name()
#再衍生出是否為假日：
from datetime import date
import holidays
tw_holidays = holidays.TW()
data['Is_Holiday'] = data['Date'].apply(lambda x: x in tw_holidays)
data["Is_Holiday"]=np.where(
    (data["Day_of_Week"]=="Saturday") | (data["Day_of_Week"]=="Sunday"), 
    True, data["Is_Holiday"])



data['Month'].value_counts() #月衍生出季節
def Season(Month):
    if 3 <= Month <= 5:
        return 'Spring'
    if 6 <= Month <= 8:
        return 'Summer'
    if 9 <= Month <= 11:
        return 'Fall'        
    else:
        return 'Winter'
data['Season'] = data['Month'].apply(Season) 
data['Season'].value_counts()



data['Hour'].value_counts() #時衍生出時段
def time_period(Hour):
    if 7 <= Hour <= 10:
        return 'Morning'
    if 11 <= Hour <= 14:
        return 'Noon'
    if 15 <= Hour <= 17:
        return 'Evening'   
    if 18 <= Hour <= 23:
        return 'Night'       
    else:
        return 'Dawn'
data['time_period'] = data['Hour'].apply(time_period) 
data['time_period'].value_counts()
   


data['Drowning_reasons'].value_counts() #格式需處理
data['Drowning_reasons'] = data['Drowning_reasons'].str.replace(r'\(.*\)', '', regex=True)



data['Drowning_results'].value_counts() #目標變數，將失蹤歸類進死亡
data["Drowning_results"]=np.where(data["Drowning_results"]=="失蹤", "死亡", data["Drowning_results"])



data['Gender'].value_counts() #不詳則填入眾數
import statistics
data["Gender"]=np.where(data["Gender"]=="不詳", statistics.mode(data["Gender"]), data["Gender"])
# data.info()



# data['Age'].value_counts() #要做離散化
# pd.cut(data["Age"],bins=5).value_counts()
# pd.qcut(data["Age"],q=8).value_counts()
# data["Age"]=pd.cut(data["Age"],bins=3, labels=["a","b","c"])#等距
# data["Age"]=pd.qcut(data["Age"],q=3, labels=["L","M","H"])#等次數


#%%0 特徵選取


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# data.info()
City_or_County=le.fit_transform(data["City_or_County"])
Types_of_waters=le.fit_transform(data["Types_of_waters"])
Drowning_reasons=le.fit_transform(data["Drowning_reasons"])
Gender=le.fit_transform(data["Gender"])
Patient_ID=le.fit_transform(data["Patient_ID"])
Swimming_skills=le.fit_transform(data["Swimming_skills"])
Region=le.fit_transform(data["Region"])
Day_of_Week=le.fit_transform(data["Day_of_Week"])
Is_Holiday=le.fit_transform(data["Is_Holiday"])
Season=le.fit_transform(data["Season"])
time_period=le.fit_transform(data["time_period"])




X=pd.DataFrame([City_or_County,data["Year"],data["Month"],data["Day"],
                data["Hour"],data["Minute"],Types_of_waters,Drowning_reasons,
                Gender,data["Age"],Patient_ID,Swimming_skills,Region,
                Day_of_Week,Is_Holiday,Season,time_period]).T
X.columns=["City_or_County","Year","Month","Day","Hour","Minute",
           "Types_of_waters","Drowning_reasons","Gender","Age","Patient_ID",
           "Swimming_skills","Region","Day_of_Week",
           "Is_Holiday","Season","time_period"]
y=data["Drowning_results"].values #讓index變成從0開始
y=pd.Series(y,name='Drowning_results')



#卡方挑變數
from sklearn.feature_selection import SelectKBest, chi2
sk=SelectKBest(chi2,k=3)
sk.fit(X,y)
print(sk.get_feature_names_out()) 
scores = sk.scores_
rounded_scores = [round(score, 2) for score in scores] #格式：小數第二位
df=pd.DataFrame([X.columns,rounded_scores]).T
df.columns=["Variable", "score"]
# 按照 Rounded_Scores 降序排序
df_sorted = df.sort_values(by="score", ascending=False)
print(df_sorted)








#決定要用的變數：
data.info()
selected_X=data[["Age","Types_of_waters","Season","Is_Holiday","Drowning_reasons",
        "time_period","Swimming_skills","Gender","Region"]]
y=y



#%%0 切割訓練集與測試集：

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test=train_test_split(
#     X,y,test_size=0.2, random_state=20240104)


#%%
# 假設 data 是你的資料框
# # 使用條件選擇找到 "Drowning_reasons" 為 "浮屍" 且 "Drowning_results" 為 "死亡" 的資料
# filtered_data = data[(data['Drowning_reasons'] == '浮屍') & (data['Drowning_results'] == '死亡')]
# # 計算滿足條件的資料筆數
# count_of_deaths = len(filtered_data)
# # 印出結果
# print(f"在 'Drowning_reasons' 為 '浮屍' 的資料中，有 {count_of_deaths} 筆 'Drowning_results' 為 '死亡'。")



# # 印出結果
# print(rounded_scores)

# X_new1=sk.transform(X)
# X_new1=pd.DataFrame(X_new1)
# X_new1.columns=[
#     'Minute','Types_of_waters', 'Age', 'Patient_ID', 'Season']



# #建立clf模型
# from sklearn.tree import DecisionTreeClassifier
# clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
#                             min_samples_split=0.25,
#                             min_samples_leaf=2)
# clf.fit(X,y)
# print("建模方法的挑選變數正確率＝",clf.score(X, y))#0.6933
# print(clf.feature_importances_)
# #看看clf模型挑了哪些變數
# from sklearn.feature_selection import SelectFromModel
# sm=SelectFromModel(clf,max_features=5)
# sm.fit(X,y)
# print(sm.get_feature_names_out())
# X_new2=sm.transform(X)
# X_new2=pd.DataFrame(X_new2)
# X_new2.columns=["income","married","children","save","mortage"]




#%%1 決策樹1

data.info()
selected_X.info()


#編碼
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

Types_of_waters=le.fit_transform(selected_X["Types_of_waters"])
Season=le.fit_transform(selected_X["Season"])
Is_Holiday=le.fit_transform(selected_X["Is_Holiday"])
Drowning_reasons=le.fit_transform(selected_X["Drowning_reasons"])
time_period=le.fit_transform(selected_X["time_period"])
Swimming_skills=le.fit_transform(selected_X["Swimming_skills"])
Gender=le.fit_transform(selected_X["Gender"])
Region=le.fit_transform(selected_X["Region"])
Age=selected_X["Age"].values

X=pd.DataFrame([Age,Types_of_waters,Season,Is_Holiday,
                 Drowning_reasons,time_period,Swimming_skills,Gender,Region]).T
X.columns=["Age","Types_of_waters","Season","Is_Holiday","Drowning_reasons",
        "time_period","Swimming_skills","Gender","Region"]
y=y



# 挑選決策樹要使用的變數：
print(df_sorted[df_sorted["Variable"].isin(X.columns)])



#看看clf模型挑了哪些變數:
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_split=0.25,
                           min_samples_leaf=2)
clf.fit(X,y)
print("建模方法的挑選變數正確率＝",clf.score(X, y))#0.7251
print(clf.feature_importances_) #發現第五個變數(Drowning_reasons)就佔了94.63%

from sklearn.feature_selection import SelectFromModel
sm=SelectFromModel(clf,max_features=5)
sm.fit(X,y)
print(sm.get_feature_names_out()) #只挑了"Drowning_reasons"一個



#觀察"Drowning_reasons"是否過度與目標變數重疊：
print(pd.crosstab(selected_X["Drowning_reasons"], y, 
    rownames=['Drowning_reasons'], colnames=['y']))##發現"浮屍"全數死亡




#將Drowning_reasons為"浮屍"的資料的資料，按比例隨機填入該變數的其他類別：

# 確定比例，這裡假設按照其他類別的比例進行填充
fill_ratio = selected_X[selected_X['Drowning_reasons'] != '浮屍']['Drowning_reasons'].value_counts(normalize=True)
# 找到 "浮屍" 資料的索引
drowning_indices = selected_X[selected_X['Drowning_reasons'] == '浮屍'].index

# 遍歷每個 "浮屍" 資料，按照其他類別的比例隨機填入其他類別
for idx in drowning_indices:
    # 隨機選擇其他類別
    random_other_reason = np.random.choice(fill_ratio.index, p=fill_ratio.values)
    # 將選擇的其他類別填入
    selected_X.at[idx, 'Drowning_reasons'] = random_other_reason



#將Drowning_reasons重新編碼
Drowning_reasons=le.fit_transform(selected_X["Drowning_reasons"])
X=pd.DataFrame([Age,Types_of_waters,Season,Is_Holiday,
                 Drowning_reasons,time_period,Swimming_skills,Gender,Region]).T
X.columns=["Age","Types_of_waters","Season","Is_Holiday","Drowning_reasons",
        "time_period","Swimming_skills","Gender","Region"]
y=y



#處理完浮屍後再來挑選一次:
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_split=0.25,
                           min_samples_leaf=2)
clf.fit(X,y)
print("建模方法的挑選變數正確率＝",clf.score(X, y))#0.7035
print(clf.feature_importances_)

from sklearn.feature_selection import SelectFromModel
sm=SelectFromModel(clf,max_features=5)
sm.fit(X,y)
print(sm.get_feature_names_out()) #這次挑了2個

importances=clf.feature_importances_
rounded_importances = [round(importance, 2) for importance in importances] #格式：小數第二位
df2=pd.DataFrame([X.columns,rounded_importances]).T
df2.columns=["Variable", "importance"]
df_sorted2 = df2.sort_values(by="importance", ascending=False)
print(df_sorted2)


X_new2=sm.transform(X)
X_new2=pd.DataFrame(X_new2)
X_new2.columns=["Age","Types_of_waters"]







#%%1 決策樹:全變數(訓練與測試)

# #上採樣
# from imblearn.over_sampling import SMOTE
# X,y=SMOTE(random_state=0).fit_resample(X, y)
# print("SMOTE")
# print(y.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20240104)



#%%1 決策樹:全變數建模(gini)調參：
clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_leaf=80, min_samples_split=0.3)
clf.fit(X_train,y_train)
print("建模正確率=",format(clf.score(X_train,y_train)*100,".2f"),"%")
print("測試正確率=",format(clf.score(X_test,y_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf.get_n_leaves())
print("樹的深度有多少層=",clf.get_depth())


# clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
#                            min_samples_leaf=10, min_samples_split=0.01)
# clf.fit(X_train,y_train)
# print("建模正確率=",format(clf.score(X_train,y_train)*100,".2f"),"%")
# print("測試正確率=",format(clf.score(X_test,y_test)*100,".2f"),"%")
# print("樹的葉子有多少個=",clf.get_n_leaves())
# print("樹的深度有多少層=",clf.get_depth())

from sklearn import tree
dot_data=tree.export_graphviz(clf,out_file=None,
                              feature_names=X.columns,
                              leaves_parallel=False,
                              impurity=True,
                              proportion=True,
                              rounded=True,
                              class_names=["死亡","獲救"],
                              filled=True
                              )
import os
os.environ["PATH"] = "/opt/local/bin/"
import graphviz
graph=graphviz.Source(dot_data)
graph.format="png"
graph.render("tree_gini",view=False)



#%%1 決策樹:全變數建模(entropy)
clf=DecisionTreeClassifier(random_state=20240104,criterion="entropy",
                            min_samples_leaf=80, min_samples_split=0.3)
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
                              rounded=True,
                              class_names=["死亡","獲救"],
                              filled=True
                              )
import os
os.environ["PATH"] = "/opt/local/bin/"
import graphviz
graph=graphviz.Source(dot_data)
graph.format="png"
graph.render("tree_entropy",view=False)





from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X,y, cv=4, scoring="accuracy")
print("交叉驗證的正確率=",format(scores.mean()*100,".2f")+"%")





#%%1 決策樹:最佳變數(訓練與測試)


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test=train_test_split(
    X_new2,y,test_size=0.2, random_state=20240104)



#%%1 決策樹:最佳變數建模(gini)調參：
clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_leaf=80, min_samples_split=0.3)
clf.fit(X2_train,y2_train)
print("建模正確率=",format(clf.score(X2_train,y2_train)*100,".2f"),"%")
print("測試正確率=",format(clf.score(X2_test,y2_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf.get_n_leaves())
print("樹的深度有多少層=",clf.get_depth())

clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_leaf=10, min_samples_split=0.01)
clf.fit(X2_train,y2_train)
print("建模正確率=",format(clf.score(X2_train,y2_train)*100,".2f"),"%")
print("測試正確率=",format(clf.score(X2_test,y2_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf.get_n_leaves())
print("樹的深度有多少層=",clf.get_depth())

from sklearn import tree
dot_data=tree.export_graphviz(clf,out_file=None,
                              feature_names=X.columns,
                              leaves_parallel=False,
                              impurity=True,
                              proportion=True,
                              rounded=True,
                              class_names=["死亡","獲救"],
                              filled=True
                              )
import os
os.environ["PATH"] = "/opt/local/bin/"
import graphviz
graph=graphviz.Source(dot_data)
graph.format="png"
graph.render("tree_gini",view=False)




#%%1 決策樹:最佳變數建模(entropy)調參：
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=20240104,criterion="entropy",
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
                              rounded=True,
                              class_names=["死亡","獲救"],
                              filled=True
                              )
import os
os.environ["PATH"] = "/opt/local/bin/"
import graphviz
graph=graphviz.Source(dot_data)
graph.format="png"
graph.render("tree_entropy",view=False)



from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X_new2,y, cv=4, scoring="accuracy")
print("交叉驗證的正確率=",format(scores.mean()*100,".2f")+"%")


#%%1 類別編號對應


X.info()

original_labels=selected_X["Types_of_waters"]
df_Types_of_waters =  pd.DataFrame([Types_of_waters,original_labels]).T
df_Types_of_waters.value_counts()

original_labels=selected_X["Season"]
df_Season =  pd.DataFrame([Season,original_labels]).T
df_Season.value_counts()

original_labels=selected_X["Is_Holiday"]
df_Is_Holiday =  pd.DataFrame([Is_Holiday,original_labels]).T
df_Is_Holiday.value_counts()

original_labels=selected_X["Drowning_reasons"]
df_Drowning_reasons =  pd.DataFrame([Drowning_reasons,original_labels]).T
df_Drowning_reasons.value_counts()
    
original_labels=selected_X["time_period"]
df_time_period =  pd.DataFrame([time_period,original_labels]).T
df_time_period.value_counts()

original_labels=selected_X["Swimming_skills"]
df_Swimming_skills =  pd.DataFrame([Swimming_skills,original_labels]).T
df_Swimming_skills.value_counts()

original_labels=selected_X["Gender"]
df_Gender =  pd.DataFrame([Gender,original_labels]).T
df_Gender.value_counts()

original_labels=selected_X["Region"]
df_Region =  pd.DataFrame([Region,original_labels]).T
df_Region.value_counts()


#%% 決策樹(輸出給Ｒ)


df_R = pd.concat([X,y],axis=1)
df_R.to_csv('new_data.csv', index=False)




#%%2 關聯法則

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








#%%3 SVM


import pandas as pd
bank=pd.read_csv("bank-data(3).csv")
bank.info()
subdata=bank[(bank["region"]=="INNER_CITY")|
             (bank["region"]=="TOWN")|(bank["region"]=="RURAL")]

x_train,y_train=subdata[["age","income"]],subdata["region"]
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(x_train)
x_train_std=ss.transform(x_train)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_train=le.fit_transform(y_train)

from sklearn.svm import LinearSVC
m=LinearSVC(C=0.1, dual=False, class_weight="balanced")
m.fit(x_train_std, y_train)
y_pred=m.predict(x_train_std)
print("訓練資料集正確率＝", m.score(x_train_std, y_train))
print("訓練資料集分類錯誤筆數＝", (y_train!=y_pred).sum())
print(1-(y_train!=y_pred).sum()/538)
from sklearn.metrics import f1_score
print("F1-score=", f1_score(y_train, y_pred, average="weighted"))










#%%4 RF

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









#%%5 KNN


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







#%%7 k-means

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:49:09 2023

@author: mac
"""

import pandas as pd
bank=pd.read_csv("bank-data(3).csv") 
#bank.info()
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
married=le.fit_transform(bank["married"])
children=le.fit_transform(bank["children"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])
pep=le.fit_transform(bank["pep"])
X=pd.DataFrame([scale(bank["age"]),sex,scale(bank["income"]),
                married,children,car,save_act,current_act,mortgage,pep]).T
X.columns=["age","sex","income","married","children","car","save_act","current_act","mortgage","pep"]


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
region=ohe.fit_transform(bank[["region"]])
region=pd.DataFrame(region)
region.columns=ohe.categories_[0]

newX=pd.concat([X,region],axis=1)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortion=[]
for i in range(10):
    kmeans=KMeans(n_clusters=i+1, init="k-means++",
                  random_state=20231221, n_init=15, max_iter=200)
    kmeans.fit(newX)
    distortion.append(kmeans.inertia_)
    
print(distortion)    

plt.plot(range(1,11),distortion, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

kmeans=KMeans(n_clusters=4, init="k-means++",
              random_state=20231221, n_init=15, max_iter=200)
kmeans.fit(newX)
centroid=pd.DataFrame(kmeans.cluster_centers_,columns=newX.columns)

'''
X_pred=kmeans.predict(newX)
print(pd.crosstab(bank["pep"], X_pred))
print("用分群來預測分類的正確率＝",(75+113+61+99)/600)
'''









































