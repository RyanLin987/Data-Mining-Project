#%%P 資料前處理1
import pandas as pd
import numpy as np
#origin=pd.read_csv("2014-2021消防機關水域救援統計.csv", encoding="cp950")
data = pd.read_csv("2014-2021消防機關水域救援統計.csv", encoding="cp950")
# data.info()

data.drop(0,axis=0,inplace=True) #刪除中文欄位名
data.drop(["Number","Location_of_drowning","Swimming_skills"],axis=1,inplace=True) #刪除欄位
## 刪除理由：
# Number
# Location_of_drowning
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
data['Patient_ID'] = data['Patient_ID'].str.strip()
data['Gender'] = data['Gender'].str.strip()
# data.info()

#擷取需要的年份資料
data = data[(data["Year"]==2020)| (data["Year"]==2021)]
data["Age"]=np.where(data["Age"].isnull(), np.nanmedian(data["Age"]), data["Age"])
#%%P 資料前處理2
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



#決定要用的變數：
selected_X=data[["Types_of_waters","Drowning_reasons","Gender","Age","Region",
        "Is_Holiday","Season","time_period"]].reset_index(drop=True)

y=data["Drowning_results"].reset_index(drop=True) #讓index變成從0開始


# data['Age'].value_counts() #離散化
# pd.cut(data["Age"],bins=5).value_counts()
# pd.qcut(data["Age"],q=8).value_counts()
# data["Age"]=pd.cut(data["Age"],bins=3, labels=["a","b","c"])#等距
# data["Age"]=pd.qcut(data["Age"],q=3, labels=["L","M","H"])#等次數
#%%0 特徵選取1(卡方)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# selected_X.info()
#編碼
Types_of_waters=le.fit_transform(data["Types_of_waters"])
Drowning_reasons=le.fit_transform(data["Drowning_reasons"])
Gender=le.fit_transform(data["Gender"])
Region=le.fit_transform(data["Region"])
Is_Holiday=le.fit_transform(data["Is_Holiday"])
Season=le.fit_transform(data["Season"])
time_period=le.fit_transform(data["time_period"])

X=pd.DataFrame([Types_of_waters,Drowning_reasons,Gender,data["Age"],Region,
                Is_Holiday,Season,time_period]).T
X.columns=["Types_of_waters","Drowning_reasons","Gender","Age","Region",
           "Is_Holiday","Season","time_period"]
y=data["Drowning_results"].reset_index(drop=True)



#卡方挑變數
from sklearn.feature_selection import SelectKBest, chi2
sk=SelectKBest(chi2,k=5)
sk.fit(X,y)
print(sk.get_feature_names_out())  
#['Types_of_waters' 'Drowning_reasons' 'Age' 'Is_Holiday' 'Season']

scores = sk.scores_
rounded_scores = [round(score, 2) for score in scores] 
df=pd.DataFrame([X.columns,rounded_scores]).T
df.columns=["Variable", "score"]
df_sorted = df.sort_values(by="score", ascending=False)
print(df_sorted) #列出每個變數的score，按照score高低排列

X_new1=sk.transform(X)
X_new1=pd.DataFrame(X_new1)
X_new1.columns=['Types_of_waters','Drowning_reasons','Age','Is_Holiday','Season']
#%%0 特徵選取2(決策樹建模)(處理浮屍)


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



#篩取其他資料
data1=pd.concat([selected_X,y],axis=1)
data1=data1[data1['Drowning_reasons'] != '浮屍']



#將Drowning_reasons為"浮屍"的資料的資料，按比例隨機填入該變數的其他類別：

# # 確定比例，這裡假設按照其他類別的比例進行填充
# fill_ratio = selected_X[selected_X['Drowning_reasons'] != '浮屍']['Drowning_reasons'].value_counts(normalize=True)
# # 找到 "浮屍" 資料的索引
# drowning_indices = selected_X[selected_X['Drowning_reasons'] == '浮屍'].index

# # 遍歷每個 "浮屍" 資料，按照其他類別的比例隨機填入其他類別
# for idx in drowning_indices:
#     # 隨機選擇其他類別
#     random_other_reason = np.random.choice(fill_ratio.index, p=fill_ratio.values)
#     # 將選擇的其他類別填入
#     selected_X.at[idx, 'Drowning_reasons'] = random_other_reason
#%% 輸出給Ｒ
data1.to_csv('new_data.csv', index=False)
#%%0 特徵選取3(卡方/決策樹建模)(無浮屍的data1)
# data1.info()
#編碼
Types_of_waters=le.fit_transform(data1["Types_of_waters"])
Drowning_reasons=le.fit_transform(data1["Drowning_reasons"])
Gender=le.fit_transform(data1["Gender"])
Age=le.fit_transform(data1["Age"])
Region=le.fit_transform(data1["Region"])
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Season=le.fit_transform(data1["Season"])
time_period=le.fit_transform(data1["time_period"])

X=pd.DataFrame([Types_of_waters,Drowning_reasons,Gender,data1["Age"],
                 Region,Is_Holiday,Season,time_period]).T
X.columns=["Types_of_waters","Drowning_reasons","Gender","Age",
           "Region","Is_Holiday","Season","time_period"]
y=data1["Drowning_results"]



#1.處理完浮屍後再卡方挑選一次:
from sklearn.feature_selection import SelectKBest, chi2
sk=SelectKBest(chi2,k=5)
sk.fit(X,y)
print(sk.get_feature_names_out())
#變成['Types_of_waters' 'Drowning_reasons' 'Gender' 'Age' 'Season']
X_new1=sk.transform(X)
X_new1=pd.DataFrame(X_new1)
X_new1.columns=['Types_of_waters','Drowning_reasons','Gender','Age','Season']

scores = sk.scores_
rounded_scores = [round(score, 2) for score in scores] 
df=pd.DataFrame([X.columns,rounded_scores]).T
df.columns=["Variable", "score"]
df_sorted = df.sort_values(by="score", ascending=False)
print(df_sorted)#列出每個變數的score，按照score高低排列，取前四高(Season與Gender落差大)
X_top4=X[['Age','Drowning_reasons','Types_of_waters','Season']]



#2.處理完浮屍後再用決策樹建模挑選一次:
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_split=0.25,
                           min_samples_leaf=2)
clf.fit(X,y)
print("建模方法的挑選變數正確率＝",clf.score(X, y))#0.6417
print(clf.feature_importances_)

from sklearn.feature_selection import SelectFromModel
sm=SelectFromModel(clf,max_features=5)
sm.fit(X,y)
print(sm.get_feature_names_out())#['Types_of_waters','Drowning_reasons','Age']

importances=clf.feature_importances_
rounded_importances = [round(importance, 2) for importance in importances] #格式：小數第二位
df2=pd.DataFrame([X.columns,rounded_importances]).T
df2.columns=["Variable", "importance"]
df_sorted2 = df2.sort_values(by="importance", ascending=False)
print(df_sorted2)#列出每個變數的importances，按照importances高低排列

X_new2=sm.transform(X)
X_new2=pd.DataFrame(X_new2)
X_new2.columns=['Types_of_waters','Drowning_reasons','Age']
#%%0 特徵選取4(比較)

#比較卡方與建模方法挑選變數的建模正確率：
X.info()#全變數，共八個
X_new1.info()#五個，多出Gender,Season
X_top4.info()#四個，多出Gender
X_new2.info()#三個

clf=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_split=0.2,
                           min_samples_leaf=50,
                           max_depth=20)

clf.fit(X, y)
print("使用全部八個變數的模型正確率＝",
      format(clf.score(X, y)*100,".2f"),"%","樹深度為",clf.get_depth())
clf.fit(X_new1, y)
print("卡方分配法找出的五個變數的模型正確率＝",
      format(clf.score(X_new1, y)*100,".2f"),"%","樹深度為",clf.get_depth())
clf.fit(X_top4, y)
print("卡方分配法找出的前四變數的模型正確率＝",
      format(clf.score(X_top4, y)*100,".2f"),"%","樹深度為",clf.get_depth())
clf.fit(X_new2, y)
print("建模方法找出的三個變數的模型正確率=",
      format(clf.score(X_new2, y)*100,".2f"),"%","樹深度為",clf.get_depth())

#%%1 決策樹(全變數X編碼與分割)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#編碼
Types_of_waters=le.fit_transform(data1["Types_of_waters"])
Drowning_reasons=le.fit_transform(data1["Drowning_reasons"])
Gender=le.fit_transform(data1["Gender"])
Age=le.fit_transform(data1["Age"])
Region=le.fit_transform(data1["Region"])
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Season=le.fit_transform(data1["Season"])
time_period=le.fit_transform(data1["time_period"])

X=pd.DataFrame([Types_of_waters,Drowning_reasons,Gender,data1["Age"],
                 Region,Is_Holiday,Season,time_period]).T
X.columns=["Types_of_waters","Drowning_reasons","Gender","Age",
           "Region","Is_Holiday","Season","time_period"]
y=data1["Drowning_results"]

#分割訓練測試
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20240104)
#%%1 決策樹:全變數X建模

# gini
clf1=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_leaf=5, min_samples_split=0.1)
clf1.fit(X_train,y_train)
print("全變數gini建模正確率=",format(clf1.score(X_train,y_train)*100,".2f"),"%")
print("全變數gini測試正確率=",format(clf1.score(X_test,y_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf1.get_n_leaves())
print("樹的深度有多少層=",clf1.get_depth())

# entropy
clf2=DecisionTreeClassifier(random_state=20240104,criterion="entropy",
                           min_samples_leaf=5, min_samples_split=0.1)
clf2.fit(X_train,y_train)
print("全變數entropy建模正確率=",format(clf2.score(X_train,y_train)*100,".2f"),"%")
print("全變數entropy測試正確率=",format(clf2.score(X_test,y_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf2.get_n_leaves())
print("樹的深度有多少層=",clf2.get_depth())

#繪圖(選_)
from sklearn import tree
dot_data=tree.export_graphviz(clf1,out_file=None,
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
graph.render("tree_全變數",view=False)
#%%1 決策樹:全變數X建模(entropy)：clf3,4
#3
clf3=DecisionTreeClassifier(random_state=20240104,criterion="entropy",
                            min_samples_leaf=80, min_samples_split=0.3)
clf3.fit(X_train,y_train)
print("建模正確率=",format(clf3.score(X_train,y_train)*100,".2f"),"%")
print("測試正確率=",format(clf3.score(X_test,y_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf3.get_n_leaves())
print("樹的深度有多少層=",clf3.get_depth())

#4
clf4=DecisionTreeClassifier(random_state=20240104,criterion="entropy",
                            min_samples_leaf=80, min_samples_split=0.3)
clf4.fit(X_train,y_train)
print("建模正確率=",format(clf4.score(X_train,y_train)*100,".2f"),"%")
print("測試正確率=",format(clf4.score(X_test,y_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf4.get_n_leaves())
print("樹的深度有多少層=",clf4.get_depth())
#繪圖(選3)
from sklearn import tree
dot_data=tree.export_graphviz(clf3,out_file=None,
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
#%%1 決策樹:部分變數X_new2(分割)
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test=train_test_split(
    X_new2,y,test_size=0.2, random_state=20240104)
#%%1 決策樹:部分變數X_new2建模

# gini
clf3=DecisionTreeClassifier(random_state=20240104,criterion="gini",
                           min_samples_leaf=80, min_samples_split=0.3)
clf3.fit(X2_train,y2_train)
print("建模正確率=",format(clf3.score(X2_train,y2_train)*100,".2f"),"%")
print("測試正確率=",format(clf3.score(X2_test,y2_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf3.get_n_leaves())
print("樹的深度有多少層=",clf3.get_depth())

# entropy
clf4=DecisionTreeClassifier(random_state=20240104,criterion="entropy",
                           min_samples_leaf=10, min_samples_split=0.01)
clf4.fit(X2_train,y2_train)
print("建模正確率=",format(clf4.score(X2_train,y2_train)*100,".2f"),"%")
print("測試正確率=",format(clf4.score(X2_test,y2_test)*100,".2f"),"%")
print("樹的葉子有多少個=",clf4.get_n_leaves())
print("樹的深度有多少層=",clf4.get_depth())

#繪圖(選_)
from sklearn import tree
dot_data=tree.export_graphviz(clf3,out_file=None,
                              feature_names=X_new2.columns,
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
graph.render("tree_部分變數",view=False)
#%% 類別編號對應
data1.info()

original_labels=data1["Types_of_waters"]
df_Types_of_waters =  pd.DataFrame([Types_of_waters,original_labels]).T
df_Types_of_waters.value_counts()

original_labels=data1["Drowning_reasons"]
df_Drowning_reasons =  pd.DataFrame([Drowning_reasons,original_labels]).T
df_Drowning_reasons.value_counts()

original_labels=data1["Gender"]
df_Gender =  pd.DataFrame([Gender,original_labels]).T
df_Gender.value_counts()

original_labels=data1["Region"]
df_Region =  pd.DataFrame([Region,original_labels]).T
df_Region.value_counts()

original_labels=data1["Is_Holiday"]
df_Is_Holiday =  pd.DataFrame([Is_Holiday,original_labels]).T
df_Is_Holiday.value_counts()

original_labels=data1["Season"]
df_Season =  pd.DataFrame([Season,original_labels]).T
df_Season.value_counts()
    
original_labels=data1["time_period"]
df_time_period =  pd.DataFrame([time_period,original_labels]).T
df_time_period.value_counts()
#%%2 關聯法則

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 23 20:16:20 2023

# ## basket data analysis
# """
# #df.info()

# import pandas as pd
# df=pd.read_csv("Retail.csv")
# df["Description"]=df["Description"].str.strip()
# df.dropna(axis=0,subset=["InvoiceNo"],inplace=True)
# df["InvoiceNo"]=df["InvoiceNo"].astype("str")
# df=df[~df["InvoiceNo"].str.contains("C")]#不要含C的，C開頭代表退貨

# basket=(
#         df[df["Country"]=="France"]
#        .groupby(["InvoiceNo","Description"])["Quantity"]
#        .sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")
#        )





# def encode_units(x):
#     if x<=0:
#         return 0
#     else:
#         return 1

# basket_sets=basket.applymap(encode_units)
# basket_sets.drop("POSTAGE",inplace=True,axis=1)

# #pip install mlxtend
# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules

# frequent_itemsets=apriori(basket_sets, min_support=0.07, use_colnames=True)
# rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.1)
# rules.to_csv("rules.csv")

#%%3 SVM前處理(使用Is_Holiday,Gender,Age)
data1.info()
#編碼
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Gender=le.fit_transform(data1["Gender"])
subdata= pd.DataFrame([Is_Holiday,Gender,data1["Age"]]).T
subdata.columns=["Is_Holiday","Gender","Age"]

X=subdata
y=pd.Series(le.fit_transform(data1["Drowning_results"]),name="Drowning_results")



#切割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20240104)



#標準化
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train)
X_train_std=ss.transform(X_train)
X_test_std=ss.transform(X_test)
#%%3 SVM

##Linear
from sklearn.svm import LinearSVC
#1
m1=LinearSVC(C=0.1, dual=False, class_weight="balanced")
m1.fit(X_train_std, y_train)
print("Linear1訓練正確率＝", m1.score(X_train_std, y_train))
print("Linear1測試正確率＝", m1.score(X_test_std, y_test))

#2
m2=LinearSVC(C=0.4, dual=False, class_weight="balanced")
m2.fit(X_train_std, y_train)
print("Linear2訓練正確率＝", m2.score(X_train_std, y_train))
print("Linear2測試正確率＝", m2.score(X_test_std, y_test))

#3
m3=LinearSVC(C=1, dual=False, class_weight="balanced")
m3.fit(X_train_std, y_train)
print("Linear3訓練正確率＝", m3.score(X_train_std, y_train))
print("Linear3測試正確率＝", m3.score(X_test_std, y_test))

#4
m4=LinearSVC(C=100, dual=False, class_weight="balanced")
m4.fit(X_train_std, y_train)
print("Linear4訓練正確率＝", m4.score(X_train_std, y_train))
print("Linear4測試正確率＝", m4.score(X_test_std, y_test))



##SVC
from sklearn.svm import SVC
#5
m5=SVC(gamma=0.8, kernel="rbf",probability=True)
m5.fit(X_train_std, y_train)
y_pred=m5.predict(X_train_std)
print("SVC1訓練正確率＝", m5.score(X_train_std, y_train))
print("SVC1測試正確率＝", m5.score(X_test_std, y_test))

#6
m6=SVC(gamma=0.5, kernel="rbf",probability=True)
m6.fit(X_train_std, y_train)
y_pred=m6.predict(X_train_std)
print("SVC2訓練正確率＝", m6.score(X_train_std, y_train))
print("SVC2測試正確率＝", m6.score(X_test_std, y_test))

#7
m7=SVC(gamma=0.3, kernel="rbf",probability=True)
m7.fit(X_train_std, y_train)
y_pred=m7.predict(X_train_std)
print("SVC3訓練正確率＝", m7.score(X_train_std, y_train))
print("SVC3測試正確率＝", m7.score(X_test_std, y_test))

#8
m8=SVC(gamma=0.1, kernel="rbf",probability=True)
m8.fit(X_train_std, y_train)
y_pred=m8.predict(X_train_std)
print("SVC4訓練正確率＝", m8.score(X_train_std, y_train))
print("SVC4測試正確率＝", m8.score(X_test_std, y_test))
#%%4 RF前處理
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#編碼
Types_of_waters=le.fit_transform(data1["Types_of_waters"])
Drowning_reasons=le.fit_transform(data1["Drowning_reasons"])
Gender=le.fit_transform(data1["Gender"])
Age=le.fit_transform(data1["Age"])
Region=le.fit_transform(data1["Region"])
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Season=le.fit_transform(data1["Season"])
time_period=le.fit_transform(data1["time_period"])

X=pd.DataFrame([Types_of_waters,Drowning_reasons,Gender,data1["Age"],
                 Region,Is_Holiday,Season,time_period]).T
X.columns=["Types_of_waters","Drowning_reasons","Gender","Age",
           "Region","Is_Holiday","Season","time_period"]
y=data1["Drowning_results"]

#分割訓練測試
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20240104)
#%%4 RF

from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier(n_estimators=800, max_depth=7, random_state=20240104,)
clf1.fit(X_train,y_train)
print("隨機森林1訓練正確率＝", clf1.score(X_train, y_train))
print("隨機森林1測試正確率＝", clf1.score(X_test, y_test))

clf2=RandomForestClassifier(n_estimators=8, max_depth=7, random_state=20240104)
clf2.fit(X_train,y_train)
print("隨機森林2訓練正確率＝", clf2.score(X_train, y_train))
print("隨機森林2測試正確率＝", clf2.score(X_test, y_test))

clf3=RandomForestClassifier(n_estimators=300, max_depth=4, random_state=20240104)
clf3.fit(X_train,y_train)
print("隨機森林3訓練正確率＝", clf3.score(X_train, y_train))
print("隨機森林3測試正確率＝", clf3.score(X_test, y_test))

clf4=RandomForestClassifier(n_estimators=3, max_depth=4, random_state=20240104)
clf4.fit(X_train,y_train)
print("隨機森林4訓練正確率＝", clf4.score(X_train, y_train))
print("隨機森林4測試正確率＝", clf4.score(X_test, y_test))
#%%5 KNN前處理
import pandas as pd
import numpy as np
import statistics
data1.info()
#編碼
#(Label Encoder) #Is_Holiday為布林值，OneHot會出問題
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Is_Holiday=pd.DataFrame(Is_Holiday, columns=["Is_Holiday"])

#(One Hot Encoder)
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(sparse=False)
Types_of_waters=ohe.fit_transform(data1[["Types_of_waters"]])
Types_of_waters=pd.DataFrame(Types_of_waters)
Types_of_waters.columns=ohe.categories_[0]

Gender=ohe.fit_transform(data1[["Gender"]])
Gender=pd.DataFrame(Gender)
Gender.columns=ohe.categories_[0]

Region=ohe.fit_transform(data1[["Region"]])
Region=pd.DataFrame(Region)
Region.columns=ohe.categories_[0]

Season=ohe.fit_transform(data1[["Season"]])
Season=pd.DataFrame(Season)
Season.columns=ohe.categories_[0]

time_period=ohe.fit_transform(data1[["time_period"]])
time_period=pd.DataFrame(time_period)
time_period.columns=ohe.categories_[0]

Drowning_reasons=ohe.fit_transform(data1[["Drowning_reasons"]])
Drowning_reasons=pd.DataFrame(Drowning_reasons)
Drowning_reasons.columns=ohe.categories_[0]

#標準化
X1=data1["Age"]
X1=pd.DataFrame(X1)
    
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X1=ss.fit_transform(X1)
X1=pd.DataFrame(X1)
X1.columns=["Age"]
Age=X1      

X=pd.concat([Types_of_waters,Drowning_reasons,Gender,Age,Region,
             Is_Holiday,Season,time_period], axis=1)
y=pd.Series(data1["Drowning_results"],name="Drowning_results")

#分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20240104)
#%%5 KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
acc=[]
for i in range(1,993):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    print("k=",i,"的測試正確率＝",accuracy_score(y_test, y_pred))
    acc.append(accuracy_score(y_test, y_pred))
  
print("測試正確率最高的＝",max(acc))

bestK_list=[]
for i in range(1,993):
    if acc[i-1]==max(acc):
        bestK_list.append(i)

print("K=",bestK_list,"的測試正確率最高=",max(acc))        



knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
print("建模正確率:",knn.score(X_train, y_train))

y_pred=knn.predict(X_test)
print("測試正確率:",accuracy_score(y_test, y_pred))
#%%6 Ensembling

data1.info()
#編碼
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Gender=le.fit_transform(data1["Gender"])
subdata= pd.DataFrame([Is_Holiday,Gender,data1["Age"]]).T
subdata.columns=["Is_Holiday","Gender","Age"]

X=subdata
y=pd.Series(le.fit_transform(data1["Drowning_results"]),name="Drowning_results")

#分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2, random_state=20240104)



#RF
from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier(n_estimators=300, max_depth=4, random_state=20240104)
#KNN
from sklearn.neighbors import KNeighborsClassifier
clf2=KNeighborsClassifier(n_neighbors=9)
#SVC
from sklearn.svm import SVC 
clf3=SVC(gamma=0.8, kernel="rbf",probability=True)

#平行集成學習(hard)
from sklearn.ensemble import VotingClassifier
clf41=VotingClassifier(estimators=[("RF",clf1),("KNN",clf2),("SVC",clf3)],
                      voting="hard", n_jobs=-1) #n_jobs常設為-1
#平行集成學習(soft)
from sklearn.ensemble import VotingClassifier
clf42=VotingClassifier(estimators=[("RF",clf1),("KNN",clf2),("SVC",clf3)],
                      voting="soft", n_jobs=-1) #n_jobs常設為-1



clf1.fit(X_train,y_train)
print("隨機森林訓練資料集正確率=",clf1.score(X_train, y_train))
print("隨機森林測試資料集正確率=",clf1.score(X_test, y_test))

from sklearn.preprocessing import StandardScaler
ss=StandardScaler().fit(X_train)
X_train_std=ss.fit_transform(X_train)
X_test_std=ss.fit_transform(X_test)
clf2.fit(X_train_std, y_train)
print("KNN訓練資料集正確率=",clf2.score(X_train_std, y_train))
print("KNN測試資料集正確率=",clf2.score(X_test_std, y_test))

clf3.fit(X_train_std, y_train)
print("SVM訓練資料集正確率=",clf3.score(X_train_std, y_train))
print("SVM測試資料集正確率=",clf3.score(X_test_std, y_test))

clf41.fit(X_train_std, y_train)
print("集成分析法hard訓練資料集正確率=",clf41.score(X_train_std, y_train))
print("集成分析法hard測試資料集正確率=",clf41.score(X_test_std, y_test))
clf42.fit(X_train_std, y_train)
print("集成分析法soft訓練資料集正確率=",clf42.score(X_train_std, y_train))
print("集成分析法soft測試資料集正確率=",clf42.score(X_test_std, y_test))
#%%7 k-means前處理
import pandas as pd
import numpy as np
import statistics

#編碼
#(Label Encoder) #Is_Holiday為布林值，OneHot會出問題
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Is_Holiday=le.fit_transform(data1["Is_Holiday"])
Is_Holiday=pd.DataFrame(Is_Holiday, columns=["Is_Holiday"])

#(One Hot Encoder)
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(sparse=False)
Types_of_waters=ohe.fit_transform(data1[["Types_of_waters"]])
Types_of_waters=pd.DataFrame(Types_of_waters)
Types_of_waters.columns=ohe.categories_[0]

Season=ohe.fit_transform(data1[["Season"]])
Season=pd.DataFrame(Season)
Season.columns=ohe.categories_[0]

Drowning_reasons=ohe.fit_transform(data1[["Drowning_reasons"]])
Drowning_reasons=pd.DataFrame(Drowning_reasons)
Drowning_reasons.columns=ohe.categories_[0]

time_period=ohe.fit_transform(data1[["time_period"]])
time_period=pd.DataFrame(time_period)
time_period.columns=ohe.categories_[0]

Swimming_skills=ohe.fit_transform(data1[["Swimming_skills"]])
Swimming_skills=pd.DataFrame(Swimming_skills)
Swimming_skills.columns=ohe.categories_[0]

Gender=ohe.fit_transform(data1[["Gender"]])
Gender=pd.DataFrame(Gender)
Gender.columns=ohe.categories_[0]

Region=ohe.fit_transform(data1[["Region"]])
Region=pd.DataFrame(Region)
Region.columns=ohe.categories_[0]



#標準化
X1=data1["Age"]
X1=pd.DataFrame(X1)
    
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X1=ss.fit_transform(X1)
X1=pd.DataFrame(X1)
X1.columns=["Age"]
Age=X1      

X=pd.concat([Age,Types_of_waters,Season,Is_Holiday,Drowning_reasons,
             time_period,Swimming_skills,Gender,Region], axis=1)
#%%7 k-means(未加入目標變數)

#Kmeans 陡坡圖
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortion=[]
for i in range(10):
    kmeans=KMeans(n_clusters=i+1, init="k-means++",
                  random_state=20240104, n_init=15, max_iter=200)
    kmeans.fit(X)
    distortion.append(kmeans.inertia_)
    
print(distortion)   
 
plt.plot(range(1,11),distortion, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")



#Kmeans (選擇分4群)
kmeans=KMeans(n_clusters=4, init="k-means++",
              random_state=20240104, n_init=15, max_iter=200)
kmeans.fit(X)
centroid=pd.DataFrame(kmeans.cluster_centers_,columns=X.columns)

X_pred=kmeans.predict(X)
print(pd.crosstab(data1["Drowning_results"], X_pred))
print("用分群來預測分類的正確率＝",(313+177+119+180)/1242) #0.6353
#%%7 k-means(加入目標變數)

Drowning_results=pd.Series(
    le.fit_transform(data1["Drowning_results"]),name="Drowning_results")

Xy=pd.concat([Age,Types_of_waters,Season,Is_Holiday,Drowning_reasons,
             time_period,Swimming_skills,Gender,Region,Drowning_results], axis=1)



#Kmeans 陡坡圖
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortion2=[]
for i in range(10):
    kmeans=KMeans(n_clusters=i+1, init="k-means++",
                  random_state=20240104, n_init=15, max_iter=200)
    kmeans.fit(Xy)
    distortion2.append(kmeans.inertia_)
    
print(distortion2)   
 
plt.plot(range(1,11),distortion2, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")



#Kmeans (選擇分4群)
kmeans2=KMeans(n_clusters=4, init="k-means++",
              random_state=20240104, n_init=15, max_iter=200)
kmeans2.fit(Xy)
centroid2=pd.DataFrame(kmeans2.cluster_centers_,columns=Xy.columns)

X_pred2=kmeans2.predict(Xy)
print(pd.crosstab(data1["Drowning_results"], X_pred2))
print("加目標變數後用分群來預測分類的正確率＝",(335+175+140+182)/1242) #0.6699

#%%7 k-means 比較

#比較加入目標變數分群的正確率變化(分4群)：
print("用分群來預測分類的SSE＝",distortion[3]) #5567.44
print("加目標變數後用分群來預測分類的SSE＝",distortion2[3]) #5832.52

#比較加入目標變數分群SSE變化(分4群)：
print("用分群來預測分類的＝",(313+177+119+180)/1242) #0.6353
print("加目標變數後用分群來預測分類的正確率＝",(335+175+140+182)/1242) #0.6699
#%%










































