#setwd("")
dat<-read.csv("new_data.csv")
str(dat)
dat$Types_of_waters<-as.factor(dat$Types_of_waters)
dat$Drowning_reasons<-as.factor(dat$Drowning_reasons)
dat$Gender<-as.factor(dat$Gender)
#dat$Age<-as.factor(dat$Age)
dat$Region<-as.factor(dat$Region)
dat$Is_Holiday<-as.factor(dat$Is_Holiday)
dat$Season<-as.factor(dat$Season)
dat$time_period<-as.factor(dat$time_period)

dat$Drowning_results[dat$Drowning_results=="死亡"] = "death"
dat$Drowning_results[dat$Drowning_results=="獲救"] = "survived"
dat$Drowning_results<-as.factor(dat$Drowning_results)
str(dat)

set.seed(20240104)
n<-nrow(dat)
sindex<-sample(n,round(n*0.8))
TrainD<-dat[sindex,]
TestD<-dat[-sindex,]


#全變數：
library(RWeka)
ctree<-J48(Drowning_results~.,data=TrainD,control=Weka_control(M=2,C=0.1))
print(ctree)
#library(partykit)
#rparty.tree<-as.party(ctree)
#plot(rparty.tree)

TrainD_predict=predict(ctree,TrainD,type="class")
TrainD$predict=TrainD_predict
cm1<-table(TrainD$Drowning_results, TrainD$predict, dnn=c("實際","預測"))#Confusion Matrix
cm1
TrainD_accuracy<-sum(diag(cm1))/sum(cm1)
TrainD_accuracy

TestD_predict=predict(ctree,TestD,type="class")
TestD$predict=TestD_predict
cm2<-table(TestD$Drowning_results, TestD$predict, dnn=c("實際","預測"))
cm2
TestD_accuracy<-sum(diag(cm2))/sum(cm2)
TestD_accuracy









#部分變數：

TrainD=TrainD[,c("Types_of_waters","Drowning_reasons","Age","Drowning_results")]
TestD=TestD[,c("Types_of_waters","Drowning_reasons","Age","Drowning_results")]

library(RWeka)
ctree<-J48(Drowning_results~.,data=TrainD,control=Weka_control(M=3,C=0.3))
print(ctree)
#library(partykit)
#rparty.tree<-as.party(ctree)
#plot(rparty.tree)

TrainD_predict=predict(ctree,TrainD,type="class")
TrainD$predict=TrainD_predict
cm3<-table(TrainD$Drowning_results, TrainD$predict, dnn=c("實際","預測"))#Confusion Matrix
cm3
TrainD_accuracy<-sum(diag(cm3))/sum(cm3)
TrainD_accuracy

TestD_predict=predict(ctree,TestD,type="class")
TestD$predict=TestD_predict
cm4<-table(TestD$Drowning_results, TestD$predict, dnn=c("實際","預測"))
cm4
TestD_accuracy<-sum(diag(cm4))/sum(cm4)
TestD_accuracy



