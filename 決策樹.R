#setwd("")
dat<-read.csv("new_data.csv")
str(dat)
dat$Types_of_waters<-as.factor(dat$Types_of_waters)
dat$Types_of_waters<-as.factor(dat$Types_of_waters)
dat$Season<-as.factor(dat$Season)
dat$Is_Holiday<-as.factor(dat$Is_Holiday)
dat$Drowning_reasons<-as.factor(dat$Drowning_reasons)
dat$time_period<-as.factor(dat$time_period)
dat$Swimming_skills<-as.factor(dat$Swimming_skills)
dat$Gender<-as.factor(dat$Gender)
dat$Region<-as.factor(dat$Region)

dat$Drowning_results[dat$Drowning_results=="死亡"] = "death"
dat$Drowning_results[dat$Drowning_results=="獲救"] = "survived"
dat$Drowning_results<-as.factor(dat$Drowning_results)
str(dat)

set.seed(20240104)
n<-nrow(dat)
sindex<-sample(n,round(n*0.7))
TrainD<-dat[sindex,]
TestD<-dat[-sindex,]

library(RWeka)
ctree<-J48(Drowning_results~.,data=TrainD,control=Weka_control(M=15,C=0.2))
print(ctree)

library(partykit)
rparty.tree<-as.party(ctree)
plot(rparty.tree)

TrainD_predict=predict(ctree,TrainD,type="class")
TrainD$predict=TrainD_predict
cm<-table(TrainD$Drowning_results, TrainD$predict, dnn=c("實際","預測"))#Confusion Matrix
cm
TrainD_accuracy<-sum(diag(cm))/sum(cm)
TrainD_accuracy

TestD_predict=predict(ctree,TestD,type="class")
TestD$predict=TestD_predict
cm2<-table(TestD$Drowning_results, TestD$predict, dnn=c("實際","預測"))
cm2
TestD_accuracy<-sum(diag(cm2))/sum(cm2)
TestD_accuracy

