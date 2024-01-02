#setwd("")
bank<-read.csv("new_data.csv")
str(bank)
bank$sex<-as.factor(bank$sex)
bank$region<-as.factor(bank$region)
bank$married<-as.factor(bank$married)
bank$children<-as.factor(bank$children)
bank$car<-as.factor(bank$car)
bank$save_act<-as.factor(bank$save_act)
bank$current_act<-as.factor(bank$current_act)
bank$mortgage<-as.factor(bank$mortgage)
bank$pep<-as.factor(bank$pep)
str(bank)

set.seed(20231026)
n<-nrow(bank)
sindex<-sample(n,round(n*0.7))
TrainD<-bank[sindex,]
TestD<-bank[-sindex,]

library(RWeka)
ctree<-J48(pep~.,data=TrainD,control=Weka_control(M=2,C=0.25))
print(ctree)

library(partykit)
rparty.tree<-as.party(ctree)
plot(rparty.tree)

TrainD_predict=predict(ctree,TrainD,type="class")
TrainD$predict=TrainD_predict
cm<-table(TrainD$pep, TrainD$predict, dnn=c("實際","預測"))#Confusion Matrix
cm
TrainD_accuracy<-sum(diag(cm))/sum(cm)
TrainD_accuracy

TestD_predict=predict(ctree,TestD,type="class")
TestD$predict=TestD_predict
cm2<-table(TestD$pep, TestD$predict, dnn=c("實際","預測"))
cm2
TestD_accuracy<-sum(diag(cm2))/sum(cm2)
TestD_accuracy

