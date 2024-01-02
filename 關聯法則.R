
bank <- read.csv("bank-full.csv",sep = ";")
str(bank)
bank <- bank[,-c(10,11,13,14)] # 刪除column 10 11 13 14  逗號之前是row 逗號之後是column
#請逐行執行
library(infotheo)
bank$age <- discretize(bank$age, "equalwidth", 3) # 離散化 discretize()
bank$age <- as.factor(bank$age$X) # 把上一行不知何故生成的X給去掉
bank$duration <- discretize(bank$duration, "equalfreq", 3)
bank$duration <- as.factor(bank$duration$X)
bank$balance <- discretize(bank$balance, "equalfreq", 3)
bank$balance <- as.factor(bank$balance$X)

bank$previous <- as.factor(bank$previous)
bank$job <- as.factor(bank$job)
bank$marital <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$default <- as.factor(bank$default)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$contact <- as.factor(bank$contact)
bank$poutcome <- as.factor(bank$poutcome)
bank$y <- as.factor(bank$y)

# R 4.0以上的板本 在跑 arules時需要把每個columns到轉成factor
require(arules)
rule <- apriori(bank, parameter = list(supp=0.1, conf=0.7),
                appearance = list(rhs=c("y=yes","y=no")))
sort.rule <- sort(rule, by="support")
subset.matrix <- as.matrix(is.subset(x=sort.rule, y= sort.rule))
subset.matrix[lower.tri(subset.matrix, diag = T)] <- NA
redundant <- colSums(subset.matrix, na.rm = T)>=1
sort.rule <- sort.rule[!redundant]
sort.rule <- as(sort.rule,"data.frame")
write.csv(sort.rule, "bank-rules.csv")

bankNO=subset(bank,y=="no")
bankYES=subset(bank,y=="yes")
n <- nrow(bankNO)
set.seed(20231130) 
sindex <- sample(n,5289)
bankNO = bankNO[sindex,]
newbank = rbind(bankNO,bankYES)

require (arules)
rule<-apriori (newbank, parameter=list(supp=0.1, conf=0.7), 
               appearance=list(rhs=c("y=yes","y=no")))
sort.rule<-sort(rule, by="support")
subset.matrix<- as.matrix (is.subset(x=sort.rule, y=sort.rule)) 
subset.matrix[lower.tri (subset.matrix, diag=T)]<-NA
redundant<-colSums(subset.matrix, na.rm=T)>=1
sort.rule<-sort.rule [!redundant]
sort.rule<-as(sort.rule, "data.frame")
write.csv(sort.rule, "bank-rules2.csv")

