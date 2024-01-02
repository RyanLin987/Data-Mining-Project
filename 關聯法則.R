
bank <- read.csv("new_data.csv")
str(bank)
#bank <- bank[,-c(10,11,13,14)] # 刪除column 10 11 13 14  逗號之前是row 逗號之後是column
#請逐行執行
library(infotheo)
bank$Age <- discretize(bank$Age, "equalwidth", 3) # 離散化 discretize()
bank$Age <- as.factor(bank$Age$X) # 把上一行不知何故生成的X給去掉

bank$Types_of_waters <- as.factor(bank$Types_of_waters)
bank$Season <- as.factor(bank$Season)
bank$Is_Holiday <- as.factor(bank$Is_Holiday)
bank$Drowning_reasons <- as.factor(bank$Drowning_reasons)
bank$time_period <- as.factor(bank$time_period)
bank$Swimming_skills <- as.factor(bank$Swimming_skills)
bank$Gender <- as.factor(bank$Gender)
bank$Region <- as.factor(bank$Region)

bank$Drowning_results <- as.factor(bank$Drowning_results)

# R 4.0以上的板本 在跑 arules時需要把每個columns到轉成factor
require(arules)
rule <- apriori(bank, parameter = list(supp=0.05, conf=0.7),
                appearance = list(rhs=c("Drowning_results=死亡","Drowning_results=獲救")))
sort.rule <- sort(rule, by="support")
subset.matrix <- as.matrix(is.subset(x=sort.rule, y= sort.rule))
subset.matrix[lower.tri(subset.matrix, diag = T)] <- NA
redundant <- colSums(subset.matrix, na.rm = T)>=1
sort.rule <- sort.rule[!redundant]
sort.rule <- as(sort.rule,"data.frame")
write.csv(sort.rule, "Survived-rules.csv")




dat_death=subset(bank,Drowning_results=="死亡") #1160
dat_survived=subset(bank,Drowning_results=="獲救") #648
n <- nrow(dat_survived)
set.seed(20240104) 
sindex <- sample(n,1160)
dat_death = dat_death[sindex,]
newbank = rbind(dat_death,dat_survived)



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

