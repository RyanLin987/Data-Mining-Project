
dat <- read.csv("new_data.csv")
str(dat)
#請逐行執行
library(infotheo)
dat$Age <- discretize(dat$Age, "equalwidth", 3) # 離散化 discretize()
dat$Age <- as.factor(dat$Age$X) # 把上一行不知何故生成的X給去掉

dat$Types_of_waters <- as.factor(dat$Types_of_waters)
dat$Drowning_reasons <- as.factor(dat$Drowning_reasons)
dat$Gender <- as.factor(dat$Gender)
dat$Region <- as.factor(dat$Region)
dat$Is_Holiday <- as.factor(dat$Is_Holiday)
dat$Season <- as.factor(dat$Season)
dat$time_period <- as.factor(dat$time_period)

dat$Drowning_results <- as.factor(dat$Drowning_results)

# R 4.0以上的板本 在跑 arules時需要把每個columns到轉成factor
require(arules)
rule <- apriori(dat, parameter = list(supp=0.05, conf=0.65),
                appearance = list(rhs=c("Drowning_results=死亡","Drowning_results=獲救")))
sort.rule <- sort(rule, by="support")
subset.matrix <- as.matrix(is.subset(x=sort.rule, y= sort.rule))
subset.matrix[lower.tri(subset.matrix, diag = T)] <- NA
redundant <- colSums(subset.matrix, na.rm = T)>=1
sort.rule <- sort.rule[!redundant]
sort.rule <- as(sort.rule,"data.frame")
write.csv(sort.rule, "Survived-rules.csv")







