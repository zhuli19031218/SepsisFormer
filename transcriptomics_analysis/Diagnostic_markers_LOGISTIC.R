
getwd()
setwd("D:/R work")

##一、数据准备
#1.数据读取（读进来）
#install.packages("readr")
library(readr)
mydata <- read.csv("LOGISTIC.csv")

#删除有缺失值的行
mydata<-na.omit(mydata)
View(mydata)

names(mydata)


#必须做，了解变量类型非常重要
str(mydata)


#attach(mydata)


head(mydata)
#head(mydata,6)
--------------------------------


#2.数据准备
#2.1数值变量准备
#2.1.1数据摘要
  
summary(mydata)
summary(mydata)

#2.2单因素Logistic回归（批量执行）

uni_glm_model<-function(x){
  FML<-as.formula(paste0("outcome==1~",x))  #outcome==1，您以后就改绿色的变量, ZXglm1<-glm(FML,data = mydate,family = binomial)glm1<-glm(FML,data=mydata,family=binomialglm1<-glm(FML,data=mydata,family=binomial)
  glm1<-glm(FML,data =mydata,family = binomial)
  glm2<-summary(glm1)
  
  #计算我们所要的指标
  OR<-round(exp(coef(glm1)),2)
  SE<-round(glm2$coefficients[,2],3)  
  CI2.5<-round(exp(coef(glm1)-1.96*SE),2)
  CI97.5<-round(exp(coef(glm1)+1.96*SE),2)
  CI<-paste0(CI2.5,'-',CI97.5)
  B<-round(glm2$coefficients[,1],3)
  Z<-round(glm2$coefficients[,3],3)
  P<-round(glm2$coefficients[,4],3)
  
  #将计算出来的指标制作为数据框
  uni_glm_model<-data.frame('characteristics'=x,
                            'B'=B,
                            'SE'=SE,
                            'OR'=OR,
                            'CI'=CI,
                            'Z' =Z,
                            'P'=P)[-1,]
  
  return(uni_glm_model)
}

#指定参与分析的若干自变量X
variable.names<-colnames(mydata)[c(2:8)]  #要核实这里的X对应的列是否对？若分开的可以这样[c,(3:18,20:40)]
variable.names

#运行上面自定义批量执行函数
uni_glm<-lapply(variable.names,uni_glm_model)
uni_glm


#install.packages("plyr")
library(plyr)

#生成单变量分析的综合结果
uni_glm<-ldply(uni_glm,data.frame)

#看下结果是啥样子的
uni_glm

View(uni_glm)


#将单因素分析的结果，写到csv中.
write.csv(uni_glm, "uni.csv")


#将P<0.05的结果挑选出来（如需）
uni_glm1 <- uni_glm[uni_glm$P<= 0.05,]
uni_glm1

#将P<0.1的结果挑选出来（如需）
uni_glm2 <- uni_glm[uni_glm$P<= 0.1,]
uni_glm2

#直接将P<0.05的变量的characteristics提取出来
uni_glm$characteristics[uni_glm$P<= 0.05]


#将P<0.05的结果，写到csv中（如需).
write.csv(uni_glm1, "p5.csv")
write.csv(uni_glm2, "p10.csv")


##1.1多因素enter回归
fml<-as.formula(outcome== 1 ~ CD59 + SERPINB2 + CFD + P2RX1)
fml

modelA<-glm(fml,data = mydata,family=binomial)

modelA             #只能拿到模型的系数

summary(modelA)    #可以拿到模型概要

#看模型的系数及95%CI
cbind(coef= coef(modelA),confint(modelA))
#看模型的OR及95%CI
exp(cbind(OR= coef(modelA),confint(modelA)))


