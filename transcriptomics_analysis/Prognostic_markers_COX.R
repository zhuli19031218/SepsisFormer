

#1.数据读取

#1.1读取
setwd("D:/R work")     #设置工作路径

library(foreign)       #调用foreign包

rawdata<-read.csv("COX.csv")        #读取R work中的

View(rawdata)          #查看数据

str(rawdata)           #查看数据的类型，非常重要

summary(rawdata)       #数据进行简单描述
summary(rawdata) 

rawdata<-na.omit(rawdata)   #删除缺失数据


#1.2数据解读
names(rawdata)                #查看rawdata的变量列表

#3.预后模型构建
#3.0软件包准备

#install.packages("survival")      #安装生存分析包
library(survival)                  #加载生存分析包


#3.3批量单因素分析


uni_cox_model<-function(x){
  FML<-as.formula(paste0("Surv(time,dead==1)~",x))  #dead==1，指的是1为目标结局事件
  cox1<-coxph(FML,data = rawdata)
  cox2<-summary(cox1)
 
  #计算我们所要的指标
  HR<-round(cox2$conf.int[,2],2)
  CI<-paste0(round(cox2$conf.int[,3:4],2),collapse = '-')
  P<-round(cox2$coefficients[,5],3)
  
  #将计算出来的指标制作为数据框
  uni_cox_model<-data.frame('characteristics'=x,
                            'HR'=HR,
                            'CI'=CI,
                            'P'=P)
   return(uni_cox_model)
  }

#指定参与分析的若干自变量X
variable.names<-colnames(rawdata)[c(3:14)]  #要核实这里的X对应的列是否对？若分开的可以这样[c(3:18,20:40)]
variable.names

#运行上面自定义批量执行函数
uni_cox<-lapply(variable.names,uni_cox_model)
uni_cox


#install.packages("plyr")
library(plyr)

#生成单变量分析的综合结果
uni_cox<-ldply(uni_cox,data.frame)

#看下结果是啥样子的
uni_cox

View(uni_cox)


write.csv(uni_cox, "cox单因素.csv")

#多因素分析

fit<-coxph(Surv(time,dead==1)~STAT5B + MTHFR + HPSE + AAK1 + MX1,data = rawdata)
fit

summary(fit)


