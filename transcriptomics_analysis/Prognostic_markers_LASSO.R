



########coxLasso##################################
library(foreign)       #调用foreign包

rawdata<-read.csv("LASSO.csv")        #读取R work中的

View(rawdata)          #查看数据

str(rawdata)           #查看数据的类型，非常重要

summary(rawdata)       #数据进行简单描述

rawdata<-na.omit(rawdata)

?glmnet

#install.packages("glmnet")
library(glmnet)
library(survival)

rawdata<-rawdata[rawdata$time != 0,]   #切记，time不可以出现等于0的case，否则后续不能运行

View(rawdata)
x<-as.matrix(rawdata[,c(3:67)])

y<-data.matrix(Surv(rawdata$time_to_event_28days,rawdata$mortality_event_28days))
y

fit<-glmnet(x,y,family = "cox")

#X轴为lnlambda
plot(fit,xvar="lambda",label=F)

#X轴为L1 norm
plot(fit,xvar="norm",label=TRUE)

#X轴为lnlambda
plot(fit,xvar="dev",label=TRUE)


print(fit)

lasso.coef<-coef(fit,s=0.00373)

lasso.coef       #在最小lambda时，lasso回归个变量系数



##################################################################
####cvlasso
cv.fit<-cv.glmnet(x,y,family="cox")
plot(cv.fit)
abline(v=log(c(cv.fit$lambda.min,cv.fit$lambda.1se)),lty=2,lwd=0.5)



#如果取最小值时
cv.fit$lambda.min
Coefficients <- coef(fit, s = cv.fit$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
Active.Index
Active.Coefficients
row.names(Coefficients)[Active.Index]


#如果取1倍标准误
cv.fit$lambda.1se
Coefficients <- coef(fit, s = cv.fit$lambda.1se)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
Active.Index
Active.Coefficients
row.names(Coefficients)[Active.Index]

###################################################



