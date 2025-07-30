library(tidyverse)
library(glmnet) 
library(readxl)
data <- read_excel("data.xlsx")


set.seed(2022) 
index <- sample(x = 1:nrow(data), size = 0.6*nrow(data))
train <- data[index,]
test <- data[-index,]

# 可视化方法确认lambda值
fit_lasso <- glmnet(x = as.matrix(train[,-1]),
                    y = as.matrix(train[,1]),
                    family = 'binomial',
                    alpha = 1)
plot(fit_lasso, xvar = "lambda")

# 交叉验证确定lambda值
fit_lasso_cv <- cv.glmnet(x = as.matrix(train[,-1]),
                          y = as.matrix(train[,1]),
                          family = 'binomial',
                          #lambda = seq(0.001, 0.05, 0.0001),
                          alpha = 1)

# 以图形的形式展示交叉验证的结果
plot(fit_lasso_cv)

# 显示交叉验证后的系数
coef(fit_lasso_cv)



# 训练集验证
pred_lasso_train <- predict(fit_lasso_cv,
                            newx = as.matrix(train[,-1]))

# 验证集验证
pred_lasso_test <- predict(fit_lasso_cv,
                           newx = as.matrix(test[,-1]))


# 绘制ROC
library(pROC)
roc_lasso_train <- roc(train$Y, as.numeric(pred_lasso_train))
roc_lasso_test <- roc(test$Y,as.numeric(pred_lasso_test))
auc(roc_lasso_train)
auc(roc_lasso_test)
plot.roc(roc_lasso_train, col = "red", legacy.axes = T)
plot.roc(roc_lasso_test, col = "blue", legacy.axes = T)

