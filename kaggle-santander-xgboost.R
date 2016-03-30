rm(list=ls())
# (03/14/2016) AUC = 0.840485 
# (03/30/2016) AUC = 0.840668
library(xgboost)
library(Matrix)
library(caret)
library(Ckmeans.1d.dp) # only necessary for variable importance plot
library(smbinning)

# 'Murica
set.seed(1776)

train = read.csv(file.choose())
test = read.csv(file.choose())

### Remove ID
train$ID = NULL
test.id = test$ID
test$ID = NULL

### Extract TARGET
train.y = train$TARGET
train$TARGET = NULL

count0 = function(x) {
  return( sum(x == 0) )
}
train$n0 = apply(train, 1, FUN = count0)
test$n0 = apply(test, 1, FUN = count0)

### Removing constant features
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] = NULL
    test[[f]] = NULL
  }
}

##### Removing identical features
features_pair = combn(names(train), 2, simplify = F)
toRemove = c()
for(pair in features_pair) {
  f1 = pair[1]
  f2 = pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove = c(toRemove, f2)
    }
  }
}

feature.names = setdiff(names(train), toRemove)

train = train[, feature.names]
test = test[, feature.names]

train$TARGET = train.y

### Bin features
# var15
# bin1.train = smbinning(train, "TARGET", "var15")
# smbinning.plot(bin1.train)
train$var15.b[train$var15 <= 24] = "1"
train$var15.b[train$var15 > 24 & train$var15 <= 26] = "2"
train$var15.b[train$var15 > 26 & train$var15 <= 28] = "3"
train$var15.b[train$var15 > 28 & train$var15 <= 106] = "4"
train$var15.b = as.factor(train$var15.b)
test$var15.b[test$var15 <= 24] = "1"
test$var15.b[test$var15 > 24 & test$var15 <= 26] = "2"
test$var15.b[test$var15 > 26 & test$var15 <= 28] = "3"
test$var15.b[test$var15 > 28] = "4"
test$var15.b = as.factor(test$var15.b)

# saldo_var30
# bin2.train = smbinning(train, "TARGET", "saldo_var30")
# smbinning.plot(bin2.train)
train$saldo_var30.b[train$saldo_var30 <= 2.94] = "1"
train$saldo_var30.b[train$saldo_var30 > 2.94 & train$saldo_var30 <= 8684.34] = "2"
train$saldo_var30.b[train$saldo_var30 > 8684.34 & train$saldo_var30 <= 88245] = "3"
train$saldo_var30.b[train$saldo_var30 > 88245] = "4"
train$saldo_var30.b = as.factor(train$saldo_var30.b)
test$saldo_var30.b[test$saldo_var30 <= 2.94] = "1"
test$saldo_var30.b[test$saldo_var30 > 2.94 & test$saldo_var30 <= 8684.34] = "2"
test$saldo_var30.b[test$saldo_var30 > 8684.34 & test$saldo_var30 <= 88245] = "3"
test$saldo_var30.b[test$saldo_var30 > 88245] = "4"
test$saldo_var30.b = as.factor(test$saldo_var30.b)

# var38
# bin3.train = smbinning(train, "TARGET", "var38")

# saldo_medio_var5_hace3
# bin4.train = smbinning(train, "TARGET", "saldo_medio_var5_hace3")

# saldo_medio_var5_hace2
# bin5.train = smbinning(train, "TARGET", "saldo_medio_var5_hace2")

train = sparse.model.matrix(TARGET ~ ., data = train)

dtrain = xgb.DMatrix(data=train, label=train.y)
watchlist = list(train=dtrain)

param = list(   objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 6,
                subsample           = 0.8,
                colsample_bytree    = 0.7
)

clf = xgb.train(    params              = param, 
                    data                = dtrain, 
                    nrounds             = 560, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

train.names = dimnames(train)[[2]]
importance_matrix = xgb.importance(train.names, model = clf)
xgb.plot.importance(importance_matrix[1:20,])

test$TARGET = -1
test = sparse.model.matrix(TARGET ~ ., data = test)
preds = predict(clf, test)
submission = data.frame(ID = test.id, TARGET = preds)
write.csv(submission, file = choose.files(caption="Save As...", filters = c("Comma Delimited Files (.csv)","*.csv")), row.names = F)
