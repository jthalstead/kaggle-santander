rm(list=ls())
# best AUC = 0.840485 (03/14/2016)
library(xgboost)
library(Matrix)
library(caret)
library(Ckmeans.1d.dp)

# 'Murica
set.seed(1776)

train = read.csv("C:\\Users\\Varanus\\Documents\\MEGA\\Kaggle\\Santander\\train.csv")
test = read.csv("C:\\Users\\Varanus\\Documents\\MEGA\\Kaggle\\Santander\\test.csv")

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

### Ensemble ROS + RUS


train = sparse.model.matrix(TARGET ~ ., data = train)

dtrain = xgb.DMatrix(data=train, label=train.y)
watchlist = list(train=dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 6,
                subsample           = 0.9,
                colsample_bytree    = 0.85
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = TRUE
)

train.names = dimnames(dtrain)[[2]]
importance_matrix = xgb.importance(train.names, model = clf)
xgb.plot.importance(importance_matrix[1:20,])

test$TARGET = -1
test = sparse.model.matrix(TARGET ~ ., data = test)
preds = predict(clf, test)
submission = data.frame(ID=test.id, TARGET=preds)
write.csv(submission, "C:\\Users\\Varanus\\Documents\\MEGA\\Kaggle\\Santander\\submission_v7.csv", row.names = F)