#forming the data
csv_data<- read.csv(file.choose(),header = T)
attach(csv_data)
header_value <- colnames(csv_data)
#names(csv_data)

#Fetching the class variable
class_variable <- csv_data$Class

#shuffling the data 

set.seed(2)
g <- runif(nrow(csv_data))
temp_data<- csv_data[order(g),]
colnames(temp_data) <- c(header_value)
temp_data <- data.frame(temp_data,class_variable)
temp_data <- temp_data[-1] #removing the class variable because it is the prediction column
#names()

#predict using class for gini index:

library(rpart)  
library(rpart.plot)
library(cvTools) 
library(caret)
library(tree)

#splitting the tree based on gini error
k_fold <- 10 # setting the value for 10 fold validation 

folds <- cvFolds(NROW(temp_data), K=k_fold)
temp_data$holdoutpred <- rep(0,nrow(temp_data))

for(i in 1:k){
  r_train_data <- temp_data[folds$subsets[folds$which != i], ] #Set the training set
  r_validation_data <- temp_data[folds$subsets[folds$which == i], ] #Set the validation set
  
  #tree model
  training_model_gini = rpart(class_variable~.,data = r_train_data,
                                parms = list(split = "gini"), method = "class")
  rpart.plot(training_model_gini,type = 3,extra = 101, fallen.leaves = T)
  
  #prediction
  prediction_model_label <- predict(training_model_gini, newdata=r_validation_data, type="class")
  prediction_model_prob <- predict(training_model_gini, newdata=r_validation_data)
  temp_data[folds$subsets[folds$which == i], ]$holdoutpred <- prediction_model_label
}

train_control <- trainControl(method="cv", number=10)
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))
model <- train(class_variable~., data=r_train_data, trControl=train_control, method="nb", tuneGrid=grid)
# summarize results
print(model)

#splitting the tree based on information gain
k_fold <- 10 # setting the value for 10 fold validation 

folds <- cvFolds(NROW(temp_data), K=k_fold)
temp_data$holdoutpred <- rep(0,nrow(temp_data))

for(i in 1:k){
  r_train_data <- temp_data[folds$subsets[folds$which != i], ] #Set the training set
  r_validation_data <- temp_data[folds$subsets[folds$which == i], ] #Set the validation set
  
  #tree model
  training_model_gini = rpart(class_variable~.,data = r_train_data,
                              parms = list(split = "information"), method = "class")
  rpart.plot(training_model_gini,type = 3,extra = 101, fallen.leaves = T)
  
  #prediction
  prediction_model_label <- predict(training_model_gini, newdata=r_validation_data, type="class")
  prediction_model_prob <- predict(training_model_gini, newdata=r_validation_data)
  temp_data[folds$subsets[folds$which == i], ]$holdoutpred <- prediction_model_label
}



