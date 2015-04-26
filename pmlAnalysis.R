##Practical machine learning course project
##NOTE - this script was written on a PC running Windows 7, with R version 3.1.1

##Load training and testing data sets 
##ASSUMES THE DATA FILES ARE IN A FOLDER TITLED "data" IN YOUR WORKING DIRECTORY
training<-read.csv(file.choose())
testing<-read.csv(file.choose())
names(training)
names(testing)

##Examine and clean data set
##Examine data within the 20 testing cases to determine which variables will be available to use in prediction
testing
##Several variables in testing set have all NAs - there is no reason to use these in predictive model
##Remove all 'summary' variables from the training and validation sets (avg, stddev,var,max,min,amplitude,skewness,kurtosis) - these are all NAs in testing set
library(dplyr)
training<-select(training,-contains("avg"))
training<-select(training,-contains("stddev"))
training<-select(training,-contains("var"))
training<-select(training,-contains("max"))
training<-select(training,-contains("min"))
training<-select(training,-contains("amplitude"))
training<-select(training,-contains("skewness"))
training<-select(training,-contains("kurtosis"))
##For building predictive model, remove variables 1-7 - User IDs, timestamps,and window IDs are not going to help identify activity based on sensors in test set
training<-select(training,8:60)

##Beacuse of large size of training set, split into a training set and a validation set
library(caret)
inTrain<-createDataPartition(y=training$classe,p=0.7,list=FALSE)
train<-training[inTrain,]
valid<-training[-inTrain,]

##Preprocessing
##Training data set has 53 variables remaining, prediction algorithms may be more efficient (scalable) if this number can be reduced
##Check for any zero covariates within training set
nzv<-nearZeroVar(train,saveMetrics=TRUE)
nzv

##Search for any highly correlated covariates in training set
correlations<-abs(cor(train[,-53]))
diag(correlations)<-0
corVars<-which(correlations>0.8,arr.ind=T)
length(corVars)
names(train)
##Several covariates representing movement variables are highly correlated - will preprocess with PCA to handle these
##First need to standardize covariates (center and scale), such that their different scales do not inappropriately influence PCA.
##Retain components that explain 99% of variance
preProc<-preProcess(train[,-53],method=c("center","scale","pca"),thresh=0.99)
##36 PC axes are retained
trainProc<-predict(preProc,train[,-53])
trainProc<-mutate(trainProc,classe=train$classe)
validProc<-predict(preProc,valid[,-53])
validProc<-mutate(validProc,classe=valid$classe)

##Build first predictive model using classification tree
treeFit<-train(classe~.,method="rpart",data=trainProc)
print(treeFit$finalModel)
library(rattle)
fancyRpartPlot(treeFit$finalModel)
confusionMatrix(predict(treeFit,trainProc[,-37]),trainProc$classe)$overall

##Build a second predictive model using random forests
set.seed(3733)
rfFit<-randomForest(classe~.,data=trainProc,importance=TRUE)
rfFit$confusion
##Predict classes for validation set using this model
rfPreds<-predict(rfFit,newdata=validProc[,-37])
##Examine how well the model performed
confusionMatrix(rfPreds,validProc$classe)
##The model seems to perform extremely well on the validation set (97.71% accuracy, with a 95% CI ranging from 97.29-98.07%)

##Try an alternative with boosting
boostFit<-train(classe~.,data=trainProc,method="gbm",verbose=FALSE)
print(boostFit)
boostPreds<-predict(boostFit,newdata=validProc[,-37])
confusionMatrix(boostPreds,validProc$classe)
##Random forests model performed better

##Apply random forests model to the test set
##Pre-process test set
testing<-select(testing,-contains("avg"))
testing<-select(testing,-contains("stddev"))
testing<-select(testing,-contains("var"))
testing<-select(testing,-contains("max"))
testing<-select(testing,-contains("min"))
testing<-select(testing,-contains("amplitude"))
testing<-select(testing,-contains("skewness"))
testing<-select(testing,-contains("kurtosis"))
testing<-select(testing,8:60)
testProc<-predict(preProc,testing[,-53])
testProc<-mutate(testProc,problem_id=testing$problem_id)
predictions<-predict(rfFit,newdata=testProc[,-37])
predictions

##Write files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predictions)
