## Alex Rivera Cruz
## Pima Indians Diabetes Prediction Project 
## HarvardX: PH125.9x - Capstone Project

#################################################
# Pima Indians Diabetes Code 
################################################

## Dataset
#Packages required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(tibble)
library(readr)
library(plyr)
library(ggrepel)
library(gridExtra)
library(readr)
library(corrplot)
library(caret)
library(xgboost)
library(e1071)
library(randomForest)

#Download the dataset from an csv file on computer (if you wnat the link : 
#https://www.kaggle.com/uciml/pima-indians-diabetes-database)
file.exists("E:\\Courses\\Data Science\\Lessons\\9. Capstone Project All Learners\\9 Capstone Project All Learners\\Pima Project\\Pima Diabetes\\diabetes.csv")
pima<- read_csv("E:\\Courses\\Data Science\\Lessons\\9. Capstone Project All Learners\\9 Capstone Project All Learners\\Pima Project\\Pima Diabetes\\diabetes.csv")

## Data Analysis
#Check the first 6 rows of the provided dataset
head(pima)%>%
  print.data.frame()

#See the overall structure of the dataset
str(pima)

#Find the proportion of the dataset that is NA
sum(is.na(pima))
sum(is.na(pima))/(ncol(pima)*nrow(pima))

## Visualisation
#Convert outcome to binary factor class#
pima$Outcome <- as.factor(pima$Outcome)

#Box and whisker plot for pregnancies stratified by Diabates status
p1<- pima %>% group_by(Outcome) %>% ggplot(aes(y=Pregnancies,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Pregnancies", x= "Diabetes Status", y = "Pregnancies") + theme(legend.position = "none")

#Box and whisker plot for blood glucose stratified by Diabates status
p2<- pima %>% group_by(Outcome) %>% ggplot(aes(y=Glucose,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Blood Glucose", x= "Diabetes Status", y = "Blood Glucose")  + theme(legend.position = "none")

#Box and whisker plot for BP stratified by Diabates status
p3<- pima %>% group_by(Outcome) %>% ggplot(aes(y=BloodPressure,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Blood Pressure", x= "Diabetes Status", y = "Blood Pressure")  + theme(legend.position = "none")

#Box and whisker plot for skin thickness stratified by Diabates status
p4<- pima %>% group_by(Outcome) %>% ggplot(aes(y=SkinThickness,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Skin Thickness", x= "Diabetes Status", y = "Skin Thickness")  + theme(legend.position = "none")

#arrange the plots in a 2x2
grid.arrange(p1,p2,p3,p4,ncol=2)

#Box and whisker plot for insulin levels stratified by Diabates status
p5<- pima %>% group_by(Outcome) %>% ggplot(aes(y=Insulin,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Insulin Levels", x= "Diabetes Status", y = "Insulin Levels")  + theme(legend.position = "none")

#Box and whisker plot for BMI stratified by Diabates status
p6<- pima %>% group_by(Outcome) %>% ggplot(aes(y=BMI,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="BMI", x= "Diabetes Status", y = "BMI")  + theme(legend.position = "none")

#Box and whisker plot for pedigree levels stratified by Diabates status
p7<- pima %>% group_by(Outcome) %>% ggplot(aes(y=DiabetesPedigreeFunction,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Diabetes Pedigree Levels", x= "Diabetes Status", y = "Pedigree Levels")  + theme(legend.position = "none")

#Box and whisker plot for ages stratified by Diabates status
p8<- pima %>% group_by(Outcome) %>% ggplot(aes(y=Age,x=Outcome)) + geom_boxplot(aes(fill=Outcome)) + 
  labs(title="Age", x= "Diabetes Status", y = "Age")  + theme(legend.position = "none")

#arrange the plots in a 2x2#
grid.arrange(p5,p6,p7,p8,ncol=2)

#scatterplot for Skin Thickness on BMI across Diabetes status
pima %>% group_by(Outcome) %>% ggplot() + geom_point(aes(x=BMI, y=SkinThickness, colour=Outcome), alpha=0.5) 
          + theme(legend.position = "none") + ggtitle("Cutaneous Thickness on BMI Across Diabetes Patients")

#correlation plot between all variables in `pima`
par( mfrow = c(1,1) )
pima$Outcome<- as.numeric(pima$Outcome)
cor1 <- cor(pima, method = c("pearson"))
cor2 <- round(cor(cor1),2)
corrplot::corrplot(cor2, method = "circle")

##Modeling Approach

#Attempt to convert the variables to class of factor
pima[sapply(pima, is.character)] <- lapply(pima[sapply(pima, is.character)], as.factor)

#Split both datasets into an 80:20 split using a seed
set.seed(1, sample.kind="Rounding")
sample_size<- floor(0.2*nrow(pima))
test_index <- sample(seq_len(nrow(pima)), size=sample_size)
pima_train<- pima[-test_index,]
pima_test<- pima[test_index,]


## Logistic Regression
#Set the plots to a 2x2
par(mfrow = c(2,2))
#Calculate the deviance residuals, coefficients, and significances
pima_train$Outcome<- as.factor(pima_train$Outcome)
mod_glm <- glm(Outcome ~ ., data = pima_train, family = binomial(link = "logit"))
summary(mod_glm)
plot(mod_glm)

#Optimised logistic regression with significant parameters
par(mfrow = c(2,2))
mod_glm2 <- glm(Outcome~ Pregnancies+Glucose+BloodPressure+BMI + DiabetesPedigreeFunction, data=pima_train, family=binomial(link= "logit"))
summary(mod_glm2)
plot(mod_glm2)


#Predicting outcomes of Test data
pima_test$Outcome<- as.factor(pima_test$Outcome)
pred_glm <- predict(mod_glm2,pima_test, type = "response")
pred_glm <- ifelse(pred_glm <= 0.5, 1, 2)
pred_glm<- as.factor(pred_glm)
cm_glm<- confusionMatrix(pred_glm,pima_test$Outcome)
cm_glm


## k-Nearest Neighbors
#Develop a model to test the accuracy on the number of neighbours
mod_knn <- train(Outcome ~ ., data= pima_test, method = "knn", tuneGrid = data.frame(k = seq(1,20,1)))
mod_knn %>% ggplot()+geom_line(aes(x=k, y=Accuracy)) + labs(title= "Change in Regression Accuracy with varying optimal kNN")

#Optimal K
mod_knn$bestTune

#Predicting outcomes of Test data
pred_knn <- predict(mod_knn, pima_test, type="raw")
cm_knn<- confusionMatrix(pred_knn,pima_test$Outcome)
cm_knn


## Random Forest
#Develop a random forest model to produce a confusion matrix
mod_rf <- train(Outcome ~ ., method = "rf", data = pima_train)
pred_rf <- predict(mod_rf, pima_test)
pima_test$Outcome<- as.factor(pima_test$Outcome)
cm_rf<- confusionMatrix(pred_rf,pima_test$Outcome)
cm_rf


## XGBoost
# Develop an XGB model to produce a confusion matrix
par(mfrow = c(2,1))
mod_xgb <- train(Outcome ~ ., method = "xgbTree", data = pima_test)
plot(mod_xgb)
pred_xgb <- predict(mod_xgb, pima_test)
cm_xgb<- confusionMatrix(pred_xgb,pima_test$Outcome)
cm_xgb


## Results
# Create a knitr table of the regression approaches and their values
cm<- data.frame(c("General Logistic Model", "k Nearest Neighbours", "Random Forest", "XGBoost"), c(cm_glm$byClass[1], cm_knn$byClass[1], cm_rf$byClass[1], cm_xgb$byClass[1]), c(cm_glm$byClass[2], cm_knn$byClass[2], cm_rf$byClass[2], cm_xgb$byClass[2]), c(cm_glm$byClass[11], cm_knn$byClass[11], cm_rf$byClass[11], cm_xgb$byClass[11]))
cm<- as_tibble(cm)
colnames(cm) <- c("Regression", "Sensitivity", "Specificity", "Balanced Accuracy")
cm %>% knitr::kable()
