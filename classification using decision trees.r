## classification using decision trees
## vficarrotta 2024
## 

## Directory
setwd('')

#################################
# Part 1: Decision Trees
####
# Divide and Conquer

##
library(C50); library(gmodels)

## data
credit <- read.csv('credit.csv')

str(credit)

table(credit$checking_balance)
table(credit$savings_balance)

summary(credit$months_loan_duration)
summary(credit$amount)

table(credit$default)

## training and test sets (90% train, 10% test)
RNGversion('3.5.2'); set.seed(123)

train_sample <- sample(1000, 900)
str(train_sample)

credit_train <- credit[train_sample, ]
credit_test <- credit[-train_sample, ]

prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

## training model
credit_model <- C5.0(x=credit_train[-17], y=as.factor(credit_train$default))
credit_model

summary(credit_model)

## model performance
credit_pred <- predict(credit_model, credit_test)

CrossTable(credit_test$default, credit_pred, prop.chisq=F, prop.c=F, prop.r=F, dnn=c('actual default', 'predicted default'))

####
# Improving model performance
####

## Boosting - splitting the decision trees into smaller, specialized units
# and combining their votes on outcomes
credit_boost10 <- C5.0(x=credit_train[-17], y=as.factor(credit_train$default), trials=10)
credit_boost10

summary(credit_boost10)

credit_boost_pred10 <- predict(credit_boost10, credit_test)

CrossTable(credit_test$default, credit_boost_pred10, prop.chisq=F, prop.c=F, prop.r=F, dnn=c('actual default', 'predicted default'))

## Penalizing matrix - penalizing certain decisions to influence decisions
matrix_dimensions <- list(c('no', 'yes'), c('no', 'yes'))
names(matrix_dimensions) <- c('predicted', 'actual')
error_cost <- matrix(c(0, 1, 4, 0), nrow=2)

credit_cost <- C5.0(credit_train[-17], as.factor(credit_train$default), costs=error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred, prop.chisq=F, prop.c=F, prop.r=F, dnn=c('actual default','predicted default'))

## The final model with error costs performs worse than the boosted model
# however the percent of actual defaults is less than the boosted model 
# therefore saving the bank more money by avoiding the costly mistakes.


#################################
# Part 2: Rule Learners
####
# Separate and Conquer

## Libraries
library(OneR); library(RWeka)

## Data
mushrooms <- read.csv('mushrooms.csv', stringsAsFactors=T)

mushrooms$veil_type <- NULL

table(mushrooms$type)

## Not building a predictive model, instead a classifier 
# across the whole known dataset.

####
# 1R classifier
####
mushroom_1R <- OneR(type ~ ., data=mushrooms)

mushroom_1R

## Evaluate the performance
# build the confusion table from running the model back across the data
mushroom_1R_pred <- predict(mushroom_1R, mushrooms)
table(actual=mushrooms$type, predicted=mushroom_1R_pred)

## some poisonous mushrooms incorrectly labeled edible: costly mistake

####
## RIPPER: Improving model performance
####
mushroom_JRip <- JRip(type ~ ., data=mushrooms)

mushroom_JRip_pred <- predict(mushroom_JRip, mushrooms)
table(actual=mushrooms$type, predicted=mushroom_JRip_pred)

## 100% of mushrooms properly identifited as either edible or poisonous
