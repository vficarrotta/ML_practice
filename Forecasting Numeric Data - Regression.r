## Forecasting Numeric Data - Regression Models
## vficarrotta 2024
## Chapter 6 Exercises

## Libraries
library('psych'); library('rpart'); library('rpart.plot'); library(Cubist)

## Directory
setwd('C:/Users/Brin/Documents/R ML/ML in R')

## Data
insurance <- read.csv('insurance.csv')
str(insurance)

    # expenses column is dependent variable

summary(insurance$expenses)
hist(insurance$expenses)

## Correlation matrix
cor(insurance[c('age', 'bmi', 'children', 'expenses')])

## SPLOM: scatter plot matrix
pairs(insurance[c('age', 'bmi', 'children','expenses')])

pairs.panels(insurance[c('age', 'bmi', 'children','expenses')])
    # ellipses indicate correlational strength: more ovular is stronger correlation

## create model
ins_model <- lm(expenses ~ age + children + bmi + sex + smoker + region, data=insurance)

    ## shorthand version
    # ins_model <- lm(expenses ~ ., data=insurance)
ins_model
summary(ins_model)

## Reading the summary
# residuals with a max of near 30K means underprediction for at least one individual
# half of errors fall withing 1Q adn 3Q with 3k over true value and 2k under true value

####
# Improving the model
####

# 1)
## adding non-linear relationship
# generate a squared term out of age variable
insurance$age2 <- insurance$age^2

# 2)
## convert numeric variable to binary indicator
# one-hot coding bmi from numeric values to 
# 0 for under 30 and 1 for over 30
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

# 3) 
## interactions: combined effect of two or more independent 
# variables is higher than seperate effects
# ex: bmi and smoking together

ins_model2 <- lm(expenses ~ age + age2 + children + bmi + sex + bmi30*smoker + region, data=insurance)
summary(ins_model2)

####
# Predictions
####
insurance$pred <- predict(ins_model2, insurance)

# compare model performance with simple pearsons correlation
cor(insurance$pred, insurance$expenses)

plot(insurance$pred, insurance$expenses)
abline(a=0, b=1, col='red', lwd=3, lty=2)

## forecasting prices

# males age 30
predict(ins_model2, data.frame(
    age=30, 
    age2=30^2,
    children=2,
    bmi=30,
    sex='male', 
    bmi30=1, 
    smoker='no', 
    region='northeast' 
))

# females age 30
predict(ins_model2, data.frame(
    age=30, 
    age2=30^2,
    children=2,
    bmi=30,
    sex='female', 
    bmi30=1, 
    smoker='no', 
    region='northeast' 
))


#####
# Part2: regression trees and model trees
#####

wine <- read.csv('whitewines.csv')
str(wine)

hist(wine$quality)

wine_train <- wine[1:3750,]
wine_test <- wine[3751:4898,]

####
# Training the model: CART-Classification and Regression Tree
####

m.rpart <- rpart(quality ~ ., data=wine_train)
summary(rpart)

rpart.plot(m.rpart, digits=4, fallen.leaves=T, type=3, extra=101)

## model performance check
p.rpart <- predict(m.rpart, wine_test)
summary(p.rpart)
summary(wine_test$quality)

cor(p.rpart, wine_test$quality)

## Mean Absolute Error: measuring performance
MAE <- function(actual, predicted){
    mean(abs(actual - predicted))
}

MAE(p.rpart, wine_test$quality)
    # model isnt super impressive but performs well enough here
mean(wine_train$quality)
    # if prediction value of 5.87 for every wine sample, we would have
    # a mean absolute error of only about 0.67

MAE(5.87, wine_test$quality) 

####
# Improving the model performance
####
m.cubist <- cubist(x=wine_train[-12], wine_train$quality)
m.cubist
summary(m.cubist)

p.cubist <- predict(m.cubist, wine_test)
summary(p.cubist)

cor(p.cubist, wine_test$quality)
    # better than 5.87
MAE(wine_test$quality, p.cubist)





