## kNN clustering
## vficarrotta
## machine learning in R
## data from: https://github.com/PacktPublishing/Machine-Learning-with-R-Fourth-Edition/



## Necessaries
setwd('~/Machine Learning in R/CH3')

## Libraries
library('class'); library('gmodels')


## Data
wbcd <- read.csv('wisc_bc_data.csv')

## Explore and Wrangle
str(wbcd)
table(wbcd$diagnosis)

# strip ids
wbcd <- wbcd[-1]

# clean diagnostic labels
wbcd$diagnosis <- factor(wbcd$diagnosis, levels=c('B', 'M'), labels=c('Benign', 'Malignant'))

round(prop.table(table(wbcd$diagnosis)) * 100, digits=1)

## summarize
summary(wbcd[c('radius_mean', 'area_mean', 'smoothness_mean')])
    ## must scale values due to the area_mean variable's range

## normalize data
normalize <- function(x){
    return( (x-min(x)) / (max(x) - min(x)) )
}

wbcd_n <- as.data.frame(lapply(wbcd[2:31], FUN=normalize))

summary(wbcd_n$area_mean)
plot(density(wbcd_n$area_mean))


## Train and Test sets
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

## Training
wbcd_test_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=21)

## Training Performance
CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=F)

####
# Alternative k values, normalized
####

kk <- rep(1:50)

perf <- list()
for(i in kk){
    wbcd_test_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=i)
    perf[[i]] <- CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=F)[1]
}

perf_clean <- data.frame(kk)
for(i in kk){
    perf_clean[i,2] <- perf[i][[1]][[1]][1,2]
    perf_clean[i,3] <- perf[i][[1]][[1]][2,1]
}
colnames(perf_clean) <- c('k', 'False Postive', 'False Negative')

perf_clean

perf_clean[,length(perf_clean)+1] <- perf_clean[,2]+perf_clean[,3]

kvals <- which(perf_clean[,4]==min(perf_clean[,4]))


####
# z-score standardization
####

wbcd_z <- as.data.frame(scale(wbcd[-1]))

summary(wbcd_z$area_mean)

## Training and Test Sets
wbcd_train <- wbcd_z[1:469, ]
wbcd_test <- wbcd_z[470:569, ]

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

## Training
wbcd_test_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=21)

## Training Performance
CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=F)

####
# Alternative k values, z-score
####

kk <- rep(1:50)

perf <- list()
for(i in kk){
    wbcd_test_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=i)
    perf[[i]] <- CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=F)[1]
}

perf_clean <- data.frame(kk)
for(i in kk){
    perf_clean[i,2] <- perf[i][[1]][[1]][1,2]
    perf_clean[i,3] <- perf[i][[1]][[1]][2,1]
}
colnames(perf_clean) <- c('k', 'False Postive', 'False Negative')

perf_clean

perf_clean[,length(perf_clean)+1] <- perf_clean[,2]+perf_clean[,3]

kvals <- which(perf_clean[,4]==min(perf_clean[,4]))


