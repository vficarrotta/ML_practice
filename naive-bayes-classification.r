## kNN clustering
## vficarrotta
## machine learning in R
## data from: https://github.com/PacktPublishing/Machine-Learning-with-R-Fourth-Edition/tree/main

## dir
setwd('')

## Libraries
library(tm); library(SnowballC); library(wordcloud); library(e1071); library(gmodels)

## Data
sms_raw <- read.csv('sms_spam.csv', stringsAsFactors=F)
str(sms_raw)

colnames(sms_raw) <- c('type', 'text')

sms_raw$type <- factor(sms_raw$type)

table(sms_raw$type)

## Create Corpus (list of text messages in the form of NLP object) or DOCUMENT
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

lapply(sms_corpus[1:2], as.character)

## Create the tm map that reduces text to all instances of the same word EX: love, LOVE, etc
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

## Remove numbers from text
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# browse built-in transformations in tm package using
# getTransformations()

## Remove filler words (Stop Words) EX: and, but, or, to
# see list of stop words: stopwords()
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

## Remove punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

## Stemming: reducing words to their root form EX learned, learning, learns = learn
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

## Remove white space
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

## Tokenization
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

#####
# Performing all cleaning steps using DTM function
#####
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control=list(tolower=T, removeNumbers=T, stopwords=T, removePunctuation=T, stemming=T))

# differences between dtm and dtm2 are due to the order of processing:
# dtm2 has cleaned words before tokenization and so are split differently
# to force dtm2 into identical state with dtm, use:
# stopwords=function(x) {removeWords(x, stopwords())}
# in DocumentTermMatrix


## Training and Test sets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

## wordcloud
wordcloud(sms_corpus_clean, min.freq=50, random.order=F)

spam <- subset(sms_raw, type == 'spam')
ham <- subset(sms_raw, type == 'ham')

wordcloud(spam$text, max.words=40, scale=c(3, 0.5))
wordcloud(ham$text, max.words=40, scale=c(3, 0.5))

## Indicator Features
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_words]

## convert the counts in train and test matrices to yes or no 
# count strings

convert_counts <- function(x){
    x <- ifelse(x>0, 'Yes', 'No')
}

sms_train <- apply(sms_dtm_freq_train,2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, 2, convert_counts)

## Build naive bayes model
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

## Evaluate model performance
sms_test_pred <- predict(sms_classifier, sms_test)

CrossTable(sms_test_pred, sms_test_labels, prop.chisq=F, prop.c=F, prop.r=F, dnn=c('predicted', 'actual'))

## Improve the model performance using Laplace estimator
## Laplace estimator accounts for multiplication of 
# probabilities where the probability is 0 because 
# it never occurs and therefore causes an output of 
# 0 where it should be a small probability.
# Laplace estimator uses a 1 instead of 0 for 
# these scenarios.

sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace=1)

sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_test_labels, prop.chisq=F, prop.c=F, prop.r=F, dnn=c('predicted', 'actual')))
