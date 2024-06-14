## kNN clustering
## vficarrotta
## machine learning in R
## data from: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

## dir
setwd('C:/Users/Vince/Documents/Machine Learning in R/CH4')

## Libraries
library(tm); library(SnowballC); library(wordcloud)

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
sms_dtm_train <- sms_dtm[1:4169]
sms_dtm_test <- sms_dtm[4170:length(sms_dtm)]

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:length(sms_raw), ]$type

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

## wordcloud
wordcloud(sms_corpus_clean, min.freq=50, random.order=F)

spam <- subset(sms_raw, type == 'spam')
ham <- subset(sms_raw, type == 'ham')

wordcloud(spam$text, max.words=40, scale=c(3, 0.5))
wordcloud(ham$text, max.words=40, scale=c(3, 0.5))


