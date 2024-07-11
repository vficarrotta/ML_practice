## Market Basket Analysis usning association rules
## vficarrotta 2024
## 

## Dir
setwd('')

## Library
library(arules)

## Data
read.csv('groceries.csv')

groceries <- read.transactions('groceries.csv', sep=',')
summary(groceries)

## Visualizing item support

    # plots items with less than 10% support
itemFrequencyPlot(groceries, support=0.1)

    # plots the top 20 purchased items
itemFrequencyPlot(groceries, topN=20)

## Visualizing transaction data
    # rows are the transaction number and columns are the item
image(groceries[1:5])

image(sample(groceries, 100))

## Training a model: apriori algorithm
groceryrules <- apriori(groceries, parameter=list(support=0.006, confidence=0.25, minlen=2))
summary(groceryrules)

## Evaluating model performance

# actionable rules provide useful insights
# trivial rules are not useful
# inexplicable rules are mysterious and not (typically) useful

inspect(groceryrules[1:3])
    #          lhs               rhs               support  confidence  coverage
    # [1] {potted plants} => {whole milk}      0.006914082 0.4000000  0.01728521
    # if potted plants is purchased then whole milk is purchased. this happens in about 0.7% of transactions in 40% of potted plant purchases

## Improving model performance
    # rules can besorted by support confidence or lift
inspect(sort(groceryrules, by='lift')[1:5])

berryrules <- subset(groceryrules, items %in% 'berries')
inspect(berryrules)



