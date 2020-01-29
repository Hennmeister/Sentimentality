# SentimentAnalysis
An ensemble classifier designed to detect the sentiment of short social media texts.
Built to use sentiment score as a feature for a model that predicts the number of upvotes a reddit post
will receive: https://github.com/Hennmeister/FreeGold


## Model Success (with current parameters):
>Training set consisted of ~1.3 million documents, with testing set of ~150,000 tweets. Both classes were evenly represented.

### Accuracy : 79.55 %

### Confusion Matrix:

|        |   pos  |  neg  |
|:-----: | :-----:| :-----|
|   pos  |62317  | 14948  |
|   neg  | 17765   | 65915 |

### Classification Report:

|                | Precision      | Recall         | f1-score      |    Support     |
| :-------------:|:-------------: |:-------------:| :-------------:| :-------------:|
| neg            |      0.81      |        0.78     |      0.79    |    80082      |
| pos            |      0.79      |        0.81     |      0.80    |    79863      |
|                 |                |               |               |               |
| accuracy       |                  |            |      0.80    |    159945      |
| macro avg      |      0.80      |        0.80     |      0.80    |    159945      |
| weighted avg   |      0.80      |        0.80     |      0.80    |    159945      |

## Scores on Other Domains
As the model is being used for reddit text, it is important to assess how well the classifier
performs on domains other than twitter. Early testing of 250 self-labelled r/AskReddit posts
has found a score of:
1) 70.33% on posts deemed not neutral
2) 65.17% on all posts
Of course, a larger testing dataset is required

## Design:
Due to the lack of labelled reddit data, this model was trained on a twitter dataset
available here: http://help.sentiment140.com/for-students

A Bag of Words approach was taken to represent the corpus as features. The
data was first cleaned by removing punctuation, capitilization and all
(NLTK) stopwords. Then, Part of Speech tagging was applied to the remaining
words and only adjectives, adverbs, and verbs were left and converted back
into a string.

Next, sklearn's CountVectorizer was used to create a BOW volcabulary and vectorize
remaining features. In order to account for negation, both unigram and bigrams were used.

Before traning, two methods of feature extraction were applied:

1) Chi2 test was done to determine features not strongly associated with either class
(pos or neg).

2) Sklearn's TfidfTransformer was used to scale down the importance of features
that appear frequently in the corpus and are thus likely less informative

Finally, the actual model used was a voting classifier consisting of 5 base classifiers:
Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, SVM, Random Forest.

These classifers are among the most popular for sentiment analysis, as per the following
literature:
1) http://ceur-ws.org/Vol-2145/p26.pdf
2) https://www.sciencedirect.com/science/article/pii/S0167923614001997
3) https://pdfs.semanticscholar.org/aa3d/afab5bd4112b3f55929582bfec48139ff4c3.pdf

These base classifiers each predict the class, and also give a confidence
(probability) score. The classifier averages the probabilities of each classifer and
determines an overall prediction.

## Next Steps
1) Testing the classifier on labelled data
       from other domains (specifically, other social media sites) to test how
       well the model classifies sentiment outside of the twitter domain
2)  Determining cutoffs in the confidence score to predict neutal /
        conflicting sentiments
