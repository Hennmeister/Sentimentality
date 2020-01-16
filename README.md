# SentimentAnalysis
An ensemble classifier designed to detect the sentiment of short social media texts.



## Model Success (with current parameters):
>Trained on ~450,000  and tested on ~50,000 tweets, both with a 50/50 sentiment split.

Score: 78.12 %

Confusion Matrix:

|        |   pos  |  neg  |
|:-----: | :-----:| :-----|
|   pos  |220532  | 5689  |
|   neg  | 5935   | 21159 |

Classification Report:

|                | Precision      | Recall         | f1-score      |    Support     |
| :-------------:|:-------------: |:-------------:| :-------------:| :-------------:|
| neg            |      0.78      |        0.78     |      0.78    |    26567      |
| pos            |      0.78      |        0.79     |      0.78    |    26848      |
|                 |                |               |               |               |
| accuracy       |                  |            |      0.78    |    53315      |
| macro avg      |      0.78      |        0.78     |      0.78    |    53315      |
| weighted avg   |      0.78      |        0.78     |      0.78    |    53315      |





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

There are two major next steps for this project:
1) Testing the classifier on labelled data
       from other domains (specifically, other social media sites) to test how
       well the model classifies sentiment outside of the twitter domain
2)  Determining cutoffs in the confidence score to predict neutal /
        conflicting sentiments
