# SentimentAnalysis
A ensemble voting classifier to detect sentiment in short social media texts.
This model is intended to be used to produce sentiment scores as a feature for
another machine-learning project with the end goal to predict the number of
upvotes a reddit post will get.

Due to the lack of labelled reddit data, this model was trained on a twitter dataset
available here: http://help.sentiment140.com/for-students

Model Success (with current parameters):





Design:
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
        1) Multinomial Naive Bayes
        2) Bernoulli Naive bayes
        3) Logistic Regression
        4) SVM
        5) Random Forest
    These classifers are among the most popular for sentiment analysis, as per the following
    literature:
    http://ceur-ws.org/Vol-2145/p26.pdf
    https://www.sciencedirect.com/science/article/pii/S0167923614001997
    https://pdfs.semanticscholar.org/aa3d/afab5bd4112b3f55929582bfec48139ff4c3.pdf

    These base classifiers each predict the class, and also give a confidence
    (probability) score. The classifier averages the probabilities of each classifer and
    determines an overall prediction.

    There are two major next steps for this project:

        1) Testing the classifier on labelled data
           from other domains (specifically, other social media sites) to test how
           well the model classifies sentiment outside of the twitter domain
        2)  Determining cutoffs in the confidence score to predict neutal /
            conflicting sentiments
