# SentimentAnalysis
featureset:  308.25088715553284
shuffling and splicing 0.9977674484253
Training NB:  67.19093823432922


30,000 Testing Tweets:
Agg_model:
    aggregate accuracy:  0.7233666666666667
    Confidence:  0.8984428571427412

    75% with twitter feature list (on 1000 tweets)
    .75.4% (1000 tweets) 75.4939% (15000 tweets) deemed not neutral by VADER
Vader:
    0.5219333333333334 (neutrals making it a lot worse, also should split sentence)
    0.71, 0.7069 on neutrals - DAMN


Plans:
-Label reddit data
-test to see how vader does
-test agg model, leaving ones labelled neutral
-test vader on ability to detect neutral (classify only neutrals)

If vader is better at all, use that
Otherwise, could use vader to sort out neutrals and then use combination

IF decide to just label pos -> T or F and neg -> T or F:
    -Build model that optimizes Threshold point for vader sentiments (pos neg and neutral)
    for when to deem something too mild
    -In that case, use agg_model to predict
    -Test if on the non-neutrals, agg_model performs better than vader
    -Want to be as context-specific as possible, so use self-labelled reddit data



REAL PLAN:


POS-tagging
Negation-Prefixing
https://kenbenoit.net/pdfs/NDATAD2013/Rice-Zorn-LSE-V01.pdf

1) Use various popular base classifiers:
    Multinomial Naive Bayes
    Bernoulli Naive bayes
    Logistic Regression
    SVM
    Random Forest

    http://ceur-ws.org/Vol-2145/p26.pdf
    https://www.sciencedirect.com/science/article/pii/S0167923614001997
    https://pdfs.semanticscholar.org/aa3d/afab5bd4112b3f55929582bfec48139ff4c3.pdf
    https://www.researchgate.net/publication/268509189_Sentiment_Mining_of_Movie_Reviews_using_Random_Forest_with_Tuned_Hyperparameters

2) Train Ensemble.VotingClassifier on Twitter Data

3) Test it on reddit data

4) Decide neutral cutoff / Combine with vader for cutoff

Pipeline:
 manually
    remove punctuation and stopwords w/ nltk
    POS tagging - J R and V
    Stopwords
 sklearn
   CountVectorizer (5, .8) --> chi2 --> TfidfTransformer --> Ensemble
   Confusion matrix, classification report
   pickle dat

Use only adjective and adverb
https://link.springer.com/chapter/10.1007/978-3-030-04284-4_13
