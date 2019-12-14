import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import random
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
from AggregateClassifier import AggregateClassifier

# Config options
NUM_TWEETS = 50000
NUM_FEATURES = 7500

all_words = []
documents = []
allowed_word_types = ["J", "R"]


def get_data(text, sentiment) -> list:
    tweet = text.readline()
    i = 0
    while tweet and i < NUM_TWEETS:
        # Store raw tweet with associated sentiment (positive or negative)
        documents.append((tweet, sentiment))

        # Remove punctuation and words that start with given punctuation
        # Relevant for the @TwitterHandles - Zike doesn't remove those,
        # shouldn't be an issue cause they wont be part of top x frequencies
        translator = str.maketrans('', '', string.punctuation)
        tweet = re.sub("@.+?&", "", tweet.translate(translator))

        # Tokenize and remove stopwords
        stop_words = list(set(stopwords.words('english')))
        words = [w for w in word_tokenize(tweet) if w not in stop_words]

        # POS tagging - only looking at adjectives for now
        pos = nltk.pos_tag(words)
        for w in pos:
            # Checking if the first letter in tag is J,
            # signifying type of adjective
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
        tweet = text.readline()
        i += 1


def create_feature_list():
    word_frequencies = nltk.FreqDist(all_words)
    # Most frequent words will be used as features
    return [(w[0], w[1]) for w in sorted(word_frequencies.items(),
                                         key=lambda freq: freq[1],
                                         reverse=True)[:NUM_FEATURES]]


def get_features(document, feature_list):
    words = word_tokenize(document)
    features = {}
    for w in feature_list:
        features[w[0]] = (w[0] in words)
    return features


def save_pickles():
    save_documents = open("documents.pickle", "wb")
    pickle.dump(documents, save_documents)
    save_documents.close()

    save_all_words = open("all_words.pickle", "wb")
    pickle.dump(all_words, save_all_words)
    save_all_words.close()


def open_document_pickle():
    doc_pickle = open("documents.pickle", "rb")
    documents = pickle.load(doc_pickle)
    doc_pickle.close()
    return documents


def open_all_words_pickle():
    words_pickle = open("all_words.pickle", "rb")
    all_words = pickle.load(words_pickle)
    words_pickle.close()
    return all_words


if __name__ == "__main__":
    get_data(open('Data/negTweets', encoding="ISO-8859-1"), 'neg')
    get_data(open('Data/posTweets', encoding="ISO-8859-1"), 'pos')
    # documents = open_document_pickle()
    # all_words = open_all_words_pickle()
    feature_list = create_feature_list()
    data_features = [(get_features(tweet, feature_list), category) for
                     (tweet, category) in documents]
    random.shuffle(data_features)
    training_set = data_features[:int(NUM_FEATURES * .9)]
    testing_set = data_features[int(NUM_FEATURES * .9):]

    NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    MNB_classifier = SklearnClassifier(MultinomialNB()).train(training_set)
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB()) \
        .train(training_set)
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression()) \
        .train(training_set)
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier()) \
        .train(training_set)
    # SVC_classifier = SklearnClassifier(SVC()).train(training_set)
    LinearSVC_classifier = SklearnClassifier(LinearSVC()).train(training_set)
    NuSVC_classifier = SklearnClassifier(NuSVC()).train(training_set)

    agg_classifier = AggregateClassifier(NB_classifier, MNB_classifier,
                                         BernoulliNB_classifier,
                                         LogisticRegression_classifier,
                                         SGDClassifier_classifier,
                                         # SVC_classifier,
                                         LinearSVC_classifier,
                                         NuSVC_classifier)

    print("aggregate accuracy: ", (nltk.classify.accuracy(agg_classifier,
                                                          testing_set)))
    print("Confidence: ", agg_classifier.recent_confidence_score())

    save_classifier = open("agg_classifier.pickle", "wb")
    pickle.dump(agg_classifier, save_pickles())
    save_classifier.close()
