import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import random
import pickle
import time

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
    save_documents = open("Data/Pickles/documents.pickle", "wb")
    pickle.dump(documents, save_documents)
    save_documents.close()

    save_all_words = open("Data/Pickles/all_words.pickle", "wb")
    pickle.dump(all_words, save_all_words)
    save_all_words.close()


def open_document_pickle():
    doc_pickle = open("Data/Pickles/documents.pickle", "rb")
    documents = pickle.load(doc_pickle)
    doc_pickle.close()
    return documents


def open_all_words_pickle():
    words_pickle = open("Data/Pickles/all_words.pickle", "rb")
    all_words = pickle.load(words_pickle)
    words_pickle.close()
    return all_words


if __name__ == "__main__":
    # get_data(open('Data/negTweets', encoding="ISO-8859-1"), 'neg')
    # get_data(open('Data/posTweets', encoding="ISO-8859-1"), 'pos')
    documents = open_document_pickle()
    all_words = open_all_words_pickle()


    # feature_list = create_feature_list()
    # feature_list_pickle = open("Data/Pickles/feature_list.pickle", "rb")
    # feature_list = pickle.load(feature_list_pickle)
    # feature_list_pickle.close()

    # data_features = [(get_features(tweet, feature_list), category) for
    #                  (tweet, category) in documents]

    load_data_features = open("Data/Pickles/data_features.pickle", "rb")
    data_features = pickle.load(load_data_features)
    load_data_features.close()

    random.shuffle(data_features)
    training_set = data_features[:5000]
    # training_set = data_features[:int(NUM_FEATURES * .9)]
    # testing_set = data_features[int(NUM_FEATURES * .9):]

    # == COMPLETE == #
    # NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
    # save_NB_classifier = open("Data/Pickles/NB_classifier.pickle", "wb")
    # pickle.dump(NB_classifier, save_NB_classifier)

    # MNB_classifier = SklearnClassifier(MultinomialNB()).train(training_set)
    # save_MNB_classifier = open("Data/Pickles/MNB_classifier.pickle", "wb")
    # pickle.dump(MNB_classifier, save_MNB_classifier)

    # BernoulliNB_classifier = SklearnClassifier(BernoulliNB()) \
    #     .train(training_set)
    # save_BNB_classifier = open("Data/Pickles/BNB_classifier.pickle", "wb")
    # pickle.dump(BernoulliNB_classifier, save_BNB_classifier)

    # LogisticRegression_classifier = SklearnClassifier(LogisticRegression()) \
    #     .train(training_set)
    # save_LogReg_classifier = open("Data/Pickles/LR_classifier.pickle", "wb")
    # pickle.dump(LogisticRegression_classifier, save_LogReg_classifier)

    # SGDClassifier_classifier = SklearnClassifier(SGDClassifier()) \
    #     .train(training_set)
    # save_SGD_classifier = open("Data/Pickles/SGD_classifier.pickle", "wb")
    # pickle.dump(SGDClassifier_classifier, save_SGD_classifier)
    #
    # LinearSVC_classifier = SklearnClassifier(LinearSVC()).train(training_set)
    # save_LSVC_classifier = open("Data/Pickles/LSVC_classifier.pickle", "wb")
    # pickle.dump(LinearSVC_classifier, save_LSVC_classifier)

    # NuSVC_classifier = SklearnClassifier(NuSVC()).train(training_set)
    # save_NuSVC_classifier = open("Data/Pickles/NuSVC_classifier.pickle", "wb")
    # pickle.dump(NuSVC_classifier, save_NuSVC_classifier)

    NB_classifier = pickle.load(open("Data/Pickles/NB_classifier.pickle", "rb"))
    MNB_classifier = pickle.load(open("Data/Pickles/MNB_classifier.pickle", "rb"))
    BernoulliNB_classifier = pickle.load(open("Data/Pickles/BNB_classifier.pickle", "rb"))
    LogisticRegression_classifier = pickle.load(open("Data/Pickles/LR_classifier.pickle", "rb"))
    SGDClassifier_classifier = pickle.load(open("Data/Pickles/SGD_classifier.pickle", "rb"))
    LinearSVC_classifier = pickle.load(open("Data/Pickles/LSVC_classifier.pickle", "rb"))
    NuSVC_classifier = pickle.load(open("Data/Pickles/NuSVC_classifier.pickle", "rb"))

    agg_classifier = AggregateClassifier(NB_classifier, MNB_classifier,
                                         BernoulliNB_classifier,
                                         LogisticRegression_classifier,
                                         SGDClassifier_classifier,
                                         # SVC_classifier,
                                         LinearSVC_classifier,
                                         NuSVC_classifier)

    print("aggregate accuracy: ", (nltk.classify.accuracy(agg_classifier,
                                                          training_set)))
    print("Confidence: ", agg_classifier.recent_confidence_score())

    # save_classifier = open("agg_classifier.pickle", "wb")
    # pickle.dump(agg_classifier, save_pickles())
    # save_classifier.close()
