from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier

from nltk.classify import ClassifierI
from statistics import mode


class AggregateClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self._votes = []

    def classify(self, featureset):
        self._votes = []
        for c in self._classifiers:
            self._votes.append(c.classify(featureset))
        return mode(self._votes)

    def recent_confidence_score(self):
        num_votes = self._votes.count(mode(self._votes))
        return num_votes / len(self._votes)
