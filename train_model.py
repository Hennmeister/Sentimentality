from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from clean_data import prepare_data
import pickle

def pickle_clf(clf, data_x, data_y):
    pickle.dump(clf, open("sentiment_classifier.pickle", 'wb'))
    pickle.dump(data_x, open("Data/Test/data_x", 'wb'))
    pickle.dump(data_y, open("Data/Test/data_y", 'wb'))


def train_classifier():
    data_x = []
    data_y = []
    num_pos = num_neg = 0
    with open('Data/Tweets/cleaned_negTweetsNouns', encoding="ISO-8859-1",
              mode='r') as output_neg:
        line = output_neg.readline().rstrip()
        while line and num_neg < 4000:
            data_x.append(line)
            data_y.append(0)
            line = output_neg.readline()
            num_neg += 1
    with open('Data/Tweets/cleaned_posTweetsNouns', mode='r',
              encoding="ISO-8859-1") as output_pos:
        line = output_pos.readline()
        while line and num_pos < 4000:
            data_x.append(line)
            data_y.append(1)
            line = output_pos.readline()
            num_pos += 1

    print('num_pos: ', num_pos, 'num_neg: ', num_neg)
    data_x_train, data_x_test, data_y_train, data_y_test = \
        train_test_split(data_x, data_y, train_size=(0.9), random_state=14)

    print(len(data_x_train))

    vectorizer = TfidfVectorizer(analyzer="word",
                                 max_df=0.9,
                                 min_df=4,
                                 ngram_range=(1, 2),
                               #  max_features=50000,
                                 )

    print("Done Vectorizing")
    # select_k_best_clf = SelectKBest(chi2, k=75000)\
    select_k_best_clf = SelectKBest(chi2, k=2000)
    print("Done feature selection")

    from trinary_voting_classifier import TrinaryVotingClassifier
    ensemble_classifier = TrinaryVotingClassifier(
        [('lsvc', CalibratedClassifierCV(LinearSVC())),
         ('mnb', MultinomialNB()),
         ('r', RandomForestClassifier(n_estimators=100)),
         ('lgrg', LogisticRegressionCV(random_state=1, max_iter=50000)),
         ('bnb', BernoulliNB())], voting='soft')

    # ensemble_classifier = VotingClassifier(
    #     [('lsvc', CalibratedClassifierCV(LinearSVC())),
    #      ('mnb', MultinomialNB()),
    #      ('r', RandomForestClassifier(n_estimators=100)),
    #      ('lgrg', LogisticRegressionCV(random_state=1, max_iter=50000)),
    #      ('bnb', BernoulliNB())], voting='soft')

    steps = [('vectorizer', vectorizer), ('select', select_k_best_clf), ('clf', ensemble_classifier)]
    classifier_pipeline = Pipeline(steps)

    classifier_pipeline = classifier_pipeline.fit(data_x_train, data_y_train)
    print('clf score', classifier_pipeline.score(data_x_test, data_y_test))

    print("Done training classifier")

    y_predicted = classifier_pipeline.predict(data_x_test)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_predicted, data_y_test))

    from sklearn.metrics import classification_report
    print(classification_report(data_y_test, y_predicted))

    # pickle_clf(classifier_pipeline, data_x_test, data_y_test)

if __name__ == '__main__':
    # prepare_data()
    train_classifier()

