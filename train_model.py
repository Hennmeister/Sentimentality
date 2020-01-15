from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

import pickle

def train_classifier():
    vectorizer = CountVectorizer(analyzer="word", \
                                 max_df = 0.9, \
                                 min_df = 3, \
                                 # max_features=5000,
                                 )

    data_x = []
    data_y = []
    num_pos = num_neg = 0
    with open('Data/Tweets/cleaned_negTweets', encoding="ISO-8859-1", mode='r') as output_neg:
        line = output_neg.readline().rstrip()
        while line:
            data_x.append(line)
            data_y.append(0)
            line = output_neg.readline()
            num_neg += 1
    with open('Data/Tweets/cleaned_posTweets', mode='r', encoding="ISO-8859-1") as output_pos:
        line = output_pos.readline()
        while line:
            data_x.append(line)
            data_y.append(1)
            line = output_pos.readline()
            num_pos += 1

    print('num_pos: ', num_pos, 'num_neg: ', num_neg)
    data_x_train, data_x_test, data_y_train, data_y_test = \
        train_test_split(data_x, data_y, train_size=(0.8), random_state=14)

    data_x_train = vectorizer.fit_transform(data_x_train)
    data_x_test = vectorizer.transform(data_x_test)
    vectorizer.vocabulary_.get('good')

    print('datax_train shape', data_x_train.shape)
    data_x_train = SelectKBest(chi2, k='all').fit_transform(data_x_train, data_y_train)
    data_x_train = TfidfTransformer().fit_transform(data_x_train)
    print('datax_train shape', data_x_train.shape)

    ensemble_classifier = VotingClassifier([('lsvc', CalibratedClassifierCV(LinearSVC())),
                                            ('mnb', MultinomialNB()),
                                            ('r', RandomForestClassifier(n_estimators=100)),
                                            ('lgrg', LogisticRegressionCV(random_state=1, max_iter=50000)),
                                            ('bnb', BernoulliNB())], voting='soft')

    ensemble_classifier = ensemble_classifier.fit(data_x_train, data_y_train)
    print('clf score', ensemble_classifier.score(data_x_test, data_y_test))

    y_predicted = ensemble_classifier.predict(data_x_test)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_predicted, data_y_test))

    from sklearn.metrics import classification_report
    print(classification_report(data_y_test, y_predicted))

    pickle.dump(ensemble_classifier, open("ensemble_classifer.pickle", 'wb'))

if __name__ == '__main__':
    train_classifier()
